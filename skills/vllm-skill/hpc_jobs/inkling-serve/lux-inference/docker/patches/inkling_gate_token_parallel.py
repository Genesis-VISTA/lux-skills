"""
Inkling MoE router GEMM at large T: token-parallel over TP plus a dim-0
all-gather of the fp32 logits, instead of the full replicated GEMM on every
TP rank.

THE PROBLEM
InklingGate.weight ([264, 4096] bf16, 258 real rows) is replicated on every
TP rank, and so is the router input (mlp_in comes out of _sconv_add_norm's
all-gather + rmsnorm). Each rank computes the same [T, 264] fp32 logits:
TP8 does 8x the work for one result. inkling_gate_fp32_splitk.py covers
T <= 64 only; above that, torch.mm(out_dtype=float32) runs on hipBLASLt
Cijk_..._BSS_MT64x16x128 on MI250X. In the 1-node TP8 mixed refill step
(8192 tokens, steps7 job f61kDFNJfuA7) that takes 484 us per MoE layer,
~35 TF/s, or 9.2 ms per step over the stage's 19 MoE layers (more in the
MTP depths, which run the same layer at T=8192).

THE FIX
moe.py: when T >= INKLING_GATE_TOKEN_PARALLEL (a token count, default
4096; "0" turns the path off), TP rank r runs the same _linear_with_fp32_out
on its contiguous ceil(T/tp) row block, and a dim-0 all-gather (rank-ordered,
no transpose copy) reassembles the [T, 264] fp32 logits. Gate weight,
loader, selection kernel and the MoERunner call are unchanged; the gathered
tensor is the router_logits the layer stashes and passes on. The default
threshold is the measured eager crossover (gtp_bench, job f61mEquyX65u,
us eager/graph, replicated vs token-parallel: 177/128 vs 222/89 at T=2048,
298/249 vs 257/146 at 4096, 528/479 vs 293/211 at 8192, 989/940 vs 413/338
at 16384); T >= 4096 only occurs in eager prefill/mixed steps.
The per-rank block (T/8 >= 512 rows) stays far above the split-K kernel's
64-token range, so both paths use the same GEMM family.

Exactness: each logit row is the same dot product over K=4096 on the same
weight, through the same _linear_with_fp32_out, only with M = ceil(T/tp)
instead of T. hipBLASLt picks its solution by problem size, and a
different solution can change the fp32 accumulation order. The same
row-split on the MTP input_proj (inkling_mtp_input_proj_tp.py, job
f61kuFbqi9HR) was bit-identical at T=1..64 and 4096 and within 1 bf16 ulp
elsewhere, the same batch-size variance the stock GEMM already has across
T. gtp_bench.py (real gate weight, random inputs): logits bit-identical at
T=512, 1024 and 2048..16384 (8191 included), within 2.4e-5 elsewhere, no top-6
routing row changed. On real hidden states a 1-ulp logit change can still
flip a near-tie: greedy graphs run (layers 0-20 + MTP, flag 2048 vs 0) was
64/64 token-identical at B=64 x 256 but 5/8 at B=8 x 128 (prefill-chunk
routing), so this is not a bit-exact patch.
Graph safety: the branch and the row block depend only on T, the TP size
and the rank, all fixed per captured graph (and the default threshold is
above the largest FULL_DECODE_ONLY capture). The all-gather is the same
collective the sconv already uses; there is no host sync.

Upstream status
---------------
Present on the pinned build 0.2.1.dev32+gbc0f47cd0 (vllm_py_src).
"""
import argparse
import importlib.util
import os
import sys

MARKER = "Zenith patch inkling_gate_token_parallel.py"

MOE = "vllm/models/inkling/amd/moe.py"

# (target, OLD, NEW): every OLD occurs exactly once in the unpatched file, and
# none overlaps inkling_gate_fp32_splitk.py (which replaces the head of
# _linear_with_fp32_out).
HUNKS = [
    (MOE,
     '''    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.kernels.linear.cute_dsl import ll_bf16
''',
     '''    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,  # Zenith patch inkling_gate_token_parallel.py
)
from vllm.model_executor.kernels.linear.cute_dsl import ll_bf16
'''),
    (MOE,
     '''

class InklingGate(nn.Module):
''',
     '''

# Zenith patch inkling_gate_token_parallel.py: the replicated router GEMM split
# by rows over TP plus a dim-0 all-gather of the fp32 logits, for T >= the
# threshold.
def _gate_tp_min_tokens() -> int:
    """INKLING_GATE_TOKEN_PARALLEL: smallest T that takes the token-parallel
    router GEMM (default 4096); 0 turns it off."""
    import os

    return max(0, int(os.environ.get("INKLING_GATE_TOKEN_PARALLEL", "4096") or 0))


def _gate_logits_token_parallel(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """_linear_with_fp32_out(x, weight) for a TP-replicated 2-D x: each rank
    runs it on its contiguous ceil(T/tp) row block, then a rank-ordered dim-0
    all-gather."""
    tp = get_tensor_model_parallel_world_size()
    t = x.shape[0]
    if tp == 1:
        return _linear_with_fp32_out(x, weight)
    rows = -(-t // tp)
    lo = min(get_tensor_model_parallel_rank() * rows, t)
    hi = min(lo + rows, t)
    if hi - lo == rows:
        part = _linear_with_fp32_out(x[lo:hi], weight)
    else:
        # Short or empty last blocks: rows past T are never read back.
        part = torch.empty(
            (rows, weight.shape[0]), dtype=torch.float32, device=x.device
        )
        if hi > lo:
            part[: hi - lo] = _linear_with_fp32_out(x[lo:hi], weight)
    return tensor_model_parallel_all_gather(part, dim=0)[:t]


class InklingGate(nn.Module):
'''),
    (MOE,
     '''        self.route_scale = route_scale
''',
     '''        self.route_scale = route_scale
        # Zenith patch inkling_gate_token_parallel.py
        self.tp_min_tokens = _gate_tp_min_tokens()
'''),
    (MOE,
     '''        """fp32 gate logits [T, n_total_experts + pad] (pad columns are junk)."""
        return _linear_with_fp32_out(x, self.weight)
''',
     '''        """fp32 gate logits [T, n_total_experts + pad] (pad columns are junk)."""
        if x.dim() == 2 and 0 < self.tp_min_tokens <= x.shape[0]:
            # Zenith patch inkling_gate_token_parallel.py
            return _gate_logits_token_parallel(x, self.weight)
        return _linear_with_fp32_out(x, self.weight)
'''),
]


def find_root():
    """Locate site-packages without `import vllm` (that touches the GPU)."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        sys.exit("vllm is not installed; run this inside the container")
    return os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report status without writing")
    ap.add_argument("--revert", action="store_true",
                    help="restore the replicated router GEMM")
    ap.add_argument("--root", help="directory holding the vllm package to "
                    "patch instead of the installed one (e.g. vllm_py_src)")
    args = ap.parse_args()

    root = args.root or find_root()
    path = os.path.join(root, MOE)
    if not os.path.exists(path):
        print(f"ERROR: expected {path} to exist; vLLM layout changed",
              file=sys.stderr)
        return 1
    with open(path) as fh:
        src = fh.read()

    applied = [new in src for _, _, new in HUNKS]
    marked = MARKER in src

    if args.revert:
        if not marked:
            print(f"not patched: {path}")
            return 0
        if not all(applied):
            print(f"ERROR: {path} carries a different version of this patch; "
                  "restore moe.py from the image", file=sys.stderr)
            return 1
        if args.check:
            print(f"would revert: {path}")
            return 0
        for _, old, new in reversed(HUNKS):
            src = src.replace(new, old, 1)
        with open(path, "w") as fh:
            fh.write(src)
        print(f"reverted: {path}")
        return 0

    if all(applied):
        print(f"already patched: {path}")
        return 0
    if marked:
        print(f"ERROR: {path} is partly patched or carries a different "
              "version of this patch; restore moe.py from the image",
              file=sys.stderr)
        return 1

    missing = [old for _, old, _ in HUNKS if src.count(old) != 1]
    if missing:
        for old in missing:
            print(f"ERROR: patch anchor not found once in {MOE}:\n{old}",
                  file=sys.stderr)
        print("vLLM changed InklingGate; check whether the router weight and "
              "its input are still replicated on every TP rank, then "
              "re-derive or retire this patch", file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {path}")
        return 0

    for _, old, new in HUNKS:
        src = src.replace(old, new, 1)
    with open(path, "w") as fh:
        fh.write(src)
    print(f"patched: {path}  (Inkling router GEMM -> token-parallel + "
          "all-gather for T >= INKLING_GATE_TOKEN_PARALLEL, default 4096; 0 "
          "to disable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
