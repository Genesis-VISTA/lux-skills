"""
Inkling MTP depth input_proj (2H -> H): token-parallel GEMM plus a dim-0
all-gather instead of the full replicated GEMM on every TP rank.

THE PROBLEM
InklingMTPDepthLayer.input_proj is a ReplicatedLinear [4096, 8192]. Every TP
rank multiplies all T rows of the fused [rmsnorm(hidden) | embed_norm(emb)]
input by the full weight, so TP8 does 8x the work for one result. On 1-node
TP8, in the mixed refill step (194 decodes + 8k-prefill chunks, 8192
tokens, steps7 job f61kDFNJfuA7), each MTP depth runs it at T=8192 as
hipBLASLt Cijk_..._BBS_MT256x256x32, taking 4.6 ms. The three depths cost
13.8 ms of the step's 61.7 ms eager drafter (388 ms step). In the B=256
decode step (T=1024) it is 0.57 ms per depth.

THE FIX
mtp.py: when T >= INKLING_MTP_INPUT_PROJ_TP (a token count, default 512;
"0" turns the path off), TP rank r runs the same ReplicatedLinear on its
contiguous ceil(T/tp) row block, and a dim-0 all-gather (rank-ordered,
no transpose copy) reassembles the [T, 4096] output. The weight, its loader
and the module tree are unchanged. Every rank holds the identical
replicated input, so the gathered rows are the rows the full GEMM would
produce (up to the accumulation order, see Exactness). Below the threshold
(graphed decode, small T), the GEMM is weight-read bound: each rank still
reads the full 64 MB weight, so the token split saves nothing and only adds
the all-gather (ipt_bench, graph replay: 192 vs 193 us at T=256, 329 vs
213 us at T=512, 4791 vs 1184 us at T=8192).

Exactness: each output row is the same dot products over the same K=8192,
on the same weight, through the same layer forward (and so the same GEMM
dispatch), only with M = ceil(T/tp) instead of T. hipBLASLt picks its
solution by problem size, and a different solution can change the fp32
accumulation order. ipt_bench.py (job f61kuFbqi9HR, real weight, 8 GCDs):
bit-identical at T=1..64 and 4096, within 1 bf16 ulp (max abs 0.125)
elsewhere -- the same batch-size variance the stock GEMM has across T.
Greedy MTP outputs and acceptance are checked end to end in the same job.
Graph safety: the branch and the row block depend only on T, the TP size
and the rank, all fixed per captured graph. The all-gather is the same
graph-captured collective the sconv already uses, and there is no host
sync.

Upstream status
---------------
Present on the pinned build 0.2.1.dev32+gbc0f47cd0 (vllm_py_src).
"""
import argparse
import importlib.util
import os
import sys

MARKER = "Zenith patch inkling_mtp_input_proj_tp.py"

MTP = "vllm/models/inkling/amd/mtp.py"

# (target, OLD, NEW): every OLD occurs exactly once in the unpatched file, and
# none overlaps inkling_draft_tp_argmax.py or inkling_mtp_pp_embed_share.py.
HUNKS = [
    (MTP,
     '''from .ops.norm import embed_dual_rmsnorm_cat, embed_rmsnorm
''',
     '''from .ops.norm import embed_dual_rmsnorm_cat, embed_rmsnorm
from vllm.distributed import (  # Zenith patch inkling_mtp_input_proj_tp.py
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
'''),
    (MTP,
     '''

class InklingMTPDepthLayer(nn.Module):
''',
     '''

# Zenith patch inkling_mtp_input_proj_tp.py: the replicated input_proj GEMM
# split by rows over TP plus a dim-0 all-gather, for T >= the threshold.
def _input_proj_tp_min_tokens() -> int:
    """INKLING_MTP_INPUT_PROJ_TP: smallest T that takes the token-parallel
    input_proj (default 512); 0 turns it off."""
    import os

    return max(0, int(os.environ.get("INKLING_MTP_INPUT_PROJ_TP", "512") or 0))


def _input_proj_token_parallel(
    proj: ReplicatedLinear, x: torch.Tensor
) -> torch.Tensor:
    """proj(x) for a TP-replicated x: each rank runs proj on its contiguous
    ceil(T/tp) row block, then a rank-ordered dim-0 all-gather."""
    tp = get_tensor_model_parallel_world_size()
    t = x.shape[0]
    if tp == 1:
        return proj(x)
    rows = -(-t // tp)
    lo = min(get_tensor_model_parallel_rank() * rows, t)
    hi = min(lo + rows, t)
    if hi - lo == rows:
        part = proj(x[lo:hi])
    else:
        # Short or empty last blocks: rows past T are never read back.
        part = x.new_empty((rows, proj.output_size))
        if hi > lo:
            part[: hi - lo] = proj(x[lo:hi])
    return tensor_model_parallel_all_gather(part, dim=0)[:t]


class InklingMTPDepthLayer(nn.Module):
'''),
    (MTP,
     '''            prefix=f"{prefix}.input_proj",
        )
''',
     '''            prefix=f"{prefix}.input_proj",
        )
        # Zenith patch inkling_mtp_input_proj_tp.py
        self.input_proj_tp_min_tokens = _input_proj_tp_min_tokens()
'''),
    (MTP,
     '''        hidden = self.input_proj(combined)
''',
     '''        if 0 < self.input_proj_tp_min_tokens <= combined.shape[0]:
            # Zenith patch inkling_mtp_input_proj_tp.py
            hidden = _input_proj_token_parallel(self.input_proj, combined)
        else:
            hidden = self.input_proj(combined)
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
                    help="restore the replicated input_proj")
    ap.add_argument("--root", help="directory holding the vllm package to "
                    "patch instead of the installed one (e.g. vllm_py_src)")
    args = ap.parse_args()

    root = args.root or find_root()
    path = os.path.join(root, MTP)
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
                  "restore mtp.py from the image", file=sys.stderr)
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
              "version of this patch; restore mtp.py from the image",
              file=sys.stderr)
        return 1

    missing = [old for _, old, _ in HUNKS if src.count(old) != 1]
    if missing:
        for old in missing:
            print(f"ERROR: patch anchor not found once in {MTP}:\n{old}",
                  file=sys.stderr)
        print("vLLM changed InklingMTPDepthLayer; check whether input_proj is "
              "still a ReplicatedLinear run in full on every TP rank, then "
              "re-derive or retire this patch", file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {path}")
        return 0

    for _, old, new in HUNKS:
        src = src.replace(old, new, 1)
    with open(path, "w") as fh:
        fh.write(src)
    print(f"patched: {path}  (Inkling MTP input_proj -> token-parallel + "
          "all-gather for T >= INKLING_MTP_INPUT_PROJ_TP, default 512; 0 to "
          "disable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
