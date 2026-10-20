"""
Inkling MTP greedy drafting: a vocab-parallel argmax (rank-local max plus a
[2, T] all-gather) instead of all-gathering the full-vocab draft logits.

THE PROBLEM
Each MTP depth samples its draft token in DraftModelSpeculator.sample_draft
as `self.model.compute_logits(h).argmax(-1)`. InklingMTP.compute_logits
runs the lm_head GEMM on this rank's vocab shard ([T, 25128] at TP8), then
all-gathers [T, 201024] bf16 logits over TP, and RCCL's dim=-1 gather adds a
transpose copy. On 1-node TP8 B=256 decode (1024 tokens, stage 1 with the
drafter, steps7 job f61kDFNJfuA7), the eager drafter spends 6.2 ms/step in
draft sampling: AG 2.38, transpose copy 0.91, full-row argmax 1.19, GEMM
1.65. Only the last 1.65 ms are needed.
vLLM already has `speculative_config.use_local_argmax_reduction` (off by
default), with InklingMTP.get_top_tokens delegating to
LogitsProcessor.get_top_tokens. That path is not exact for Inkling:
  - The lm_head is built as ParallelLMHead(201024, org_num_embeddings=201024),
    so shard_indices.num_org_vocab_padding is 0 and the 966 config padding
    columns past vocab_size=200058 (which compute_logits slices off) are not
    masked. They can win the argmax.
  - It runs `_apply_head` (a different GEMM call, without the muP alpha)
    instead of compute_logits' `addmm(alpha=1/mup)`. Different hipBLASLt
    solutions and roundings can flip bf16 near-ties.

THE FIX
  1. mtp.py: InklingMTP.get_top_tokens computes this rank's logits shard with
     exactly compute_logits' GEMM (the same addmm, or the LogitsProcessor
     path with skip_gather), so the shard is bit-identical to its slice of
     the gathered logits. It then:
       - sets the columns at or past vocab_size to -inf;
       - takes the shard's (max, first index);
       - packs [fp32 bits of max, global index] as int32 [2, T];
       - all-gathers that pair along dim 0 (no transpose);
       - picks the rank with torch.argmax over the gathered maxima.
     InklingMTP sets `draft_tp_argmax_exact` from INKLING_DRAFT_TP_ARGMAX
     (default on, "0" = off).
  2. speculator.py: DraftModelSpeculator.load_model turns
     use_local_argmax_reduction on for a draft that sets
     draft_tp_argmax_exact, provided drafting is greedy and adaptive
     verification (which needs the draft logits) is off. It then runs the
     stock validation, which logs "Using local argmax reduction".

Why the result equals torch.argmax over the gathered, sliced logits:
  - On CUDA/ROCm, torch.max(dim) and torch.argmax reduce with the same
    GreaterOrNan comparator. Any NaN beats any number, and between NaNs, or
    between equal values (including -0.0 == +0.0, and +/-inf), the lower
    index wins. The result is therefore the first NaN if the row has one,
    else the first maximum, independent of reduction order.
  - The shards are contiguous column ranges in rank order (the AG
    concatenates dim -1 by rank). So the global winner sits in the first
    rank whose shard holds it, at that shard's own winner, and the
    cross-rank argmax (same comparator, first rank on ties or NaN) selects
    exactly that rank.
  - The values travel exactly: bf16 -> fp32 is exact, and the fp32 bits move
    as int32 and are viewed back, so NaN stays NaN. Indices are int32 and
    never pass through floating point.
  - Columns past vocab_size are set to -inf before the local max, which also
    overwrites any NaN or +inf there. As the highest indices, they lose
    every tie, so they never win (an all -inf row gives index 0 in both).
    torch.argmax over the sliced row never sees them either.
  - With the flag off nothing changes: compute_logits + argmax, as before.
Graph safety (FULL_DECODE_ONLY captures the drafter): every shape is fixed
by T and the TP size, the -inf fill is a static slice, and there is no host
sync or data-dependent control flow.

Upstream status
---------------
Present on the pinned build 0.2.1.dev32+gbc0f47cd0 (vllm_py_src).
"""
import argparse
import importlib.util
import os
import sys

MARKER = "Zenith patch inkling_draft_tp_argmax.py"

MTP = "vllm/models/inkling/amd/mtp.py"
SPEC = "vllm/v1/worker/gpu/spec_decode/speculator.py"

# (target, OLD, NEW): every OLD occurs exactly once in the unpatched file.
HUNKS = [
    (MTP,
     '''from collections.abc import Iterable

import regex as re
''',
     '''import os
from collections.abc import Iterable

import regex as re
'''),
    (MTP,
     '''from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
''',
     '''from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.model_executor.layers.linear import ReplicatedLinear
'''),
    (MTP,
     '''

class InklingMTP(nn.Module, SupportsMultiModalEmbeddings):
''',
     '''

# Zenith patch inkling_draft_tp_argmax.py: greedy draft tokens from a
# rank-local argmax over the vocab shard plus a [2, T] all-gather, instead of
# all-gathering [T, vocab] logits. Same result as torch.argmax over the
# gathered logits, including ties (first index) and NaN (first NaN).
def _draft_tp_argmax_local(
    logits: torch.Tensor, col_start: int, n_valid: int
) -> torch.Tensor:
    """[T, V/tp] logits shard starting at global column col_start, of which
    the first n_valid columns are real vocab -> int32 [2, T]: the shard's
    max (fp32 bits) and its global index (first occurrence)."""
    if n_valid < logits.shape[-1]:
        logits[:, n_valid:] = float("-inf")
    vals, idx = logits.max(dim=-1)
    gidx = idx.to(torch.int32) + col_start
    return torch.stack([vals.float().view(torch.int32), gidx])


def _draft_tp_argmax_reduce(gathered: torch.Tensor) -> torch.Tensor:
    """[tp * 2, T] rank-ordered _draft_tp_argmax_local pairs -> int64 [T]."""
    g = gathered.view(-1, 2, gathered.shape[-1])
    rank = g[:, 0].view(torch.float32).argmax(dim=0, keepdim=True)
    return g[:, 1].gather(0, rank).squeeze(0).long()


class InklingMTP(nn.Module, SupportsMultiModalEmbeddings):
'''),
    (MTP,
     '''        self._logits_zero: torch.Tensor | None = None
''',
     '''        self._logits_zero: torch.Tensor | None = None
        # Zenith patch inkling_draft_tp_argmax.py: get_top_tokens is exact,
        # so the speculator uses it for greedy drafting unless this is off.
        self.draft_tp_argmax_exact = (
            os.environ.get("INKLING_DRAFT_TP_ARGMAX", "1") != "0"
        )
'''),
    (MTP,
     '''    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy draft tokens via rank-local argmax + tiny (value, index)
        reduction — no full-vocab logits all-gather. The muP divisor is a
        positive scalar, so the argmax is invariant and the scaling is
        skipped entirely."""
        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)
''',
     '''    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy draft tokens via rank-local argmax + tiny (value, index)
        reduction — no full-vocab logits all-gather. Bit-identical to
        compute_logits(hidden_states).argmax(-1) (Zenith patch
        inkling_draft_tp_argmax.py): the shard comes from the same GEMM as
        compute_logits, and the vocab padding columns are masked."""
        mup = self.config.logits_mup_width_multiplier
        if mup:
            assert self.logits_processor.soft_cap is None
            assert self.logits_processor.scale == 1.0
            w = self.lm_head.weight
            if self._logits_zero is None:
                self._logits_zero = w.new_zeros(1)
            logits = torch.addmm(
                self._logits_zero,
                hidden_states,
                w.t(),
                beta=0.0,
                alpha=1.0 / mup,
            )
        else:
            logits = self.logits_processor(
                self.lm_head, hidden_states, skip_gather=True
            )
        col_start = self.lm_head.shard_indices.padded_org_vocab_start_index
        n_valid = min(
            logits.shape[-1],
            max(0, self.logits_processor.org_vocab_size - col_start),
        )
        pair = _draft_tp_argmax_local(logits, col_start, n_valid)
        return _draft_tp_argmax_reduce(
            tensor_model_parallel_all_gather(pair, dim=0)
        )
'''),
    (SPEC,
     '''        self.model = self.load_draft_model(target_model, target_attn_layer_names)
        self._validate_local_argmax_reduction()
''',
     '''        self.model = self.load_draft_model(target_model, target_attn_layer_names)
        # Zenith patch inkling_draft_tp_argmax.py: a draft whose get_top_tokens
        # is bit-identical to argmax over the gathered logits drafts greedily
        # through it by default.
        if (
            not self.use_local_argmax_reduction
            and getattr(self.model, "draft_tp_argmax_exact", False)
            and self.speculative_config.draft_sample_method == "greedy"
            and not self.use_acceptance_estimator
        ):
            self.use_local_argmax_reduction = True
        self._validate_local_argmax_reduction()
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
                    help="restore the original get_top_tokens / load_model")
    ap.add_argument("--root", help="directory holding the vllm package to "
                    "patch instead of the installed one (e.g. vllm_py_src)")
    args = ap.parse_args()

    root = args.root or find_root()
    srcs = {}
    for rel in (MTP, SPEC):
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"ERROR: expected {path} to exist; vLLM layout changed",
                  file=sys.stderr)
            return 1
        with open(path) as fh:
            srcs[rel] = fh.read()

    applied = [new in srcs[rel] for rel, _, new in HUNKS]
    marked = any(MARKER in s for s in srcs.values())

    if args.revert:
        if not marked:
            print(f"not patched: {root}")
            return 0
        if not all(applied):
            print(f"ERROR: {root} carries a different version of this patch; "
                  "restore mtp.py and speculator.py from the image",
                  file=sys.stderr)
            return 1
        if args.check:
            print(f"would revert: {root}")
            return 0
        for rel, old, new in reversed(HUNKS):
            srcs[rel] = srcs[rel].replace(new, old, 1)
        for rel, src in srcs.items():
            with open(os.path.join(root, rel), "w") as fh:
                fh.write(src)
        print(f"reverted: {root}")
        return 0

    if all(applied):
        print(f"already patched: {root}")
        return 0
    if marked:
        print(f"ERROR: {root} is partly patched or carries a different "
              "version of this patch; restore mtp.py and speculator.py from "
              "the image", file=sys.stderr)
        return 1

    missing = [(rel, old) for rel, old, _ in HUNKS if srcs[rel].count(old) != 1]
    if missing:
        for rel, old in missing:
            print(f"ERROR: patch anchor not found once in {rel}:\n{old}",
                  file=sys.stderr)
        print("vLLM changed InklingMTP.get_top_tokens or "
              "DraftModelSpeculator.load_model; check whether greedy drafting "
              "still all-gathers the full-vocab logits, then re-derive or "
              "retire this patch", file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {root}")
        return 0

    for rel, old, new in HUNKS:
        srcs[rel] = srcs[rel].replace(old, new, 1)
    for rel, src in srcs.items():
        with open(os.path.join(root, rel), "w") as fh:
            fh.write(src)
    print(f"patched: {root}  (Inkling MTP greedy drafting -> vocab-parallel "
          "argmax, INKLING_DRAFT_TP_ARGMAX=0 to disable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
