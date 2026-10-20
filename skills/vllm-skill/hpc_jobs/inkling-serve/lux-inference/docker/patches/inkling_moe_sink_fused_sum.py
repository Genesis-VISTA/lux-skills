"""
InklingMoE at prefill sizes: add the sink experts' output inside the
routed experts' moe_sum instead of in a separate pass, and with
inkling_sconv_zero_copy.py, write it rank-major there too.

THE PROBLEM

Above VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD tokens, InklingMoE.forward
runs the routed experts and then the sink experts on one stream, and then
joins them with a second full [T, 4096] bf16 pass:
- out.add_(sink_out): 165 us at T=8192 on MI250X;
- or, with inkling_sconv_zero_copy.py, add_to_slabs(out, sink_out, tp),
  which also writes the result rank-major (tp, T, C) for the sconv
  reduce-scatter.

THE FIX

Needs rocm_triton_moe_sum.py, which adds the one-shot TritonExperts
MOE_SUM_ADDEND slot. On this sequential path:
1. Run the sink experts first.
2. Put (sink_out, slabs) in the slot, where slabs is a fresh (tp, T, C)
   tensor when the zero-copy sconv path is on (tp > 1), else None.
3. Run the routed experts. Their moe_sum then writes
   bf16(bf16(sum) + sink_out), either into the output or rank-major into
   slabs, which is bit-identical to the sum followed by add_ / add_to_slabs.

This path is taken only at T >= MOE_SUM_ADDEND_MIN_TOKENS (1024), where
the Triton sum takes the addend. If the slot was still not consumed, forward
does the add (or add_to_slabs) itself, so the result is unchanged on every
path. That covers non-Triton experts and chunked MoE. Without the slot, the
original code runs. The decode-sized path (sinks on the aux stream,
overlapped with the routed GEMMs) is unchanged.

The patch only inserts code ahead of the routed/sink join, so it applies
before or after inkling_sconv_zero_copy.py; it finds that patch's
add_to_slabs / _SCONV_ZC at run time.

At T=8192 the sum plus the fused add takes 459 us, versus 657 us for
moe_sum followed by add_ (moe_sum_bench.py).
"""
import argparse
import importlib.util
import os
import sys

TARGET = "vllm/models/inkling/amd/moe.py"

OLD = '''        out, sink_out = maybe_execute_in_parallel(
            lambda: self.experts(hidden_states=x, router_logits=router_logits),
'''

NEW = '''        slot = _moe_sum_addend_slot(num_tokens)
        if (
            slot is not None
            and num_tokens > envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
        ):
            # Zenith patch inkling_moe_sink_fused_sum.py: sequential path,
            # sinks first, routed moe_sum adds them.
            return _forward_sink_fused_sum(self, x, gammas, router_logits, slot)

        out, sink_out = maybe_execute_in_parallel(
            lambda: self.experts(hidden_states=x, router_logits=router_logits),
'''

OLD_HELPER = '''def _inkling_moe_ep_size() -> int:
'''

NEW_HELPER = '''def _moe_sum_addend_slot(num_tokens: int) -> list | None:
    """TritonExperts' one-shot moe_sum addend slot, if this vLLM has it
    (patch rocm_triton_moe_sum.py) and its moe_sum takes an addend at this
    size; see inkling_moe_sink_fused_sum.py."""
    from vllm.model_executor.layers.fused_moe.experts import triton_moe

    slot = getattr(triton_moe, "MOE_SUM_ADDEND", None)
    min_tokens = getattr(triton_moe, "MOE_SUM_ADDEND_MIN_TOKENS", None)
    if slot is None or min_tokens is None or num_tokens < min_tokens:
        return None
    return slot


def _forward_sink_fused_sum(
    layer, x: torch.Tensor, gammas: torch.Tensor, router_logits: torch.Tensor,
    slot: list,
) -> torch.Tensor:
    """InklingMoE.forward's routed/sink join with the sink add done by the
    routed moe_sum (patch inkling_moe_sink_fused_sum.py). Returns what the
    unpatched join returns: out + sink_out, as rank-major (tp, T, C) slabs
    when inkling_sconv_zero_copy.py's path is on, else as [T, H]."""
    sink_out = layer.sink_experts(x, gammas)
    tp = get_tensor_model_parallel_world_size()
    to_slabs = globals().get("add_to_slabs")
    slabs = None
    if (
        globals().get("_SCONV_ZC")
        and to_slabs is not None
        and tp > 1
        and sink_out.dim() == 2
        and sink_out.shape[1] % tp == 0
        and sink_out.is_contiguous()
    ):
        t, h = sink_out.shape
        slabs = sink_out.new_empty((tp, t, h // tp))
    slot[0] = (sink_out, slabs)
    try:
        out = layer.experts(hidden_states=x, router_logits=router_logits)
    finally:
        fused = slot[0] is None
        slot[0] = None
    layer._routed_sel = None
    if fused:
        return out if slabs is None else slabs
    if (
        slabs is not None
        and out.dim() == 2
        and out.shape == sink_out.shape
        and out.stride(1) == 1
    ):
        return to_slabs(out, sink_out, tp)
    return out.add_(sink_out)


def _inkling_moe_ep_size() -> int:
'''

MARKER = "Zenith patch inkling_moe_sink_fused_sum.py"
EDITS = ((OLD, NEW), (OLD_HELPER, NEW_HELPER))


def find_target():
    """Locate moe.py in the installed vLLM without importing vllm (importing
    it touches the GPU at module scope)."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        sys.exit("vllm is not installed; run this inside the container")
    root = os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))
    path = os.path.join(root, TARGET)
    if not os.path.exists(path):
        sys.exit(f"expected {path} to exist; vLLM layout changed")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report status without writing")
    ap.add_argument("--revert", action="store_true",
                    help="restore the original sink add")
    ap.add_argument("--path", help="patch this file instead of the installed "
                    "vLLM's (e.g. a copy bind-mounted over the image file)")
    args = ap.parse_args()

    path = args.path or find_target()
    with open(path) as fh:
        src = fh.read()

    if args.revert:
        if MARKER not in src:
            print(f"not patched: {path}")
            return 0
        if any(new not in src for _, new in EDITS):
            print(f"ERROR: {path} carries a different version of this patch; "
                  "restore moe.py from the image", file=sys.stderr)
            return 1
        for old, new in EDITS:
            src = src.replace(new, old, 1)
        with open(path, "w") as fh:
            fh.write(src)
        print(f"reverted: {path}")
        return 0

    if MARKER in src:
        print(f"already patched: {path}")
        return 0

    missing = [i for i, (old, _) in enumerate(EDITS) if src.count(old) != 1]
    if missing:
        print(f"ERROR: patch target(s) {missing} not found exactly once in {path}",
              file=sys.stderr)
        print("vLLM changed InklingMoE.forward's routed/sink join; check "
              "whether it still joins them with maybe_execute_in_parallel, "
              "then re-derive or retire this patch", file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {path}")
        return 0

    for old, new in EDITS:
        src = src.replace(old, new, 1)
    with open(path, "w") as fh:
        fh.write(src)
    print(f"patched: {path}  (prefill sink add fused into the routed moe_sum)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
