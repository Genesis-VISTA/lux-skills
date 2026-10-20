#!/usr/bin/env python3
"""
Stop Inkling's split-KV decode from writing NaN for padded (empty) batch rows.

THE BUG AS IT PRESENTS

With CUDA graphs on (FULL_DECODE_ONLY), the server dies with a GPU memory
fault at a host address as soon as 3 or more requests decode concurrently.
The fault was reported by Brandon (LLNL) on an earlier zenith release. It
reproduced across runs and node sets, and went away with INKLING_SPLIT_KV=0
or --enforce-eager.

WHY

vLLM replays decode graphs at fixed batch sizes, so 3 live requests run in
the batch-4 graph with one padded slot whose cache_seqlens entry is 0. In
vllm/models/inkling/amd/ops/rel_attention_decode.py, _split_kv_stage1 skips
every split of an empty row (effective_len == 0) and _split_kv_stage2 then
combines zero splits: acc and row_sum both stay 0 and the final store is

    acc / row_sum        ->  0 / 0  =  NaN

for every head of the padded row. The NaN flows through the residual into
the MoE gate; with a NaN gate logit the gate kernel's top-k argmax can return
a lane past the routed experts, and fused_moe_kernel faults indexing expert
weights out of range. On Tioga (MI250X, Inkling-MXFP4, TP8 x PP4) the
unpatched image reproduced it with 3 concurrent requests, one of them about
5k tokens: fused_moe_kernel memory fault. Eager mode never pads the batch,
so it has no empty rows.

This guard is necessary but not sufficient. The single-pass decode kernel
(INKLING_SPLIT_KV=0, and every layer where split-KV is not selected, e.g.
sliding-window layers at block size 16) never writes the padded row at all,
which leaves torch.empty garbage that can be NaN. That is fixed separately by
inkling_attn_output_zero_init.py; apply both.

THE FIX

Guard the stage-2 denominator: an empty row stores 0 (acc is already 0 when
no split contributed, so dividing by 1 gives exactly 0). Rows with at least
one split are bit-identical to before. This matches the guard Brandon's team
mounts over the image file, which has run with zero faults at 262k context
and 64 concurrent users.
"""
import argparse
import importlib.util
import os
import sys

TARGET = "vllm/models/inkling/amd/ops/rel_attention_decode.py"

# the exact stage-2 store we expect to find (upstream, unpatched)
OLD = '''    tl.store(
        out_ptr + pid_t * stride_o_t + pid_h * stride_o_h + off_d,
        acc / row_sum,
        mask=d_valid,
    )
'''

NEW = '''    # A padded CUDA-graph slot has cache_len 0, so no split contributed and
    # acc / row_sum would be 0 / 0 = NaN. Store 0 for it instead (zenith patch
    # inkling_split_kv_empty_row_guard.py).
    safe_row_sum = tl.where(row_sum > 0.0, row_sum, 1.0)
    tl.store(
        out_ptr + pid_t * stride_o_t + pid_h * stride_o_h + off_d,
        acc / safe_row_sum,
        mask=d_valid,
    )
'''

MARKER = "safe_row_sum = tl.where(row_sum > 0.0, row_sum, 1.0)"


def find_target():
    """Locate the decode kernel inside whichever site-packages vLLM lives in.

    Deliberately does NOT `import vllm` (see rocm_aiter_ray_env_refresh.py):
    importing it touches the GPU at module scope and fails in a build container.
    """
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
    ap.add_argument("--path", help="patch this file instead of the installed "
                    "vLLM's (e.g. a copy bind-mounted over the image file)")
    args = ap.parse_args()

    path = args.path or find_target()
    with open(path) as fh:
        src = fh.read()

    if MARKER in src:
        print(f"already patched: {path}")
        return 0

    if OLD not in src:
        print(f"ERROR: patch target not found in {path}", file=sys.stderr)
        print("vLLM changed _split_kv_stage2; check whether upstream now guards "
              "empty rows and re-derive or retire this patch", file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {path}")
        return 0

    with open(path, "w") as fh:
        fh.write(src.replace(OLD, NEW, 1))
    print(f"patched: {path}  (split-KV stage 2 guards empty rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
