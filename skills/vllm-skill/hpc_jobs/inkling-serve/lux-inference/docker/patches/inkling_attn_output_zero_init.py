#!/usr/bin/env python3
"""
Zero-initialize Inkling's attention output so padded CUDA-graph rows are 0,
not leftover graph-pool memory.

THE BUG AS IT PRESENTS

With CUDA graphs on (FULL_DECODE_ONLY) and inkling_split_kv_empty_row_guard.py
applied, the server still dies with a GPU memory fault in fused_moe_kernel.
On Tioga (MI250X, Inkling-MXFP4, TP8 x PP4) it took 7 concurrent requests,
which replay the batch-8 graph with one padded slot. Every GPU of the first
pipeline stage faulted at once. --enforce-eager ran the same load cleanly at
7 and 12 concurrent requests.

WHY

vLLM replays decode graphs at fixed batch sizes and pads the token count
(7 -> 8), but query_start_loc only covers the 7 real requests. The padded
token row belongs to no request. InklingAttention.forward allocates the output
with torch.empty, and the single-pass decode kernel (_inkling_rel_attention_kernel,
selected per layer when split-KV does not apply) launches one program per
request, so nothing ever writes that row. Inside a graph the buffer is
graph-pool memory, so the row holds whatever an earlier op left there.

Instrumenting every layer showed exactly one source: layer 0's attention
output, padded row 7, non-finite on some replays and not others, differing
across TP ranks. The garbage flows through the residual into the MoE gate.
With a NaN gate logit, the gate kernel's top-k argmax can return a lane past
the routed experts, and fused_moe_kernel indexes expert weights out of range.
Zeroing that row alone made the same 7-request load pass.

Finite garbage in a padded row is harmless because the row is discarded. It
only faults when the bit pattern is NaN/Inf, which is why the crash depends on
memory layout and batch size and looks intermittent.

THE FIX

Allocate the output with torch.zeros. Real rows are always fully overwritten
by the kernel, so outputs are unchanged. Padded rows become 0, the same value
the split-KV guard stores. The cost is one memset per layer of
tokens x heads x head_dim bf16 (2 KiB per token at TP8), which is negligible
next to the attention itself.
"""
import argparse
import importlib.util
import os
import sys

TARGET = "vllm/models/inkling/amd/attention.py"

# the exact output allocation we expect to find (upstream, unpatched)
OLD = '''        attn_output = torch.empty(
            (num_tokens, self.num_heads, self.head_dim),
            dtype=qkvr.dtype,
            device=qkvr.device,
        )
'''

NEW = '''        # Zeros, not empty: under a CUDA graph the padded decode rows belong to
        # no request and the single-pass kernel never writes them, so they
        # would keep graph-pool garbage (NaN reaches the MoE gate). Zenith
        # patch inkling_attn_output_zero_init.py.
        attn_output = torch.zeros(
            (num_tokens, self.num_heads, self.head_dim),
            dtype=qkvr.dtype,
            device=qkvr.device,
        )
'''

MARKER = "patch inkling_attn_output_zero_init.py"


def find_target():
    """Locate the attention module inside whichever site-packages vLLM lives in.

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
        print("vLLM changed InklingAttention.forward; check whether the output "
              "is still torch.empty and whether every row is still written, "
              "then re-derive or retire this patch", file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {path}")
        return 0

    with open(path, "w") as fh:
        fh.write(src.replace(OLD, NEW, 1))
    print(f"patched: {path}  (attention output zero-initialized)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
