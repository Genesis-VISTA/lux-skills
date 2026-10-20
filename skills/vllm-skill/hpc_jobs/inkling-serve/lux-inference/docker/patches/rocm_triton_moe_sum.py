"""
TritonExperts.moe_sum on ROCm at prefill sizes: a Triton top-k reduction
that can also add one same-shaped tensor (e.g. a model's shared-expert
output) in the same pass.

THE PROBLEM

TritonExperts.apply ends with ops.moe_sum(intermediate_cache3, output),
the [M, topk, H] -> [M, H] sum over each token's experts. With topk=6 and
H=4096 (Inkling-Small, TP8 per-rank shapes) on MI250X it runs at about
0.96 TB/s: 491 us at M=8192, 242 us at 4096, 117 us at 2048.
Models with shared experts then add their output in a second pass, e.g.
InklingMoE's out.add_(sink_out), which is another 165 us at M=8192.

THE FIX

For M >= 1024 on ROCm, sum with a Triton kernel: one row x 2048 columns per
program, 8 warps, fp32 accumulation in top-k order, one rounding to the
output dtype. That is the same arithmetic as ops.moe_sum at topk > 4, and
the outputs are bit-identical.
Measured on one GCD (moe_sum_bench.py):

| M | ops.moe_sum | Triton |
|---|---|---|
| 8192 | 491 us | 399 us |
| 4096 | 242 us | 199 us |
| 2048 | 117 us | 100 us |
| 1024 | 58 us | 52 us |

Below 1024 the C++ kernel stays.

MOE_SUM_ADDEND is a one-shot slot. A caller that puts (addend, slabs) in
MOE_SUM_ADDEND[0] before calling the experts gets the addend (a tensor of
the output's shape and dtype) added in the same kernel, as
out = bf16(bf16(sum) + addend), which is bit-identical to moe_sum followed
by out.add_(addend). If slabs is a [tp, M, N // tp] tensor, the result goes
there rank-major instead (the layout inkling_sconv_zero_copy.py's
add_to_slabs writes for the sconv reduce-scatter), with the same values.
- The slot is cleared (set to None) when it is consumed.
- A caller whose slot is still set afterwards must do the add itself. That
  happens below the size threshold, on other experts classes, or with
  chunked MoE.

At M=8192, sum plus fused add takes 459 us, versus 657 us for moe_sum
followed by add_.
"""
import argparse
import importlib.util
import os
import sys

TARGET = "vllm/model_executor/layers/fused_moe/experts/triton_moe.py"

OLD_IMPORT = '''from vllm.platforms import current_platform
from vllm.triton_utils import tl
'''

NEW_IMPORT = '''from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
'''

OLD_SUM = '''    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        ops.moe_sum(input, output)


class TritonWNA16Experts(TritonExperts):
'''

NEW_SUM = '''    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        # (patch rocm_triton_moe_sum.py) Triton sum at prefill sizes on ROCm,
        # taking a caller-provided MOE_SUM_ADDEND in the same pass.
        if _moe_sum_triton_ok(input, output):
            add, slabs = _moe_sum_take_addend(input, output)
            _moe_sum_triton(input, output, add, slabs)
            return
        ops.moe_sum(input, output)


# Zenith patch rocm_triton_moe_sum.py: ops.moe_sum runs at ~0.96 TB/s on
# MI250X (491 us at M=8192, topk 6, H 4096); this Triton kernel does it in
# 399 us, and adds a same-shaped addend in the same pass (459 us, vs 657 us
# for moe_sum + add_). Both are bit-identical to the unfused ops.
#
# One-shot request for the next TritonExperts.moe_sum: (addend, slabs).
#   addend  [M, N] tensor added to the sum, as bf16(bf16(sum) + addend).
#   slabs   None, or a [tp, M, N // tp] tensor; the result is then written
#           there rank-major (element (m, r*C + c) at slabs[r, m, c]) and the
#           [M, N] output is left unwritten.
# Consumed (reset to None) only when honoured, which needs
# M >= MOE_SUM_ADDEND_MIN_TOKENS; a caller that finds it still set must do
# the add (and any slab layout) itself.
MOE_SUM_ADDEND: list[tuple[torch.Tensor, torch.Tensor | None] | None] = [None]
MOE_SUM_ADDEND_MIN_TOKENS = 1024
_MOE_SUM_BLOCK_M = 1
_MOE_SUM_BLOCK_N = 2048


@triton.jit(do_not_specialize=["M", "slab_stride"])
def _moe_sum_kernel(
    in_ptr,  # [M, TOPK, N]
    out_ptr,  # [M, N], or [N // SEG, M, SEG] slabs
    add_ptr,  # [M, N] (read only if HAS_ADD)
    M,
    slab_stride,  # M * SEG for slabs, 0 for a plain [M, N] output
    N: tl.constexpr,
    SEG: tl.constexpr,  # N for a plain output
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_ADD: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    rows = offs_m[:, None].to(tl.int64)
    src = in_ptr + rows * (TOPK * N) + offs_n[None, :]
    acc = tl.load(src, mask=mask, other=0.0).to(tl.float32)
    for k in tl.static_range(1, TOPK):
        acc += tl.load(src + k * N, mask=mask, other=0.0).to(tl.float32)
    res = acc.to(out_ptr.dtype.element_ty)
    if HAS_ADD:
        a = tl.load(add_ptr + rows * N + offs_n[None, :], mask=mask, other=0.0)
        res = (res.to(tl.float32) + a.to(tl.float32)).to(out_ptr.dtype.element_ty)
    dst = (
        (offs_n[None, :] // SEG).to(tl.int64) * slab_stride
        + rows * SEG
        + offs_n[None, :] % SEG
    )
    tl.store(out_ptr + dst, res, mask=mask)


def _moe_sum_triton(
    input: torch.Tensor,
    output: torch.Tensor,
    add: torch.Tensor | None,
    slabs: torch.Tensor | None,
) -> None:
    m, topk, n = input.shape
    if slabs is None:
        out, seg, slab_stride = output, n, 0
    else:
        out, seg, slab_stride = slabs, slabs.shape[2], m * slabs.shape[2]
    grid = (triton.cdiv(m, _MOE_SUM_BLOCK_M), triton.cdiv(n, _MOE_SUM_BLOCK_N))
    _moe_sum_kernel[grid](
        input,
        out,
        add if add is not None else output,
        m,
        slab_stride,
        N=n,
        SEG=seg,
        TOPK=topk,
        BLOCK_M=_MOE_SUM_BLOCK_M,
        BLOCK_N=_MOE_SUM_BLOCK_N,
        HAS_ADD=add is not None,
        num_warps=8,
    )


def _moe_sum_take_addend(
    input: torch.Tensor, output: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    \"\"\"Take MOE_SUM_ADDEND if it fits this call (else leave it set).\"\"\"
    req = MOE_SUM_ADDEND[0]
    if req is None:
        return None, None
    add, slabs = req
    m, _, n = input.shape
    if not (
        add.shape == output.shape
        and add.dtype == output.dtype
        and add.device == output.device
        and add.is_contiguous()
    ):
        return None, None
    if slabs is not None and not (
        slabs.dim() == 3
        and slabs.shape[1] == m
        and slabs.shape[0] * slabs.shape[2] == n
        and slabs.dtype == output.dtype
        and slabs.device == output.device
        and slabs.is_contiguous()
    ):
        return None, None
    MOE_SUM_ADDEND[0] = None
    return add, slabs


def _moe_sum_triton_ok(input: torch.Tensor, output: torch.Tensor) -> bool:
    return (
        current_platform.is_rocm()
        and input.dim() == 3
        and input.shape[0] >= MOE_SUM_ADDEND_MIN_TOKENS
        and input.dtype in (torch.bfloat16, torch.float16)
        and output.dtype == input.dtype
        and output.shape == (input.shape[0], input.shape[2])
        and input.is_contiguous()
        and output.is_contiguous()
    )


class TritonWNA16Experts(TritonExperts):
'''

MARKER = "Zenith patch rocm_triton_moe_sum.py"
EDITS = ((OLD_IMPORT, NEW_IMPORT), (OLD_SUM, NEW_SUM))


def find_target():
    """Locate the file in the installed vLLM without importing vllm."""
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
                    help="restore the original moe_sum")
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
                  "restore triton_moe.py from the image", file=sys.stderr)
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
        print("vLLM changed TritonExperts.moe_sum or the module imports; check "
              "whether moe_sum still calls ops.moe_sum, then re-derive or "
              "retire this patch", file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {path}")
        return 0

    for old, new in EDITS:
        src = src.replace(old, new, 1)
    with open(path, "w") as fh:
        fh.write(src)
    print(f"patched: {path}  (ROCm prefill moe_sum -> Triton, optional fused addend)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
