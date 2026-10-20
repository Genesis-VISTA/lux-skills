"""
Inkling MoE router GEMM on ROCm: a split-K Triton kernel for small token
counts instead of hipBLASLt's bf16 x bf16 -> fp32 GEMM.

THE PROBLEM

InklingGate.compute_logits computes fp32 router logits
[T, 264] = x[T, 4096] (bf16) @ W[264, 4096]^T (bf16) through
_linear_with_fp32_out. On NVIDIA (sm90) small T goes to the low-latency
ll_bf16 kernel; everywhere else it falls back to
torch.mm(flat, weight.T, out_dtype=torch.float32), which hipBLASLt runs on
MI250X as Cijk_Alik_Bljk_BSS_..._MT256x32x32 (T=4) / MT128x32x64 (T=1).
The macro tile covers 256 (or 128) of the 264 experts, so the GEMM runs on
2-3 workgroups that walk K=4096 serially: 58 us at T=1 and 162-206 us at
T=4 (MTP verify), for a 2.2 MB weight read. That is 20 calls per PP stage
per decode pass (one per MoE layer), 6.6 ms/pass at c1 on 2-node PP2.

THE FIX

For T <= 64 on ROCm, compute the same product with a split-K Triton kernel:
(experts / 32) x 8 programs each take a 32-expert x 512-K slice, multiply
in bf16 MFMA with fp32 accumulation (the products of two bf16 values are
exact in fp32, as in hipBLASLt), and write fp32 partials; a second kernel
sums the 8 partials in a fixed order. Tile picked by a sweep on MI250X
(BLOCK_E 16/32/64 x K-split 128/256/512 x BLOCK_K x warps), real gate
weights: 8-12 us at T=1..16, 15-16 us at T=64. Output dtype, shape and padding
columns are unchanged. The result is deterministic, but it is not
bit-identical to hipBLASLt's, since the fp32 accumulation order differs
(hipBLASLt itself switches solution and order at larger T). T > 64 and
non-ROCm keep the original path.

ARCH GATE

The tile and the hipBLASLt comparison are MI250X-only, so the split-K path is
on by default only on gfx90a (vllm.platforms.rocm.on_gfx90a, resolved once at
import through amdsmi without CUDA init). INKLING_GATE_SPLITK=1 forces it on and
=0 forces it off on any ROCm GPU; any other value is an error. On gfx90a with
the variable unset, the dispatch and kernels are the same as the first version
of this patch (ac9bcf13), whose patched moe.py this script upgrades in place.
"""
import argparse
import importlib.util
import os
import sys

TARGET = "vllm/models/inkling/amd/moe.py"

OLD = '''def _linear_with_fp32_out(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    leading = list(x.shape[:-1])
    flat = x.flatten(0, -2)
    if (
'''

_HEAD_V1 = '''# Zenith patch inkling_gate_fp32_splitk.py: on ROCm, hipBLASLt runs the small-T
# bf16 x bf16 -> fp32 router GEMM on 2-3 workgroups (58-206 us); a split-K
# Triton kernel with a fixed-order reduction does it in a few us.
_INKLING_GATE_SPLITK_MAX_TOKENS = 64
_GATE_SPLITK_BLOCK_E = 32
_GATE_SPLITK_BLOCK_K = 256
_GATE_SPLITK_K_PER_SPLIT = 512


'''

_HEAD = '''# Zenith patch inkling_gate_fp32_splitk.py: on ROCm, hipBLASLt runs the small-T
# bf16 x bf16 -> fp32 router GEMM on 2-3 workgroups (58-206 us); a split-K
# Triton kernel with a fixed-order reduction does it in a few us. On by default
# only on gfx90a, where it was tuned; INKLING_GATE_SPLITK=1/0 forces it on/off.
import os

_INKLING_GATE_SPLITK_MAX_TOKENS = 64
_GATE_SPLITK_BLOCK_E = 32
_GATE_SPLITK_BLOCK_K = 256
_GATE_SPLITK_K_PER_SPLIT = 512


'''

_KERNELS = '''@triton.jit(do_not_specialize=["T"])
def _gate_fp32_splitk_kernel(
    x_ptr,  # [T, K] bf16
    w_ptr,  # [E, K] bf16
    part_ptr,  # [S, T, E] fp32 out
    T,
    E,
    stride_x,
    stride_w,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_t = tl.arange(0, BLOCK_T)
    offs_e = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    mask_t = offs_t < T
    mask_e = offs_e < E
    acc = tl.zeros([BLOCK_T, BLOCK_E], dtype=tl.float32)
    for kk in tl.static_range(0, K_PER_SPLIT, BLOCK_K):
        offs_k = pid_s * K_PER_SPLIT + kk + tl.arange(0, BLOCK_K)
        xt = tl.load(x_ptr + offs_t[:, None] * stride_x + offs_k[None, :],
                     mask=mask_t[:, None], other=0.0)
        wt = tl.load(w_ptr + offs_e[:, None] * stride_w + offs_k[None, :],
                     mask=mask_e[:, None], other=0.0)
        acc = tl.dot(xt, tl.trans(wt), acc)
    out = part_ptr + pid_s * T * E + offs_t[:, None] * E + offs_e[None, :]
    tl.store(out, acc, mask=mask_t[:, None] & mask_e[None, :])


@triton.jit(do_not_specialize=["N"])
def _gate_fp32_splitk_reduce_kernel(
    part_ptr, out_ptr, N, S: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.load(part_ptr + offs, mask=mask, other=0.0)
    for s in tl.static_range(1, S):
        acc += tl.load(part_ptr + s * N + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc, mask=mask)


def _gate_fp32_splitk(
    flat: torch.Tensor,
    weight: torch.Tensor,
    block_e: int = _GATE_SPLITK_BLOCK_E,
    block_k: int = _GATE_SPLITK_BLOCK_K,
    k_per_split: int = _GATE_SPLITK_K_PER_SPLIT,
    num_warps: int = 2,
) -> torch.Tensor:
    t, k = flat.shape
    e = weight.shape[0]
    splits = k // k_per_split
    part = torch.empty((splits, t, e), dtype=torch.float32, device=flat.device)
    _gate_fp32_splitk_kernel[(triton.cdiv(e, block_e), splits)](
        flat, weight, part, t, e, flat.stride(0), weight.stride(0),
        BLOCK_T=max(16, triton.next_power_of_2(t)),
        BLOCK_E=block_e,
        BLOCK_K=block_k,
        K_PER_SPLIT=k_per_split,
        num_warps=num_warps,
    )
    out = torch.empty((t, e), dtype=torch.float32, device=flat.device)
    n = t * e
    _gate_fp32_splitk_reduce_kernel[(triton.cdiv(n, 1024),)](
        part, out, n, S=splits, BLOCK=1024
    )
    return out


'''

_TAIL = '''        and 0 < flat.shape[0] <= _INKLING_GATE_SPLITK_MAX_TOKENS
        and flat.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and flat.is_cuda
        and flat.stride(-1) == 1
        and weight.is_contiguous()
        and flat.shape[1] % _GATE_SPLITK_K_PER_SPLIT == 0
    )


def _linear_with_fp32_out(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    leading = list(x.shape[:-1])
    flat = x.flatten(0, -2)
    if _gate_splitk_ok(flat, weight):
        return _gate_fp32_splitk(flat, weight).view(*leading, weight.shape[0])
    if (
'''

_OK_V1 = '''def _gate_splitk_ok(flat: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        current_platform.is_rocm()
''' + _TAIL

_OK = '''def _gate_splitk_enabled() -> bool:
    if not current_platform.is_rocm():
        return False
    flag = os.environ.get("INKLING_GATE_SPLITK", "")
    if flag not in ("", "0", "1"):
        raise ValueError(f"INKLING_GATE_SPLITK must be 0 or 1, got {flag!r}")
    if flag:
        return flag == "1"
    from vllm.platforms.rocm import on_gfx90a

    return on_gfx90a()


# Resolved once at import (amdsmi, no CUDA init); a plain bool under Dynamo.
_INKLING_GATE_SPLITK = _gate_splitk_enabled()


def _gate_splitk_ok(flat: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        _INKLING_GATE_SPLITK
''' + _TAIL

# NEW_V1 is the block the first version (ac9bcf13) inserted; it is upgraded or
# reverted in place.
NEW_V1 = _HEAD_V1 + _KERNELS + _OK_V1
NEW = _HEAD + _KERNELS + _OK

MARKER = "Zenith patch inkling_gate_fp32_splitk.py"


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
                    help="restore the original _linear_with_fp32_out")
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
        block = NEW if NEW in src else NEW_V1 if NEW_V1 in src else None
        if block is None:
            print(f"ERROR: {path} carries a different version of this patch; "
                  "restore moe.py from the image", file=sys.stderr)
            return 1
        with open(path, "w") as fh:
            fh.write(src.replace(block, OLD, 1))
        print(f"reverted: {path}")
        return 0

    # rocm_inkling_decode_tgemm later edits the tail of NEW, so on a fully
    # patched tree NEW no longer matches verbatim; the gate line marks this version.
    if NEW in src or "\n_INKLING_GATE_SPLITK = _gate_splitk_enabled()\n" in src:
        print(f"already patched: {path}")
        return 0

    if NEW_V1 in src:
        if args.check:
            print(f"would upgrade (gfx90a gate): {path}")
            return 0
        with open(path, "w") as fh:
            fh.write(src.replace(NEW_V1, NEW, 1))
        print(f"upgraded: {path}  (split-K router GEMM now gfx90a-only by default)")
        return 0

    if MARKER in src:
        print(f"ERROR: {path} carries a different version of this patch; "
              "restore moe.py from the image", file=sys.stderr)
        return 1

    if OLD not in src:
        print(f"ERROR: patch target not found in {path}", file=sys.stderr)
        print("vLLM changed _linear_with_fp32_out; check whether the ROCm "
              "router GEMM still falls back to torch.mm(out_dtype=float32), "
              "then re-derive or retire this patch", file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {path}")
        return 0

    with open(path, "w") as fh:
        fh.write(src.replace(OLD, NEW, 1))
    print(f"patched: {path}  (ROCm small-T router GEMM -> split-K Triton, "
          "default on gfx90a only, INKLING_GATE_SPLITK=0/1 overrides)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
