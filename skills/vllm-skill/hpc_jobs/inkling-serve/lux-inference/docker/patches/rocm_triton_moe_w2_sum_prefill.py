"""
TritonExperts at prefill sizes on ROCm (MI250X): GEMM2 at 16 warps above
T=8192, and a moe_sum launch whose rank-major slab stores vectorize. Both
are bit-identical to the kernels they replace.

Needs rocm_triton_moe_sum.py and rocm_triton_moe_w2_config.py (this file
edits code those two add); it sorts after both in patches/.

THE PROBLEM

1. _w2_astat (GEMM2, A-stationary) runs the tuned 8192-token entry
   (BM 64, BN 128, BK 128, 8 warps, 2 stages, ASTAT_GROUPS 2) for every
   prefill chunk from 8192 tokens up, 16k chunks included. There it sits at
   about 43% of the MFMA rate, and B traffic is not the limit: remapping
   every block to one L2-resident expert saves only 2%. With more blocks
   than at 8192, 16 warps hide more of the load/LDS latency.
2. _moe_sum_kernel takes slab_stride as an unspecialized runtime argument,
   so the compiler cannot prove the rank-major slab address is aligned and
   splits every store. 2048 columns x 8 warps per program is also not the
   best launch shape for a [T, 6, 4096] -> [8, T, 512] sum.

THE FIX

1. _w2_astat launches with 16 warps when the A operand has more than
   _W2_ASTAT_W16_MIN_ROWS rows (T * topk > 8192 * 6) and the config is
   exactly that tuned entry. Each tile's K reduction is the same MFMA chain
   in the same order, so c3 is bit-identical. At T=8192 16 warps is within
   noise (-9 / +13 us on layers 11 / 20), so it stays at 8 there.
2. _moe_sum_kernel computes the slab stride in-kernel as M * SEG (SEG is a
   constexpr, so the stride is provably a multiple of SEG), and launches
   512 columns x 2 warps per program. Same fp32 accumulation in top-k
   order and the same roundings, so the output is bit-identical.

Measured on one MI250X GCD (Tioga, job f61wu9dFE78j, moe_val.py):
Inkling-Small-bf16 TP8 per-rank shapes, real layer 11 / 20 routed weights
and captured routing, served env. The chain is _g1act -> _w2_astat ->
moe_sum with the sink addend into rank-major slabs, as TritonExperts.apply
runs it. Served and patched triton_moe.py were loaded side by side and
timed as interleaved medians; c3 and slabs are torch.equal on every row.

| per layer, L11 / L20 | T | served | patched |
|---|---|---|---|
| GEMM2 | 12288 | 2049 / 2052 us | 1996 / 1988 us |
| GEMM2 | 16384 | 2628 / 2688 us | 2501 / 2594 us |
| moe_sum + sink addend | 8192 | 502 / 502 us | 470 / 471 us |
| moe_sum + sink addend | 12288 | 768 / 767 us | 711 / 712 us |
| moe_sum + sink addend | 16384 | 1009 / 1001 us | 931 / 927 us |
| chain | 8192 | 4153 / 4170 us | 4109 / 4146 us |
| chain | 12288 | 5955 / 5960 us | 5839 / 5838 us |
| chain | 16384 | 7721 / 7772 us | 7524 / 7610 us |

(A pure streaming kernel moving the moe_sum bytes takes 457 / 907 us at
8192 / 16384.)

INKLING_MOE_W2_W16=0 keeps GEMM2 at the config's warps; the moe_sum change
has no switch (it is the same arithmetic with a different launch shape).
"""
import argparse
import importlib.util
import os
import sys

TARGET = "vllm/model_executor/layers/fused_moe/experts/triton_moe.py"

OLD_SUM_SIG = '''@triton.jit(do_not_specialize=["M", "slab_stride"])
def _moe_sum_kernel(
    in_ptr,  # [M, TOPK, N]
    out_ptr,  # [M, N], or [N // SEG, M, SEG] slabs
    add_ptr,  # [M, N] (read only if HAS_ADD)
    M,
    slab_stride,  # M * SEG for slabs, 0 for a plain [M, N] output
    N: tl.constexpr,
    SEG: tl.constexpr,  # N for a plain output
'''

NEW_SUM_SIG = '''@triton.jit(do_not_specialize=["M"])
def _moe_sum_kernel(
    in_ptr,  # [M, TOPK, N]
    out_ptr,  # [M, N], or [N // SEG, M, SEG] slabs
    add_ptr,  # [M, N] (read only if HAS_ADD)
    M,
    N: tl.constexpr,
    SEG: tl.constexpr,  # N for a plain output
'''

OLD_SUM_DST = '''    dst = (
        (offs_n[None, :] // SEG).to(tl.int64) * slab_stride
        + rows * SEG
        + offs_n[None, :] % SEG
    )
'''

NEW_SUM_DST = '''    # Zenith patch rocm_triton_moe_w2_sum_prefill.py: slab stride M * SEG
    # computed here, not passed in, so the stores are provably aligned (for a
    # plain output SEG == N and the slab index is always 0).
    dst = (
        (offs_n[None, :] // SEG).to(tl.int64) * (M.to(tl.int64) * SEG)
        + rows * SEG
        + offs_n[None, :] % SEG
    )
'''

OLD_SUM_LAUNCH = '''    if slabs is None:
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
'''

NEW_SUM_LAUNCH = '''    if slabs is None:
        out, seg = output, n
    else:
        out, seg = slabs, slabs.shape[2]
    grid = (triton.cdiv(m, _MOE_SUM_BLOCK_M), triton.cdiv(n, _MOE_SUM_BLOCK_N))
    _moe_sum_kernel[grid](
        input,
        out,
        add if add is not None else output,
        m,
        N=n,
        SEG=seg,
        TOPK=topk,
        BLOCK_M=_MOE_SUM_BLOCK_M,
        BLOCK_N=_MOE_SUM_BLOCK_N,
        HAS_ADD=add is not None,
        num_warps=_MOE_SUM_WARPS,
    )
'''

OLD_SUM_BLOCK = '''_MOE_SUM_BLOCK_M = 1
_MOE_SUM_BLOCK_N = 2048
'''

NEW_SUM_BLOCK = '''_MOE_SUM_BLOCK_M = 1
# (patch rocm_triton_moe_w2_sum_prefill.py) was 2048 columns x 8 warps
_MOE_SUM_BLOCK_N = 512
_MOE_SUM_WARPS = 2
'''

OLD_W2_DEF = '''def _w2_astat(a, w2, c3, topk_weights, sorted_token_ids, expert_ids,
              num_tokens_post_padded, config) -> None:
'''

NEW_W2_DEF = '''# Zenith patch rocm_triton_moe_w2_sum_prefill.py: the tuned 8192-token GEMM2
# entry runs 16 warps above T * topk = 8192 * 6 rows (same MFMA chain per
# tile, bit-identical; MI250X Inkling-Small: -53..-64 us at T=12288,
# -94..-127 us at T=16384 per layer). INKLING_MOE_W2_W16=0 disables.
import os as _w2_os  # noqa: E402

_W2_ASTAT_W16_MIN_ROWS = 8192 * 6
_W2_ASTAT_W16 = _w2_os.environ.get("INKLING_MOE_W2_W16", "1") != "0"


def _w2_astat_warps(a, config) -> int:
    warps = config.get("num_warps", 8)
    if (
        _W2_ASTAT_W16
        and a.size(0) > _W2_ASTAT_W16_MIN_ROWS
        and warps == 8
        and config.get("num_stages", 2) == 2
        and config.get("waves_per_eu", 0) == 0
        and config["BLOCK_SIZE_M"] == 64
        and config["BLOCK_SIZE_N"] == 128
        and config["BLOCK_SIZE_K"] == 128
    ):
        return 16
    return warps


def _w2_astat(a, w2, c3, topk_weights, sorted_token_ids, expert_ids,
              num_tokens_post_padded, config) -> None:
'''

OLD_W2_WARPS = '''        num_warps=config.get("num_warps", 8), num_stages=config.get("num_stages", 2),
        waves_per_eu=config.get("waves_per_eu", 0),
        matrix_instr_nonkdim=config.get("matrix_instr_nonkdim", 16),
        kpack=config.get("kpack", 1),
    )
'''

NEW_W2_WARPS = '''        num_warps=_w2_astat_warps(a, config), num_stages=config.get("num_stages", 2),
        waves_per_eu=config.get("waves_per_eu", 0),
        matrix_instr_nonkdim=config.get("matrix_instr_nonkdim", 16),
        kpack=config.get("kpack", 1),
    )
'''

MARKER = "Zenith patch rocm_triton_moe_w2_sum_prefill.py"
EDITS = (
    (OLD_SUM_BLOCK, NEW_SUM_BLOCK),
    (OLD_SUM_SIG, NEW_SUM_SIG),
    (OLD_SUM_DST, NEW_SUM_DST),
    (OLD_SUM_LAUNCH, NEW_SUM_LAUNCH),
    (OLD_W2_DEF, NEW_W2_DEF),
    (OLD_W2_WARPS, NEW_W2_WARPS),
)
PREREQS = ("Zenith patch rocm_triton_moe_sum.py", "Zenith patch rocm_triton_moe_w2_config.py")


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
                    help="restore the rocm_triton_moe_sum / _w2_config versions")
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
        if any(src.count(new) != 1 for _, new in EDITS):
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

    missing_pre = [p for p in PREREQS if p not in src]
    if missing_pre:
        print(f"ERROR: {path} lacks {missing_pre}; apply rocm_triton_moe_sum.py "
              "and rocm_triton_moe_w2_config.py first", file=sys.stderr)
        return 1

    missing = [i for i, (old, _) in enumerate(EDITS) if src.count(old) != 1]
    if missing:
        print(f"ERROR: patch target(s) {missing} not found exactly once in {path}",
              file=sys.stderr)
        print("rocm_triton_moe_sum.py or rocm_triton_moe_w2_config.py changed "
              "_moe_sum_kernel / _w2_astat; re-derive this patch against them",
              file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {path}")
        return 0

    for old, new in EDITS:
        src = src.replace(old, new, 1)
    with open(path, "w") as fh:
        fh.write(src)
    print(f"patched: {path}  (ROCm prefill GEMM2 16 warps above T=8192, "
          "moe_sum aligned slab stores)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
