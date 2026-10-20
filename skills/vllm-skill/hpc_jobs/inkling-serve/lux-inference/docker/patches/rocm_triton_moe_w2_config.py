"""
Triton fused MoE: let a tuned config entry give GEMM2 (w2) its own tile
config instead of reusing GEMM1's (w13), and optionally run GEMM2 with an
"A-stationary" Triton kernel.

THE PROBLEM

try_get_optimal_moe_config returns one config per batch size, and
TritonExperts.apply passes it to both fused_moe_kernel launches. The two
GEMMs have very different shapes. For Inkling-Small at TP8 per rank:
- GEMM1 is K=4096, N=512;
- GEMM2 is K=256, N=4096.
So they want different tiles. GEMM2 alone on MI250X (gemm2_occ.py,
f61mxtJ7di71) runs 12-35% faster at M=8192..2048 with num_stages=1,
BLOCK_SIZE_K=128 and matrix_instr_nonkdim=16 than with the shared tuned
config. A shared config cannot take that without moving GEMM1 too.

THE FIX

A config entry may carry an optional "w2" dict, for example:

    "8192": {"BLOCK_SIZE_M": 128, ..., "num_stages": 2,
             "w2": {"BLOCK_SIZE_K": 128, "num_warps": 4, "num_stages": 1,
                    "matrix_instr_nonkdim": 16}}

1. TritonExperts.apply runs GEMM2 with the entry overlaid by its "w2" keys.
2. If the overlay changes BLOCK_SIZE_M, apply also builds GEMM2's own
   expert-aligned token order.
3. invoke_fused_moe_triton_kernel drops the "w2" key before launching, so
   GEMM1, and any other caller handed such an entry, see the plain config.

An overlay with "ASTAT": 1 (and "ASTAT_GROUPS": G) runs GEMM2 with
_w2_astat_kernel instead of fused_moe_kernel. fused_moe_kernel runs one
program per (BLOCK_SIZE_M rows, BLOCK_SIZE_N cols) tile. At K=256 that is a
handful of K steps per program, and its LDS use allows one program per CU,
so memory latency is exposed. _w2_astat_kernel instead:
1. loads a block's gathered A rows [BLOCK_SIZE_M, K] once and keeps them in
   registers;
2. loops over N / (BLOCK_SIZE_N * G) column tiles, with the B loads
   pipelined;
3. finishes each tile with the same epilogue: fp32 accumulator times the
   fp32 top-k weight, one cast, scatter store.
GEMM2 alone on MI250X (g2_astat.py, f61n4xqgeFM9), in us, shared tuned config
vs A-stationary BM64 BN128 BK128 G2 w8 st2 nk16:
- M=16384: 3579 -> 2744
- M=8192: 2122 -> 1536
- M=4096: 1482 -> 1012
- M=2048: 1152 -> 749
Output is torch.equal to fused_moe_kernel's in all 480 variants tested,
across BK 64-256 and nonkdim 16/32. The kernel is used only for unquantized
bf16/fp16 GEMM2 with top-k weights applied in the epilogue, no bias, no
expert_map, and shapes it tiles exactly. Otherwise an "ASTAT" entry runs
GEMM2 exactly as stock vLLM does: fused_moe_kernel with the shared config
and token order. The A-stationary tiles need not be valid for
fused_moe_kernel; BK256 at BM64 BN128, for one, needs 96 KiB of LDS.

Entries without "w2" behave exactly as before: same configs, same launches,
bit-identical output.
"""
import argparse
import importlib.util
import os
import sys

MARKER = "Zenith patch rocm_triton_moe_w2_config.py"

FUSED_MOE = "vllm/model_executor/layers/fused_moe/fused_moe.py"
TRITON_MOE = "vllm/model_executor/layers/fused_moe/experts/triton_moe.py"

OLD_INVOKE = '''    config = config.copy()
    config["SPLIT_K"] = 1
    BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")
'''

NEW_INVOKE = '''    config = config.copy()
    # Zenith patch rocm_triton_moe_w2_config.py: GEMM2's overlay and the
    # A-stationary keys are not kernel arguments.
    for key in ("w2", "ASTAT", "ASTAT_GROUPS"):
        config.pop(key, None)
    config["SPLIT_K"] = 1
    BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")
'''

OLD_ALIGN = '''        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            _prepare_expert_assignment(
                topk_ids,
                config,
                num_tokens,
                top_k_num,
                num_align_experts,
                expert_map,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                block_shape=self.block_shape,
            )
        )
'''

NEW_ALIGN = OLD_ALIGN + '''
        # Zenith patch rocm_triton_moe_w2_config.py: GEMM2 runs with the
        # entry's "w2" overlay, with its own token order if BLOCK_SIZE_M
        # changes.
        config_w2 = config
        sorted_token_ids_w2, expert_ids_w2, num_tokens_post_padded_w2 = (
            sorted_token_ids, expert_ids, num_tokens_post_padded
        )
        if isinstance(config.get("w2"), dict):
            config_w2 = {**config, **config["w2"]}
            del config_w2["w2"]
            if config_w2["BLOCK_SIZE_M"] != config["BLOCK_SIZE_M"]:
                sorted_token_ids_w2, expert_ids_w2, num_tokens_post_padded_w2 = (
                    _prepare_expert_assignment(
                        topk_ids,
                        config_w2,
                        num_tokens,
                        top_k_num,
                        num_align_experts,
                        expert_map,
                        use_int8_w8a16=self.quant_config.use_int8_w8a16,
                        use_int4_w4a16=self.quant_config.use_int4_w4a16,
                        block_shape=self.block_shape,
                    )
                )
'''

OLD_W2 = '''        def _base_w2_fn():
            invoke_fused_moe_triton_kernel(
                qintermediate_cache2,
                w2,
                intermediate_cache3,
                a2q_scale,
                self.w2_scale,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                not apply_router_weight_on_input,
                1,
                config,
                compute_type=compute_type,
                use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
'''

NEW_W2 = '''        def _base_w2_fn():
            # Zenith patch rocm_triton_moe_w2_config.py
            if _w2_astat_ok(
                config_w2, qintermediate_cache2, w2, intermediate_cache3,
                a2q_scale, self.w2_scale, self.w2_bias, topk_weights,
                sorted_token_ids_w2, expert_map, apply_router_weight_on_input,
            ):
                _w2_astat(
                    qintermediate_cache2, w2, intermediate_cache3, topk_weights,
                    sorted_token_ids_w2, expert_ids_w2, num_tokens_post_padded_w2,
                    config_w2,
                )
                return
            w2_config, w2_ids, w2_eids, w2_ntpp = (
                config_w2, sorted_token_ids_w2, expert_ids_w2, num_tokens_post_padded_w2
            )
            if config_w2.get("ASTAT"):
                # A-stationary tiles fused_moe_kernel may not fit: run stock.
                w2_config, w2_ids, w2_eids, w2_ntpp = (
                    config, sorted_token_ids, expert_ids, num_tokens_post_padded
                )
            invoke_fused_moe_triton_kernel(
                qintermediate_cache2,
                w2,
                intermediate_cache3,
                a2q_scale,
                self.w2_scale,
                topk_weights,
                w2_ids,
                w2_eids,
                w2_ntpp,
                not apply_router_weight_on_input,
                1,
                w2_config,
                compute_type=compute_type,
                use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
'''

APPEND = '''

# Zenith patch rocm_triton_moe_w2_config.py: A-stationary GEMM2 (see the
# patch docstring). Used by TritonExperts.apply when its GEMM2 config
# carries "ASTAT": 1.
from vllm.triton_utils import triton as _astat_triton  # noqa: E402


@_astat_triton.jit
def _w2_astat_kernel(
    a_ptr,  # [M * topk, K] activations, row = sorted token id
    b_ptr,  # [E, N, K] w2, stride 1 along K
    c_ptr,  # [M * topk, N] rows, stride 1 along N
    tw_ptr,  # [M * topk] top-k weights
    sorted_ptr,
    eid_ptr,
    ntpp_ptr,
    num_valid,
    stride_am,
    stride_be,
    stride_bn,
    stride_cm,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NPP: tl.constexpr,  # column tiles per program
):
    NK: tl.constexpr = K // BK
    tl.static_assert(NK == 1 or NK == 2 or NK == 4)
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    if pid_m * BM >= tl.load(ntpp_ptr):
        return
    tok = tl.load(sorted_ptr + pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    tmask = tok < num_valid
    e = tl.load(eid_ptr + pid_m).to(tl.int64)
    w = tl.load(tw_ptr + tok, mask=tmask, other=0)
    offs_k = tl.arange(0, BK)
    a_rows = a_ptr + tok[:, None] * stride_am + offs_k[None, :]
    a0 = tl.load(a_rows, mask=tmask[:, None], other=0.0)
    if NK >= 2:
        a1 = tl.load(a_rows + BK, mask=tmask[:, None], other=0.0)
    if NK >= 4:
        a2 = tl.load(a_rows + 2 * BK, mask=tmask[:, None], other=0.0)
        a3 = tl.load(a_rows + 3 * BK, mask=tmask[:, None], other=0.0)
    b_base = b_ptr + e * stride_be + offs_k[:, None]
    c_rows = c_ptr + tok[:, None] * stride_cm
    for j in range(0, NPP):
        offs_n = (pid_g * NPP + j) * BN + tl.arange(0, BN)
        bp = b_base + offs_n[None, :] * stride_bn
        acc = tl.dot(a0, tl.load(bp))
        if NK >= 2:
            acc += tl.dot(a1, tl.load(bp + BK))
        if NK >= 4:
            acc += tl.dot(a2, tl.load(bp + 2 * BK))
            acc += tl.dot(a3, tl.load(bp + 3 * BK))
        acc *= w[:, None]
        tl.store(c_rows + offs_n[None, :], acc.to(c_ptr.dtype.element_ty),
                 mask=tmask[:, None])


def _w2_astat_ok(
    config, a, w2, c3, a_scale, w2_scale, w2_bias, topk_weights,
    sorted_token_ids, expert_map, apply_router_weight_on_input,
) -> bool:
    if not config.get("ASTAT") or sorted_token_ids is None or expert_map is not None:
        return False
    if a_scale is not None or w2_scale is not None or w2_bias is not None:
        return False
    if apply_router_weight_on_input or topk_weights is None:
        return False
    if a.dtype not in (torch.bfloat16, torch.float16) or w2.dtype != a.dtype:
        return False
    if c3.dtype != a.dtype or c3.dim() != 3 or a.dim() != 2:
        return False
    if a.stride(1) != 1 or w2.stride(2) != 1 or c3.stride(2) != 1:
        return False
    if not topk_weights.is_contiguous() or topk_weights.numel() != a.shape[0]:
        return False
    if c3.shape[0] * c3.shape[1] != a.shape[0] or c3.stride(0) != c3.shape[1] * c3.stride(1):
        return False
    k, n = w2.shape[2], w2.shape[1]
    bk, bn = config["BLOCK_SIZE_K"], config["BLOCK_SIZE_N"]
    groups = config.get("ASTAT_GROUPS", 1)
    return (
        a.shape[1] == k
        and c3.shape[2] == n
        and k % bk == 0
        and k // bk in (1, 2, 4)
        and n % (bn * groups) == 0
    )


def _w2_astat(a, w2, c3, topk_weights, sorted_token_ids, expert_ids,
              num_tokens_post_padded, config) -> None:
    bm, bn = config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"]
    groups = config.get("ASTAT_GROUPS", 1)
    n = w2.shape[1]
    em = sorted_token_ids.size(0)
    if a.size(0) < bm:
        # as invoke_fused_moe_triton_kernel: at most a.size(0) blocks are used
        em = min(em, a.size(0) * bm)
    grid = (_astat_triton.cdiv(em, bm), groups)
    _w2_astat_kernel[grid](
        a, w2, c3, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
        a.size(0), a.stride(0), w2.stride(0), w2.stride(1), c3.stride(1),
        K=w2.shape[2], BM=bm, BN=bn, BK=config["BLOCK_SIZE_K"], NPP=n // bn // groups,
        num_warps=config.get("num_warps", 8), num_stages=config.get("num_stages", 2),
        waves_per_eu=config.get("waves_per_eu", 0),
        matrix_instr_nonkdim=config.get("matrix_instr_nonkdim", 16),
        kpack=config.get("kpack", 1),
    )
'''

# (target, ((old, new), ...)); OLD_W2 is matched before the
# TritonWNA16Experts class, which this patch leaves alone. APPEND goes at
# the end of triton_moe.py.
TARGETS = (
    (FUSED_MOE, ((OLD_INVOKE, NEW_INVOKE),)),
    (TRITON_MOE, ((OLD_ALIGN, NEW_ALIGN), (OLD_W2, NEW_W2))),
)


def vllm_root():
    """Locate the installed vLLM without importing it (importing vllm
    touches the GPU at module scope)."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        sys.exit("vllm is not installed; run this inside the container")
    return os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))


def count_before_wna16(src, old):
    """Occurrences of old in TritonExperts, i.e. before TritonWNA16Experts."""
    cut = src.find("class TritonWNA16Experts")
    return src[: cut if cut >= 0 else len(src)].count(old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report status without writing")
    ap.add_argument("--revert", action="store_true",
                    help="restore the shared GEMM1/GEMM2 config")
    ap.add_argument("--root", help="patch the vLLM tree under this directory "
                    "instead of the installed one")
    args = ap.parse_args()

    root = args.root or vllm_root()
    srcs = {}
    for rel, _ in TARGETS:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            sys.exit(f"expected {path} to exist; vLLM layout changed")
        with open(path) as fh:
            srcs[rel] = fh.read()

    patched = [rel for rel, _ in TARGETS if MARKER in srcs[rel]]

    if args.revert:
        if not patched:
            print(f"not patched: {root}")
            return 0
        for rel, edits in TARGETS:
            if rel not in patched:
                continue
            if any(srcs[rel].count(new) != 1 for _, new in edits) or (
                rel == TRITON_MOE and srcs[rel].count(APPEND) != 1
            ):
                print(f"ERROR: {rel} carries a different version of this patch; "
                      "restore it from the image", file=sys.stderr)
                return 1
        for rel, edits in TARGETS:
            if rel not in patched:
                continue
            src = srcs[rel]
            for old, new in edits:
                src = src.replace(new, old, 1)
            if rel == TRITON_MOE:
                src = src.replace(APPEND, "", 1)
            with open(os.path.join(root, rel), "w") as fh:
                fh.write(src)
        print(f"reverted: {root}")
        return 0

    if len(patched) == len(TARGETS):
        print(f"already patched: {root}")
        return 0
    if patched:
        print(f"ERROR: only {patched} carry this patch; restore them from the "
              "image and re-apply", file=sys.stderr)
        return 1

    missing = []
    for rel, edits in TARGETS:
        for i, (old, _) in enumerate(edits):
            n = (count_before_wna16(srcs[rel], old) if rel == TRITON_MOE
                 else srcs[rel].count(old))
            if n != 1 or srcs[rel].count(old) < 1:
                missing.append(f"{rel}[{i}]")
    if missing:
        print(f"ERROR: patch target(s) {missing} not found exactly once", file=sys.stderr)
        print("vLLM changed how TritonExperts.apply configures its two "
              "fused_moe_kernel launches; re-derive or retire this patch",
              file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {root}")
        return 0

    for rel, edits in TARGETS:
        src = srcs[rel]
        for old, new in edits:
            src = src.replace(old, new, 1)
        if rel == TRITON_MOE:
            src += APPEND
        with open(os.path.join(root, rel), "w") as fh:
            fh.write(src)
    print(f"patched: {root}  (TritonExperts GEMM2 takes a config entry's \"w2\" overlay)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
