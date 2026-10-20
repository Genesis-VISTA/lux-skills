"""
Triton fused MoE: optionally run GEMM1 (w13) with SiLU-and-mul in its
epilogue, writing intermediate_cache2 directly.

THE PROBLEM

TritonExperts.apply runs the gated activation as its own pass:
1. fused_moe_kernel writes the w13 output, intermediate_cache1
   [M * topk, 2 * I];
2. torch.ops._C.silu_and_mul reads it back and writes intermediate_cache2
   [M * topk, I].
For Inkling-Small at TP8 per rank (I = 256), the pass costs about 125 us at
T=8192 on MI250X, plus the cache1 round trip.

THE FIX

A tuned config entry may carry an optional "g1act" dict, for example:

    "8192": {"BLOCK_SIZE_M": 64, ..., "g1act": {"BLOCK_SIZE_N": 128,
                                                "num_stages": 1}}

When it does, TritonExperts.apply launches _g1act_kernel instead of the
GEMM1 fused_moe_kernel plus activation. The kernel takes the entry overlaid
by the "g1act" keys. Each program computes two accumulators over the same A
rows:
- gate columns n..n+BLOCK_SIZE_N of w13;
- the matching up columns I+n..I+n+BLOCK_SIZE_N.
The epilogue reproduces the served math step by step:
1. round each accumulator to bf16, as the cache1 store does;
2. s = bf16(g / (1 + exp(-g))), as vLLM's silu_kernel does;
3. store bf16(s * u).
On MI250X this is bit-identical to fused_moe_kernel -> silu_and_mul:
- for all 65536 bf16 gate values (silu_exact.py, f61nabwyYLdM);
- on the real InklingMoE layer.

The path is used only when all of these hold:
- the entry has "g1act" and the overlay keeps the entry's BLOCK_SIZE_M,
  because GEMM2 reuses the token order built at that BM;
- the activation is SILU with no clamp, and the class does not override
  TritonExperts.activation;
- bf16, unquantized, no w13 bias, no LoRA, no tensor-descriptor loads.
Otherwise apply runs exactly as stock. invoke_fused_moe_triton_kernel drops
the "g1act" key, so GEMM2 and any other caller see the plain config.
Entries without "g1act" behave exactly as before.

Order-independent with rocm_triton_moe_w2_config.py: the anchors here are
stock lines that patch does not edit.
"""
import argparse
import importlib.util
import os
import sys

MARKER = "Zenith patch rocm_triton_moe_g1act.py"

FUSED_MOE = "vllm/model_executor/layers/fused_moe/fused_moe.py"
TRITON_MOE = "vllm/model_executor/layers/fused_moe/experts/triton_moe.py"

OLD_INVOKE = '''    BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")
    if block_shape is not None:
'''

NEW_INVOKE = '''    BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")
    # Zenith patch rocm_triton_moe_g1act.py: not a kernel argument.
    config.pop("g1act", None)
    if block_shape is not None:
'''

OLD_DECIDE = '''        if lora_context is not None and lora_context.aux_stream is not None:
            # add_inputs=False: kernel overwrites lora_delta_w13. zeros (not
'''

NEW_DECIDE = '''        # Zenith patch rocm_triton_moe_g1act.py: GEMM1 + SiLU-and-mul in one
        # kernel, writing intermediate_cache2 directly.
        g1act = _g1act_ok(
            self, config, activation, hidden_states, w1, intermediate_cache2,
            input_scale, top_k_num, lora_context,
        )

''' + OLD_DECIDE

OLD_W13 = '''        else:
            _base_w13_fn()
            if lora_context is not None:
'''

NEW_W13 = '''        elif g1act:
            _g1act(
                hidden_states, w1, intermediate_cache2, sorted_token_ids,
                expert_ids, num_tokens_post_padded, top_k_num, config,
            )
''' + OLD_W13

OLD_ACT = '''        else:
            self.activation(
                activation, intermediate_cache2, intermediate_cache1.view(-1, N)
            )
'''

NEW_ACT = '''        else:
            if not g1act:  # Zenith patch rocm_triton_moe_g1act.py: already done
                self.activation(
                    activation, intermediate_cache2, intermediate_cache1.view(-1, N)
                )
'''

APPEND = '''

# Zenith patch rocm_triton_moe_g1act.py: GEMM1 with SiLU-and-mul in the
# epilogue (see the patch docstring). Used by TritonExperts.apply when the
# config entry carries "g1act".
@triton.jit
def _g1act_kernel(
    a_ptr,  # [M, K] hidden states
    b_ptr,  # [E, 2 * I, K] w13, gate rows then up rows
    c_ptr,  # [M * topk, I] intermediate_cache2
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N2,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    naive_block_assignment: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
):
    # Program -> tile mapping and token gather as in fused_moe_kernel.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N2, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    if not naive_block_assignment:
        offs_token = tl.load(sorted_token_ids_ptr + pid_m * BLOCK_SIZE_M + offs)
    else:
        offs_token = tl.where(offs == 0, pid_m, num_valid_tokens)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N2)

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # stock: zero cache1 rows, and silu(0) * 0 == 0
        tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.bfloat16),
                 mask=c_mask)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N2
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    bg_ptrs = (b_ptr + off_experts * stride_be
               + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn))
    bu_ptrs = bg_ptrs + N2 * stride_bn
    acc_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        kmask = offs_k < K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & kmask[None, :], other=0.0)
        bg = tl.load(bg_ptrs, mask=kmask[:, None], other=0.0)
        bu = tl.load(bu_ptrs, mask=kmask[:, None], other=0.0)
        acc_g += tl.dot(a, bg)
        acc_u += tl.dot(a, bu)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        bg_ptrs += BLOCK_SIZE_K * stride_bk
        bu_ptrs += BLOCK_SIZE_K * stride_bk
    # cache1 rounding, then silu_and_mul's bf16(silu(g)) * u in fp32
    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16).to(tl.float32)
    s = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(c_ptrs, (s * u).to(tl.bfloat16), mask=c_mask)


def _g1act_ok(experts, config, activation, hidden_states, w1, cache2, input_scale,
              top_k, lora_context) -> bool:
    over = config.get("g1act")
    if not isinstance(over, dict) or lora_context is not None:
        return False
    if over.get("BLOCK_SIZE_M", config["BLOCK_SIZE_M"]) != config["BLOCK_SIZE_M"]:
        return False
    if activation != MoEActivation.SILU or experts.activation_config.clamp_limit is not None:
        return False
    if type(experts).activation is not TritonExperts.activation:
        return False
    qc = experts.quant_config
    if (qc.use_fp8_w8a8 or qc.use_int8_w8a8 or qc.use_int8_w8a16 or qc.use_int4_w4a16
            or input_scale is not None or experts.w1_scale is not None
            or experts.w1_bias is not None):
        return False
    if hidden_states.dtype != torch.bfloat16 or w1.dtype != torch.bfloat16:
        return False
    if cache2.dtype != torch.bfloat16 or not cache2.is_contiguous():
        return False
    if hidden_states.dim() != 2 or w1.dim() != 3 or w1.stride(2) != 1:
        return False
    n = w1.shape[1]
    if n % 2 or tuple(cache2.shape) != (hidden_states.shape[0] * top_k, n // 2):
        return False
    from vllm.model_executor.layers.fused_moe import fused_moe as _fm

    return not _fm.resolve_moe_use_td()


_G1ACT_LAUNCH_KEYS = ("GROUP_SIZE_M", "num_warps", "num_stages", "waves_per_eu",
                      "matrix_instr_nonkdim", "kpack")


def _g1act(hidden_states, w1, cache2, sorted_token_ids, expert_ids,
           num_tokens_post_padded, top_k, config) -> None:
    cfg = {**config, **config["g1act"]}
    bm, bn = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"]
    m = hidden_states.size(0)
    num_tokens = m * top_k
    if sorted_token_ids is not None:
        em = sorted_token_ids.size(0)
        if m < bm:
            # as invoke_fused_moe_triton_kernel: skip blocks that cannot be used
            em = min(em, m * top_k * bm)
    else:
        em = num_tokens * bm
    n2 = w1.size(1) // 2
    kw = {k: cfg[k] for k in _G1ACT_LAUNCH_KEYS if k in cfg}
    kw.setdefault("GROUP_SIZE_M", 1)
    _g1act_kernel[(triton.cdiv(em, bm) * triton.cdiv(n2, bn),)](
        hidden_states, w1, cache2, sorted_token_ids, expert_ids, num_tokens_post_padded,
        n2, w1.size(2), em, num_tokens,
        hidden_states.stride(0), hidden_states.stride(1),
        w1.stride(0), w1.stride(2), w1.stride(1),
        cache2.stride(0), cache2.stride(1),
        naive_block_assignment=sorted_token_ids is None,
        BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
        top_k=top_k, **kw,
    )
'''

# (target, ((old, new), ...)); triton_moe.py anchors are matched before the
# TritonWNA16Experts class, which this patch leaves alone. APPEND goes at
# the end of triton_moe.py.
TARGETS = (
    (FUSED_MOE, ((OLD_INVOKE, NEW_INVOKE),)),
    (TRITON_MOE, ((OLD_DECIDE, NEW_DECIDE), (OLD_W13, NEW_W13), (OLD_ACT, NEW_ACT))),
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
                    help="restore the separate GEMM1 and activation launches")
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
            if n != 1:
                missing.append(f"{rel}[{i}]")
    if missing:
        print(f"ERROR: patch target(s) {missing} not found exactly once", file=sys.stderr)
        print("vLLM changed how TritonExperts.apply runs GEMM1 and the "
              "activation; re-derive or retire this patch", file=sys.stderr)
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
    print(f"patched: {root}  (TritonExperts GEMM1 takes a config entry's \"g1act\" "
          "overlay: SiLU-and-mul fused in the epilogue)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
