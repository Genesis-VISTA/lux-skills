#!/usr/bin/env python3
"""
Split the KV range of Inkling's global-layer attention across programs when
each request has 2-4 query tokens (the MTP verify step).

THE BUG AS IT PRESENTS

With MTP 3 every decode step verifies 4 query tokens per request. The split-KV
decode kernel (rel_attention_decode.py) only accepts max_query_len == 1, so the
verify falls back to the single-pass _inkling_rel_attention_kernel. On the 7
global layers that kernel launches one program per (request, KV head), and at
TP8 there is one KV head per rank. At batch 1 a single workgroup walks the whole
~8k-token context while the rest of the GCD's 110 CUs sit idle. Global-layer
attention cost 641-915 us per call at B = 1-128 (gbench f61hkVS7bnSs), ~29 ms
per pass at c=1 in the 2-node profile.

THE FIX

For global layers (window_left < 0) with 1 < max_seqlen_q <= 4 the KV range
[0, kv_len) of each request is split across num_splits programs. Each runs the
same online-softmax loop as the single-pass kernel over its share and writes an
unnormalized partial (acc, row max, row sum) in base 2. A second kernel merges
the partials and writes only the request's real query rows. Padded CUDA-graph
slots (q_len 0) are skipped by both kernels, exactly as the single-pass kernel
skips them, so inkling_attn_output_zero_init.py still covers them.

num_splits depends only on the batch size and max_kv_len, never on the data, and
each program sizes its share from the request's real kv_len. A graph captured
with max_kv_len = max_model_len is therefore correct for any context. Splits
that end up empty write (0, -inf, 0) and contribute nothing.

The result differs from the single-pass kernel only by summation order (fp32
accumulation, bf16 output). INKLING_VERIFY_SPLIT_KV=0 turns the path off.

SPLIT COUNT

num_splits = min(32, target // (batch * kv_heads)), at least 1. The target
number of programs is 256 up to 32 (request, KV head) pairs and 1024 above:
small batches are latency bound and lose to the merge cost of many short
splits, large ones are bandwidth bound and need the extra programs to reach it.
INKLING_VERIFY_SPLIT_TARGET fixes the target instead. Global-layer verify, qlen
4, ctx 8-9k, one GCD, graph replay, us per call (gbench f61isQaXUFCK,
f61j5HpVAbtb):

    B             1     8     16    32    64    128    256
    single-pass   655   708   700   715   734   1107   1425
    split         46    74    111   185   305   620    1197
"""
import argparse
import importlib.util
import os
import sys

TARGET = "vllm/models/inkling/amd/ops/fa4_rel_attention.py"
MARKER = "patch inkling_verify_split_kv.py"

OLD_KERNEL_ANCHOR = '''@torch.no_grad()
def inkling_fa4_rel_attention(
'''

NEW_KERNEL = '''def verify_split_count(*, batch: int, num_kv_heads: int, max_kv_len: int) -> int:
    """Number of KV splits for the global-layer verify path (1 = do not split).

    Depends only on values fixed at CUDA-graph capture. Zenith patch
    inkling_verify_split_kv.py.
    """
    ctas = max(1, batch * num_kv_heads)
    target = int(os.getenv("INKLING_VERIFY_SPLIT_TARGET", "0")) or (256 if ctas <= 32 else 1024)
    by_ctas = target // ctas
    by_len = triton.cdiv(max(1, max_kv_len), 256)
    return max(1, min(32, by_ctas, by_len))


def use_verify_split_kv(*, max_query_len: int, window_left: int, num_splits: int) -> bool:
    return (
        os.getenv("INKLING_VERIFY_SPLIT_KV", "1") == "1"
        and 1 < max_query_len <= 4
        and window_left < 0
        and num_splits > 1
    )


@triton.heuristics(
    {
        "BLOCK_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_QH": lambda args: (
            args["BLOCK_Q"] * triton.next_power_of_2(args["gqa_group_size"])
        ),
    }
)
@triton.jit(do_not_specialize_on_alignment=["cache_seqlens", "cu_seqlens_q"])
def _inkling_rel_attention_split_kernel(
    q_ptr,  # [total_q, Hq, D]
    k_ptr,  # [blocks, page, Hkv, D]
    v_ptr,  # [blocks, page, Hkv, D]
    rel_ptr,  # [total_q, Hq, rel_extent]
    mid_acc_ptr,  # [batch, Hkv, splits, BLOCK_QH, BLOCK_D] fp32
    mid_ml_ptr,  # [batch, Hkv, splits, 2, BLOCK_QH] fp32 (row max, row sum)
    block_table_ptr,  # [batch, max_pages]
    cache_seqlens,
    cu_seqlens_q,
    gqa_group_size,
    head_dim,
    softmax_scale,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_k_b,
    stride_k_p,
    stride_k_h,
    stride_k_d,
    stride_v_b,
    stride_v_p,
    stride_v_h,
    stride_v_d,
    stride_r_t,
    stride_r_h,
    stride_r_e,
    stride_bt_b,
    page_size: tl.constexpr,
    rel_extent: tl.constexpr,
    window_left: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_QH: tl.constexpr,
):
    # Same body as _inkling_rel_attention_kernel for one query block of
    # BLOCK_Q >= q_len rows, restricted to this program's share of the keys.
    pid_s = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)

    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    if q_len <= 0:
        return

    kv_len = tl.load(cache_seqlens + pid_b)
    prefix_len = kv_len - q_len
    q_head_start = pid_kh * gqa_group_size
    bt_row = block_table_ptr + pid_b * stride_bt_b

    k_lo = 0
    if window_left >= 0:
        k_lo = tl.maximum(prefix_len - window_left, 0) // BLOCK_K * BLOCK_K
    k_hi = kv_len
    split_len = tl.cdiv(tl.cdiv(k_hi - k_lo, num_splits), BLOCK_K) * BLOCK_K
    split_start = k_lo + pid_s * split_len
    split_end = tl.minimum(split_start + split_len, k_hi)

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_start * stride_q_t + q_head_start * stride_q_h,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_q_t, stride_q_h, stride_q_d),
        offsets=(0, 0, 0),
        block_shape=(BLOCK_Q, BLOCK_H, BLOCK_D),
        order=(2, 1, 0),
    )
    q = tl.load(q_block_ptr, boundary_check=(0, 1, 2), padding_option="zero")
    q = tl.reshape(q, (BLOCK_QH, BLOCK_D))

    q_rows = tl.arange(0, BLOCK_Q)
    q_abs = prefix_len + q_rows
    q_valid = q_rows < q_len
    off_d = tl.arange(0, BLOCK_D)
    d_valid = off_d < head_dim

    m_i = tl.full((BLOCK_QH,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_QH,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_QH, BLOCK_D), dtype=tl.float32)
    log2e: tl.constexpr = 1.4426950408889634

    for k_start in range(split_start, split_end, BLOCK_K):
        k_pos = k_start + tl.arange(0, BLOCK_K)
        k_valid = k_pos < split_end
        safe_pos = tl.minimum(k_pos, tl.maximum(kv_len - 1, 0))
        page = tl.load(bt_row + safe_pos // page_size).to(tl.int64)
        page_off = safe_pos % page_size

        k = tl.load(
            k_ptr
            + page[None, :] * stride_k_b
            + page_off[None, :] * stride_k_p
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d,
            mask=d_valid[:, None] & k_valid[None, :],
            other=0.0,
        )
        scores = tl.dot(q, k) * (softmax_scale * log2e)

        dist = q_abs[:, None] - k_pos[None, :]
        score_valid = q_valid[:, None] & k_valid[None, :] & (dist >= 0)
        if window_left >= 0:
            score_valid &= dist <= window_left

        off_h = tl.arange(0, BLOCK_H)
        rel_dist = dist[:, None, :]
        rel_valid = (
            q_valid[:, None, None]
            & (off_h[None, :, None] < gqa_group_size)
            & (rel_dist >= 0)
            & (rel_dist < rel_extent)
        )
        safe_dist = tl.maximum(0, tl.minimum(rel_dist, rel_extent - 1))
        bias = tl.load(
            rel_ptr
            + (q_start + q_rows[:, None, None]) * stride_r_t
            + (q_head_start + off_h[None, :, None]) * stride_r_h
            + safe_dist * stride_r_e,
            mask=rel_valid,
            other=0.0,
        )
        bias = tl.reshape(bias, (BLOCK_QH, BLOCK_K))
        scores += bias.to(tl.float32) * log2e
        score_valid = tl.reshape(
            score_valid[:, None, :] & (off_h[None, :, None] < gqa_group_size),
            (BLOCK_QH, BLOCK_K),
        )
        scores = tl.where(score_valid, scores, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        has_scores = tl.sum(score_valid.to(tl.int32), axis=1) > 0
        alpha = tl.where(has_scores, tl.exp2(m_i - m_ij), 1.0)
        p = tl.where(
            score_valid,
            tl.exp2(scores - m_ij[:, None]),
            0.0,
        )
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc *= alpha[:, None]

        v = tl.load(
            v_ptr
            + page[:, None] * stride_v_b
            + page_off[:, None] * stride_v_p
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d,
            mask=k_valid[:, None] & d_valid[None, :],
            other=0.0,
        )
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    # Unnormalized partial; an empty split stores (0, -inf, 0).
    part = (pid_b * tl.num_programs(1) + pid_kh) * num_splits + pid_s
    off_qh = tl.arange(0, BLOCK_QH)
    tl.store(
        mid_acc_ptr
        + part * (BLOCK_QH * BLOCK_D)
        + off_qh[:, None] * BLOCK_D
        + off_d[None, :],
        acc,
    )
    tl.store(mid_ml_ptr + part * (2 * BLOCK_QH) + off_qh, m_i)
    tl.store(mid_ml_ptr + part * (2 * BLOCK_QH) + BLOCK_QH + off_qh, l_i)


@triton.heuristics(
    {
        "BLOCK_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_QH": lambda args: (
            args["BLOCK_Q"] * triton.next_power_of_2(args["gqa_group_size"])
        ),
    }
)
@triton.jit(do_not_specialize_on_alignment=["cu_seqlens_q"])
def _inkling_rel_attention_split_combine(
    mid_acc_ptr,
    mid_ml_ptr,
    out_ptr,  # [total_q, Hq, D]
    cu_seqlens_q,
    gqa_group_size,
    head_dim,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    num_splits: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_QH: tl.constexpr,
):
    pid_kh = tl.program_id(0)
    pid_b = tl.program_id(1)

    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    if q_len <= 0:
        return

    off_qh = tl.arange(0, BLOCK_QH)
    off_d = tl.arange(0, BLOCK_D)
    base = (pid_b * tl.num_programs(0) + pid_kh) * num_splits

    m = tl.full((BLOCK_QH,), float("-inf"), dtype=tl.float32)
    for s in range(num_splits):
        m = tl.maximum(m, tl.load(mid_ml_ptr + (base + s) * (2 * BLOCK_QH) + off_qh))
    # Rows with no visible key at all (masked GQA lanes, rows past q_len) keep
    # m = -inf; give them a finite reference so nothing below is -inf - -inf.
    m_ref = tl.where(m == float("-inf"), 0.0, m)

    l_tot = tl.zeros((BLOCK_QH,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_QH, BLOCK_D), dtype=tl.float32)
    for s in range(num_splits):
        ml = mid_ml_ptr + (base + s) * (2 * BLOCK_QH)
        w = tl.exp2(tl.load(ml + off_qh) - m_ref)
        l_tot += w * tl.load(ml + BLOCK_QH + off_qh)
        acc += w[:, None] * tl.load(
            mid_acc_ptr
            + (base + s) * (BLOCK_QH * BLOCK_D)
            + off_qh[:, None] * BLOCK_D
            + off_d[None, :]
        )
    acc /= tl.where(l_tot > 0.0, l_tot, 1.0)[:, None]

    acc = tl.reshape(acc, (BLOCK_Q, BLOCK_H, BLOCK_D))
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr + q_start * stride_o_t + pid_kh * gqa_group_size * stride_o_h,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_o_t, stride_o_h, stride_o_d),
        offsets=(0, 0, 0),
        block_shape=(BLOCK_Q, BLOCK_H, BLOCK_D),
        order=(2, 1, 0),
    )
    tl.store(
        out_block_ptr,
        acc.to(out_ptr.dtype.element_ty),
        boundary_check=(0, 1, 2),
    )


@torch.no_grad()
def inkling_fa4_rel_attention(
'''

OLD_DISPATCH = '''    block_q = 1 if max_seqlen_q == 1 else 4
    grid = (triton.cdiv(max_seqlen_q, block_q), num_kv_heads, batch)
'''

NEW_DISPATCH = '''    # Global-layer MTP verify: split the KV range across programs. Zenith
    # patch inkling_verify_split_kv.py.
    verify_splits = verify_split_count(
        batch=batch, num_kv_heads=num_kv_heads, max_kv_len=max_kv_len
    )
    if use_verify_split_kv(
        max_query_len=max_seqlen_q,
        window_left=window_size[0],
        num_splits=verify_splits,
    ):
        block_h = triton.next_power_of_2(gqa_group_size)
        block_d = triton.next_power_of_2(q.shape[2])
        part_shape = (batch, num_kv_heads, verify_splits)
        mid_acc = torch.empty(
            (*part_shape, 4 * block_h, block_d), dtype=torch.float32, device=q.device
        )
        mid_ml = torch.empty(
            (*part_shape, 2, 4 * block_h), dtype=torch.float32, device=q.device
        )
        _inkling_rel_attention_split_kernel[(verify_splits, num_kv_heads, batch)](
            q,
            key_cache,
            value_cache,
            rel_logits,
            mid_acc,
            mid_ml,
            block_table,
            cache_seqlens,
            cu_seqlens_q,
            gqa_group_size,
            q.shape[2],
            softmax_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            value_cache.stride(3),
            rel_logits.stride(0),
            rel_logits.stride(1),
            rel_logits.stride(2),
            block_table.stride(0),
            page_size=key_cache.shape[1],
            rel_extent=rel_extent,
            window_left=window_size[0],
            num_splits=verify_splits,
            BLOCK_Q=4,
            BLOCK_K=64,
            num_warps=4,
            num_stages=1,
        )
        _inkling_rel_attention_split_combine[(num_kv_heads, batch)](
            mid_acc,
            mid_ml,
            out,
            cu_seqlens_q,
            gqa_group_size,
            q.shape[2],
            out.stride(0),
            out.stride(1),
            out.stride(2),
            num_splits=verify_splits,
            BLOCK_Q=4,
            num_warps=4,
        )
        return out
    block_q = 1 if max_seqlen_q == 1 else 4
    grid = (triton.cdiv(max_seqlen_q, block_q), num_kv_heads, batch)
'''

SITES = [
    ("split kernels", OLD_KERNEL_ANCHOR, NEW_KERNEL),
    ("verify dispatch", OLD_DISPATCH, NEW_DISPATCH),
]


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

    patched = src
    problems = []
    for name, old, new in SITES:
        n = patched.count(old)
        if n != 1:
            problems.append(f"{name} (matched {n}x, expected 1)")
            continue
        patched = patched.replace(old, new, 1)

    if problems:
        print(f"ERROR: patch target not found in {path}:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print("vLLM changed inkling_fa4_rel_attention; check whether qlen > 1 "
              "global layers still use the single-pass kernel, then re-derive "
              "or retire this patch", file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {path}")
        return 0

    with open(path, "w") as fh:
        fh.write(patched)
    print(f"patched: {path}  (global-layer MTP verify split across KV)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
