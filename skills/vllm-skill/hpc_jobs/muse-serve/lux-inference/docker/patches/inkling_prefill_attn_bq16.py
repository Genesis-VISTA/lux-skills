#!/usr/bin/env python3
"""
Run Inkling's prefill attention (steps with max_query_len > 4) with 16-query
programs instead of 4-query ones, with the same rounding as the served kernel:
the output is bit-identical.

THE PROBLEM

Every step carrying a prefill chunk runs _inkling_rel_attention_kernel (or,
for mixed steps, its flat-grid copy from inkling_mixed_flat_grid.py) with
BLOCK_Q 4. With 4 query heads per KV head a program is a 16-row tile: each
64-key tile is loaded, page-translated, masked and biased for 4 queries only,
and a global layer's 8k-query prompt walks ~33M key positions per KV head
with the full mask + bias work on every tile. MI250X, real captured inputs of
Inkling-Small TP8 rank 0 (4 q heads, 1 KV head, D 128, page 32): a global
layer costs 3.0 ms for one 7.6k prompt, 5.8 ms for two; a local layer 0.48 /
0.92 ms.

THE FIX

_inkling_rel_attention_prefill_kernel, used for every step with
max_query_len >= INKLING_PREFILL_ATTN_BQ16_MIN_Q (default 2048) on gfx90a
(INKLING_PREFILL_ATTN_BQ16, default 2, see below; 0 restores the BLOCK_Q 4
launch):

  * BLOCK_Q 16 (64 query rows x 64 keys per tile), matrix_instr_nonkdim 16,
    w4, st1;
  * a flat grid of the (request, 16-query block) pairs that exist, as in
    inkling_mixed_flat_grid.py (unit size cdiv(q_len, 16)), walked from the
    last unit backwards so the longest causal walks start first;
  * on global layers the key tiles every query of the block sees past
    rel_extent (bias 0, causal-valid, in range) skip the mask and bias work.

Bit-identity. Each output row reduces the same keys in the same 64-key tiles
in the same order as the BLOCK_Q 4 kernel; the dot products, max and exp2 of
the served kernel round identically per element at either tile height. What
differs is the order of the 64 additions in the row sum of p: with BLOCK_Q 4
on gfx90a the compiler's reduction is, per row, per column c in 0..15,
((x[c] + x[c+16]) + x[c+32]) + x[c+48], then the 16 partial sums pairwise
over lane bits c1, c0, then c3, c2 (found by searching all 4188 candidate
trees against the kernel's own sums, job f61qC3CCJtab). _inkling_rel_rowsum
spells that tree out with tl.split / size-2 sums, which the compiler cannot
reassociate. On the unmasked far tiles an opaque register copy of the scores
(v_mov_b32 via inline asm) stops the compiler contracting
scores * c - m into one fma, which the served kernel never does because its
scores pass through tl.where first.

Mixed steps: decode rows (q_len 2 with MTP 1) take one 16-query program
each, 14 of the 16 rows empty, with INKLING_PREFILL_ATTN_BQ16=1. The default,
2, keeps requests with q_len <= 4 on inkling_mixed_flat_grid.py's BLOCK_Q 4
kernel (a second launch) and runs only the longer ones here. A step whose
longest chunk is short (the 1024-token tail of an 8k prompt) has too few
16-query programs and is slower here than at BLOCK_Q 4, so steps below
INKLING_PREFILL_ATTN_BQ16_MIN_Q keep the BLOCK_Q 4 launch.

EVIDENCE

Tioga, 1 GCD, real inputs captured from the served model (job f61q6UtgWdXD,
layers 4 local / 5 global, two fresh prompts of 7618 and 7615 tokens),
served kernel interleaved with the candidate, us per call (job f61qGudbyD2w;
ndiff 0, max|d| 0 against the served output on every case):

    layer   prompts      BLOCK_Q 4    this
    global  1 x 7.6k       3033       1230
    global  2 x 7.6k       5804       2281
    local   1 x 7.6k        479        370
    local   2 x 7.6k        923        641

Mixed steps (job f61qTD3yqT1R, same captures plus 512 MTP-1 decode rows of
q_len 2; default = mode 2 with MIN_Q 2048; ndiff 0, max|d| 0 against the
served output in every case and mode), us per call:

    layer   step                          served   default
    local   2 x 7.6k + 512 decode           1133       921
    local   7k chunk + 512 decode            626       511
    local   1k tail chunk + 512 decode       241       241  (BLOCK_Q 4)
    global  2 x 7.6k + 512 decode           8321      4833
    global  7k chunk + 512 decode           4936      3532
    global  1k tail chunk + 512 decode      3033      3031  (BLOCK_Q 4)

Served e2e (2 nodes, TP8 x PP2, MTP 1, 16k budget, envelope score, quality
gate passed on all): 1436.6 / 1383.5 / 1408.0 / 1458.9 (mean 1421.8) against
1418.2 / 1400.6 / 1366.7 (mean 1395.2) without this patch.

Pure decode / MTP verify (max_query_len <= 4, the CUDA-graph-captured
steps), the verify split and the split-KV decode path are left alone.
"""
import argparse
import importlib.util
import os
import sys

TARGET = "vllm/models/inkling/amd/ops/fa4_rel_attention.py"
MARKER = "Zenith patch inkling_prefill_attn_bq16.py"

# Both anchors are inkling_mixed_flat_grid.py's own text (this patch sorts
# after it and needs its flat kernel for INKLING_PREFILL_ATTN_BQ16=2). Neither
# overlaps the anchors of inkling_single_pass_kv_bounds.py or
# inkling_verify_split_kv.py, and nothing added here matches them.
OLD_KERNEL_ANCHOR = '''# Mixed prefill + decode steps: _inkling_rel_attention_kernel on a flat grid
# of the (request, query block) pairs that exist. Zenith patch
# inkling_mixed_flat_grid.py.
@triton.jit
def _inkling_rel_attention_flat_units(
'''

NEW_KERNEL = '''def prefill_attn_bq16_mode(max_query_len: int) -> int:
    """0: BLOCK_Q 4 kernels. 1: _inkling_rel_attention_prefill_kernel for
    every request of the step. 2: only for requests with q_len > 4; the rest
    stay on the BLOCK_Q 4 flat kernel. Steps whose longest query is shorter
    than INKLING_PREFILL_ATTN_BQ16_MIN_Q stay on BLOCK_Q 4: too few 16-query
    programs to fill the GPU. Zenith patch inkling_prefill_attn_bq16.py.
    """
    if max_query_len <= 4:
        return 0
    if max_query_len < int(os.getenv("INKLING_PREFILL_ATTN_BQ16_MIN_Q", "2048")):
        return 0
    # The row-sum order below is the BLOCK_Q 4 kernel's on gfx90a only.
    from vllm.platforms.rocm import on_gfx90a

    if not on_gfx90a():
        return 0
    return int(os.getenv("INKLING_PREFILL_ATTN_BQ16", "2"))


@triton.jit
def _inkling_rel_attention_prefill_units(
    cu_seqlens_q,
    cu_units,  # [batch + 1] int32 out: prefix sums of the units per request
    batch,
    BLOCK_Q: tl.constexpr,
    SELECT: tl.constexpr,  # 0 every request, 1 q_len > 4 only, 2 q_len <= 4 only
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    m = i < batch
    q_len = tl.load(cu_seqlens_q + i + 1, mask=m, other=0) - tl.load(
        cu_seqlens_q + i, mask=m, other=0
    )
    if SELECT == 1:
        m &= q_len > 4
    if SELECT == 2:
        m &= q_len <= 4
    units = tl.where(m, (q_len + BLOCK_Q - 1) // BLOCK_Q, 0)
    tl.store(cu_units + 1 + i, tl.cumsum(units, 0), mask=i < batch)
    tl.store(cu_units + i, units * 0, mask=i == 0)


@triton.jit
def _inkling_rel_rowsum(p, R: tl.constexpr):
    # Row sums of p [R, 64] in the BLOCK_Q 4 kernel's order on gfx90a. Column
    # bits b1 b0 c3 c2 c1 c0: per column c, ((x[b=0] + x[b=1]) + x[b=2]) +
    # x[b=3], then lanes c1, c0, then c3, c2. Each tl.sum is one add.
    x = tl.reshape(p, (R, 2, 2, 2, 2, 2, 2))
    x = tl.permute(x, (0, 3, 4, 5, 6, 1, 2))  # R c3 c2 c1 c0 b1 b0
    xb0, xb1 = tl.split(x)  # b0 = 0 / 1
    x00, x10 = tl.split(xb0)  # (b1, b0) = (0, 0) / (1, 0)
    x01, x11 = tl.split(xb1)  # (0, 1) / (1, 1)
    x = ((x00 + x01) + x10) + x11  # R c3 c2 c1 c0
    x = tl.sum(x, axis=3)  # c1
    x = tl.sum(x, axis=3)  # c0
    x = tl.sum(x, axis=1)  # c3
    x = tl.sum(x, axis=1)  # c2
    return x


@triton.heuristics(
    {
        "BLOCK_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_QH": lambda args: (
            args["BLOCK_Q"] * triton.next_power_of_2(args["gqa_group_size"])
        ),
    }
)
@triton.jit(
    do_not_specialize_on_alignment=["cache_seqlens", "cu_seqlens_q", "cu_units"]
)
def _inkling_rel_attention_prefill_kernel(
    q_ptr,  # [total_q, Hq, D]
    k_ptr,  # [blocks, page, Hkv, D]
    v_ptr,  # [blocks, page, Hkv, D]
    rel_ptr,  # [total_q, Hq, rel_extent]
    out_ptr,  # [total_q, Hq, D]
    block_table_ptr,  # [batch, max_pages]
    cache_seqlens,
    cu_seqlens_q,
    cu_units,  # [batch + 1] from _inkling_rel_attention_prefill_units
    batch,
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
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_bt_b,
    page_size: tl.constexpr,
    rel_extent: tl.constexpr,
    window_left: tl.constexpr,
    SEARCH: tl.constexpr,  # binary-search steps, >= log2(batch)
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_QH: tl.constexpr,
):
    # Program (unit, KV head), units taken last first: the request b with
    # cu_units[b] <= unit < cu_units[b + 1] and its query block
    # unit - cu_units[b]. Output rounding is _inkling_rel_attention_kernel's
    # at BLOCK_Q 4 (see _inkling_rel_rowsum).
    num_units = tl.load(cu_units + batch)
    pid_f = tl.program_id(0)
    pid_kh = tl.program_id(1)
    if pid_f >= num_units:
        return
    unit = num_units - 1 - pid_f
    lo = pid_f * 0
    hi = lo + batch
    for _ in tl.static_range(SEARCH):
        mid = (lo + hi) // 2
        go = tl.load(cu_units + mid) <= unit
        lo = tl.where(go, mid, lo)
        hi = tl.where(go, hi, mid)
    pid_b = lo

    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block = (unit - tl.load(cu_units + pid_b)) * BLOCK_Q
    kv_len = tl.load(cache_seqlens + pid_b)
    prefix_len = kv_len - q_len
    q_head_start = pid_kh * gqa_group_size
    bt_row = block_table_ptr + pid_b * stride_bt_b
    off_d = tl.arange(0, BLOCK_D)
    d_valid = off_d < head_dim
    off_h = tl.arange(0, BLOCK_H)
    log2e: tl.constexpr = 1.4426950408889634

    q = tl.load(
        tl.make_block_ptr(
            base=q_ptr + q_start * stride_q_t + q_head_start * stride_q_h,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_q_t, stride_q_h, stride_q_d),
            offsets=(q_block, 0, 0),
            block_shape=(BLOCK_Q, BLOCK_H, BLOCK_D),
            order=(2, 1, 0),
        ),
        boundary_check=(0, 1, 2),
        padding_option="zero",
    )
    q = tl.reshape(q, (BLOCK_QH, BLOCK_D))
    q_rows = q_block + tl.arange(0, BLOCK_Q)
    q_abs = prefix_len + q_rows
    q_valid = q_rows < q_len
    m_i = tl.full((BLOCK_QH,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_QH,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_QH, BLOCK_D), dtype=tl.float32)

    # Tiles some query of the block can see: [k_lo, k_hi). On global layers
    # the tiles below k_mid are past rel_extent and causal-valid for every
    # query of the block, so they need neither mask nor bias.
    k_lo = 0
    if window_left >= 0:
        k_lo = tl.maximum(prefix_len + q_block - window_left, 0) // BLOCK_K * BLOCK_K
    k_hi = tl.minimum(kv_len, prefix_len + tl.minimum(q_block + BLOCK_Q, q_len))
    k_mid = k_lo
    if window_left < 0:
        k_mid = tl.minimum(
            tl.maximum(prefix_len + q_block - rel_extent + 1, 0) // BLOCK_K * BLOCK_K,
            k_hi,
        )
    for k_start in range(k_lo, k_mid, BLOCK_K):
        k_pos = k_start + tl.arange(0, BLOCK_K)
        page = tl.load(bt_row + k_pos // page_size).to(tl.int64)
        page_off = k_pos % page_size
        k = tl.load(
            k_ptr
            + page[None, :] * stride_k_b
            + page_off[None, :] * stride_k_p
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d
        )
        v = tl.load(
            v_ptr
            + page[:, None] * stride_v_b
            + page_off[:, None] * stride_v_p
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d
        )
        scores = tl.dot(q, k) * (softmax_scale * log2e)
        # Opaque copy: keeps scores * c - m from contracting into one fma.
        scores = tl.inline_asm_elementwise(
            "v_mov_b32 $0, $1", "=v,v", [scores],
            dtype=tl.float32, is_pure=True, pack=1,
        )
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp2(m_i - m_ij)
        p = tl.exp2(scores - m_ij[:, None])
        l_i = l_i * alpha + _inkling_rel_rowsum(p, BLOCK_QH)
        acc *= alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    for k_start in range(k_mid, k_hi, BLOCK_K):
        k_pos = k_start + tl.arange(0, BLOCK_K)
        k_valid = k_pos < kv_len
        # Masked lanes still need an in-range page-table load.
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
        v = tl.load(
            v_ptr
            + page[:, None] * stride_v_b
            + page_off[:, None] * stride_v_p
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d,
            mask=k_valid[:, None] & d_valid[None, :],
            other=0.0,
        )
        scores = tl.dot(q, k) * (softmax_scale * log2e)
        dist = q_abs[:, None] - k_pos[None, :]
        score_valid = q_valid[:, None] & k_valid[None, :] & (dist >= 0)
        if window_left >= 0:
            score_valid &= dist <= window_left
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
        p = tl.where(score_valid, tl.exp2(scores - m_ij[:, None]), 0.0)
        l_i = l_i * alpha + _inkling_rel_rowsum(p, BLOCK_QH)
        acc *= alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    acc /= l_i[:, None]
    acc = tl.reshape(acc, (BLOCK_Q, BLOCK_H, BLOCK_D))
    tl.store(
        tl.make_block_ptr(
            base=out_ptr + q_start * stride_o_t + q_head_start * stride_o_h,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_o_t, stride_o_h, stride_o_d),
            offsets=(q_block, 0, 0),
            block_shape=(BLOCK_Q, BLOCK_H, BLOCK_D),
            order=(2, 1, 0),
        ),
        acc.to(out_ptr.dtype.element_ty),
        boundary_check=(0, 1, 2),
    )


# Mixed prefill + decode steps: _inkling_rel_attention_kernel on a flat grid
# of the (request, query block) pairs that exist. Zenith patch
# inkling_mixed_flat_grid.py.
@triton.jit
def _inkling_rel_attention_flat_units(
'''

# inkling_mixed_flat_grid.py's branch; the prefill branch goes in front of it.
OLD_DISPATCH = '''    if max_seqlen_q > 4 and batch > 1:
        # Mixed prefill + decode step: one program per (request, query block)
'''

NEW_DISPATCH = '''    bq16 = prefill_attn_bq16_mode(max_seqlen_q)
    if bq16:
        # Prefill (max_query_len > 4): 16-query programs on a flat grid,
        # bit-identical to the BLOCK_Q 4 launch below. With bq16 == 2 the
        # requests with q_len <= 4 (decode rows of a mixed step) stay on the
        # BLOCK_Q 4 flat kernel. Zenith patch inkling_prefill_attn_bq16.py.
        args = (
            q,
            key_cache,
            value_cache,
            rel_logits,
            out,
            block_table,
            cache_seqlens,
            cu_seqlens_q,
        )
        strides = (
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
            out.stride(0),
            out.stride(1),
            out.stride(2),
            block_table.stride(0),
        )
        consts = dict(
            page_size=key_cache.shape[1],
            rel_extent=rel_extent,
            window_left=window_size[0],
            SEARCH=max(11, batch.bit_length()),
        )
        units_block = triton.next_power_of_2(batch + 1)
        cu_units = torch.empty(batch + 1, dtype=torch.int32, device=q.device)
        _inkling_rel_attention_prefill_units[(1,)](
            cu_seqlens_q,
            cu_units,
            batch,
            BLOCK_Q=16,
            SELECT=1 if bq16 == 2 else 0,
            BLOCK=units_block,
            num_warps=4,
        )
        # sum_b cdiv(q_len_b, 16) <= (total_q + 15 * batch) // 16
        _inkling_rel_attention_prefill_kernel[
            ((q.shape[0] + 15 * batch) // 16, num_kv_heads)
        ](
            *args,
            cu_units,
            batch,
            gqa_group_size,
            q.shape[2],
            softmax_scale,
            *strides,
            **consts,
            BLOCK_Q=16,
            BLOCK_K=64,
            num_warps=4,
            num_stages=1,
            matrix_instr_nonkdim=16,
        )
        if bq16 == 2 and batch > 1:
            cu_units_dec = torch.empty_like(cu_units)
            _inkling_rel_attention_prefill_units[(1,)](
                cu_seqlens_q,
                cu_units_dec,
                batch,
                BLOCK_Q=4,
                SELECT=2,
                BLOCK=units_block,
                num_warps=4,
            )
            _inkling_rel_attention_flat_kernel[(batch, num_kv_heads)](
                *args,
                cu_units_dec,
                batch,
                gqa_group_size,
                q.shape[2],
                softmax_scale,
                *strides,
                **consts,
                BLOCK_Q=4,
                BLOCK_K=64,
                num_warps=4,
                num_stages=1,
            )
        return out
    if max_seqlen_q > 4 and batch > 1:
        # Mixed prefill + decode step: one program per (request, query block)
'''

SITES = [
    ("prefill kernel", OLD_KERNEL_ANCHOR, NEW_KERNEL),
    ("prefill dispatch", OLD_DISPATCH, NEW_DISPATCH),
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
    ap.add_argument("--revert", action="store_true",
                    help="restore the BLOCK_Q 4 prefill launch")
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
        reverted = src
        for _, old, new in SITES:
            if reverted.count(new) != 1:
                print(f"ERROR: {path} carries a different version of this "
                      "patch; restore fa4_rel_attention.py from the image",
                      file=sys.stderr)
                return 1
            reverted = reverted.replace(new, old, 1)
        if args.check:
            print(f"would revert: {path}")
            return 0
        with open(path, "w") as fh:
            fh.write(reverted)
        print(f"reverted: {path}")
        return 0

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
        print("this patch needs inkling_mixed_flat_grid.py applied first; if it "
              "is, vLLM changed inkling_fa4_rel_attention: check whether steps "
              "with max_query_len > 4 still launch the BLOCK_Q 4 kernels, then "
              "re-derive or retire this patch", file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {path}")
        return 0

    with open(path, "w") as fh:
        fh.write(patched)
    print(f"patched: {path}  (prefill attention at BLOCK_Q 16, bit-identical; "
          "INKLING_PREFILL_ATTN_BQ16=0 to disable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
