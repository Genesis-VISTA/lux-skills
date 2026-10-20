# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm paged attention with Inkling's query-dependent relative bias.

The NVIDIA implementation uses the score-mod hook in tml-fa4.  ROCm Flash
Attention and AITER do not expose an equivalent hook, so this module implements
the same operation directly in Triton.  Query heads belonging to one KV head
are processed together and KV pages are gathered through vLLM's block table.
"""

from __future__ import annotations

import os
from typing import cast

import torch

from vllm.models.inkling.amd.ops.rel_attention_decode import (
    inkling_rel_attention_split_kv_decode,
    use_split_kv_decode,
)
from vllm.platforms.rocm import on_gfx950
from vllm.triton_utils import tl, triton


def bucket_max_seqlen_q(max_seqlen_q: int) -> int:
    """Round the scheduling bound up to a power of two."""
    return 1 << max(0, max_seqlen_q - 1).bit_length()


def use_gfx950_gluon_decode(
    *, max_query_len: int, page_size: int, head_dim: int
) -> bool:
    """Use the vendored TokenSpeed CDNA4 decode kernel where it is supported."""
    return (
        os.getenv("INKLING_GFX950_GLUON", "1") == "1"
        and on_gfx950()
        and max_query_len == 1
        and page_size in (64, 128, 256)
        and head_dim in (64, 128)
    )


def use_gfx950_gluon_extend(
    *,
    max_query_len: int,
    max_kv_len: int,
    page_size: int,
    head_dim: int,
    window_left: int,
) -> bool:
    """Use Gluon only for the long full-attention extend regime it wins."""
    return (
        os.getenv("INKLING_GFX950_GLUON", "1") == "1"
        and on_gfx950()
        and max_query_len > 1
        and max_kv_len >= 8192
        and window_left < 0
        and page_size in (64, 128, 256)
        and head_dim in (64, 128)
    )


def inkling_fa4_num_splits(
    *,
    is_local: bool,
    batch_size: int,
    max_query_len: int,
    num_heads: int,
    num_kv_heads: int,
    max_kv_len: int,
) -> int:
    """Keep the NVIDIA-facing split heuristic as API-compatible metadata.

    The ROCm Triton implementation performs online softmax in one program and
    does not consume the result.  Keeping this function unchanged avoids
    platform-specific scheduling branches in :mod:`inkling.amd.attention`.
    """
    if is_local:
        return 1

    q_rows = max_query_len * (num_heads // num_kv_heads)
    q_tiles = (q_rows + 255) // 256
    base_ctas = batch_size * num_kv_heads * q_tiles
    target_ctas = (
        256 if q_tiles == 1 and batch_size == 1 else (128 if q_tiles == 1 else 64)
    )
    max_splits = 128
    if q_tiles == 1 and batch_size == 1:
        if num_kv_heads == 8:
            max_splits = 16
        elif num_kv_heads == 4 or max_kv_len <= 8192:
            max_splits = 32
        elif max_kv_len <= 65536:
            max_splits = 64
    return max(
        1,
        min(target_ctas // base_ctas, max_splits, (max_kv_len + 127) // 128),
    )


def prefill_attn_bq16_mode(max_query_len: int) -> int:
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
    cu_seqlens_q,
    cu_units,  # [batch + 1] int32 out: prefix sums of cdiv(q_len, BLOCK_Q)
    batch,
    BLOCK_Q: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    m = i < batch
    q_len = tl.load(cu_seqlens_q + i + 1, mask=m, other=0) - tl.load(
        cu_seqlens_q + i, mask=m, other=0
    )
    units = tl.where(m, (q_len + BLOCK_Q - 1) // BLOCK_Q, 0)
    tl.store(cu_units + 1 + i, tl.cumsum(units, 0), mask=m)
    tl.store(cu_units + i, units * 0, mask=i == 0)


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
def _inkling_rel_attention_flat_kernel(
    q_ptr,  # [total_q, Hq, D]
    k_ptr,  # [blocks, page, Hkv, D]
    v_ptr,  # [blocks, page, Hkv, D]
    rel_ptr,  # [total_q, Hq, rel_extent]
    out_ptr,  # [total_q, Hq, D]
    block_table_ptr,  # [batch, max_pages]
    cache_seqlens,
    cu_seqlens_q,
    cu_units,  # [batch + 1] from _inkling_rel_attention_flat_units
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
    # Program (unit, KV head): the request b with
    # cu_units[b] <= unit < cu_units[b + 1] (empty requests own no unit) and
    # its query block unit - cu_units[b]. Below the row lookup this is
    # _inkling_rel_attention_kernel's body unchanged.
    pid_f = tl.program_id(0)
    pid_kh = tl.program_id(1)
    if pid_f >= tl.load(cu_units + batch):
        return
    lo = pid_f * 0
    hi = lo + batch
    for _ in tl.static_range(SEARCH):
        mid = (lo + hi) // 2
        go = tl.load(cu_units + mid) <= pid_f
        lo = tl.where(go, mid, lo)
        hi = tl.where(go, hi, mid)
    pid_b = lo

    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block = (pid_f - tl.load(cu_units + pid_b)) * BLOCK_Q

    kv_len = tl.load(cache_seqlens + pid_b)
    prefix_len = kv_len - q_len
    q_head_start = pid_kh * gqa_group_size
    bt_row = block_table_ptr + pid_b * stride_bt_b

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_start * stride_q_t + q_head_start * stride_q_h,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_q_t, stride_q_h, stride_q_d),
        offsets=(q_block, 0, 0),
        block_shape=(BLOCK_Q, BLOCK_H, BLOCK_D),
        order=(2, 1, 0),
    )
    q = tl.load(q_block_ptr, boundary_check=(0, 1, 2), padding_option="zero")
    q = tl.reshape(q, (BLOCK_QH, BLOCK_D))

    q_rows = q_block + tl.arange(0, BLOCK_Q)
    q_abs = prefix_len + q_rows
    q_valid = q_rows < q_len
    off_d = tl.arange(0, BLOCK_D)
    d_valid = off_d < head_dim

    m_i = tl.full((BLOCK_QH,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_QH,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_QH, BLOCK_D), dtype=tl.float32)
    log2e: tl.constexpr = 1.4426950408889634

    # Only walk tiles some query of this block can see: from the window start
    # of its first query (local layers) to the causal limit of its last one.
    # Every skipped tile was fully masked.
    k_lo = 0
    if window_left >= 0:
        k_lo = tl.maximum(prefix_len + q_block - window_left, 0) // BLOCK_K * BLOCK_K
    k_hi = tl.minimum(kv_len, prefix_len + tl.minimum(q_block + BLOCK_Q, q_len))
    for k_start in range(k_lo, k_hi, BLOCK_K):
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
        scores = tl.dot(q, k) * (softmax_scale * log2e)

        dist = q_abs[:, None] - k_pos[None, :]
        score_valid = q_valid[:, None] & k_valid[None, :] & (dist >= 0)
        if window_left >= 0:
            score_valid &= dist <= window_left

        # rel_logits is query- and head-dependent. Expand [Q, K] distance
        # indices across the GQA heads, then flatten to match QK's row order.
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
        # A sliding-window query can have entire early KV tiles masked. Avoid
        # the -inf - -inf NaN in online softmax until its first live tile.
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

    acc /= l_i[:, None]
    acc = tl.reshape(acc, (BLOCK_Q, BLOCK_H, BLOCK_D))
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr + q_start * stride_o_t + q_head_start * stride_o_h,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_o_t, stride_o_h, stride_o_d),
        offsets=(q_block, 0, 0),
        block_shape=(BLOCK_Q, BLOCK_H, BLOCK_D),
        order=(2, 1, 0),
    )
    tl.store(
        out_block_ptr,
        acc.to(out_ptr.dtype.element_ty),
        boundary_check=(0, 1, 2),
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
def _inkling_rel_attention_kernel(
    q_ptr,  # [total_q, Hq, D]
    k_ptr,  # [blocks, page, Hkv, D]
    v_ptr,  # [blocks, page, Hkv, D]
    rel_ptr,  # [total_q, Hq, rel_extent]
    out_ptr,  # [total_q, Hq, D]
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
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_bt_b,
    page_size: tl.constexpr,
    rel_extent: tl.constexpr,
    window_left: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_QH: tl.constexpr,
    RELSKIP: tl.constexpr = False,  # patch rocm_inkling_rel_attn_dec.py
):
    # A program owns BLOCK_Q query positions and every query head sharing one
    # KV head. This makes the QK/PV operations MFMA-friendly on CDNA.
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)

    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block = pid_q * BLOCK_Q
    if q_block >= q_len:
        return

    kv_len = tl.load(cache_seqlens + pid_b)
    prefix_len = kv_len - q_len
    q_head_start = pid_kh * gqa_group_size
    bt_row = block_table_ptr + pid_b * stride_bt_b

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_start * stride_q_t + q_head_start * stride_q_h,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_q_t, stride_q_h, stride_q_d),
        offsets=(q_block, 0, 0),
        block_shape=(BLOCK_Q, BLOCK_H, BLOCK_D),
        order=(2, 1, 0),
    )
    q = tl.load(q_block_ptr, boundary_check=(0, 1, 2), padding_option="zero")
    q = tl.reshape(q, (BLOCK_QH, BLOCK_D))

    q_rows = q_block + tl.arange(0, BLOCK_Q)
    q_abs = prefix_len + q_rows
    q_valid = q_rows < q_len
    off_d = tl.arange(0, BLOCK_D)
    d_valid = off_d < head_dim

    m_i = tl.full((BLOCK_QH,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_QH,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_QH, BLOCK_D), dtype=tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    # first key that is within rel_extent of the block's first query row
    rel_lo = prefix_len + q_block - rel_extent + 1

    # Only walk tiles some query of this block can see: from the window start
    # of its first query (local layers) to the causal limit of its last one.
    # Every skipped tile was fully masked.
    # Zenith patch inkling_single_pass_kv_bounds.py.
    k_lo = 0
    if window_left >= 0:
        k_lo = tl.maximum(prefix_len + q_block - window_left, 0) // BLOCK_K * BLOCK_K
    k_hi = tl.minimum(kv_len, prefix_len + tl.minimum(q_block + BLOCK_Q, q_len))
    for k_start in range(k_lo, k_hi, BLOCK_K):
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
        qk = tl.dot(q, k)

        dist = q_abs[:, None] - k_pos[None, :]
        score_valid = q_valid[:, None] & k_valid[None, :] & (dist >= 0)
        if window_left >= 0:
            score_valid &= dist <= window_left

        # rel_logits is query- and head-dependent. Expand [Q, K] distance
        # indices across the GQA heads, then flatten to match QK's row order.
        off_h = tl.arange(0, BLOCK_H)
        # (patch rocm_inkling_rel_attn_dec.py) RELSKIP: a tile ending before
        # rel_extent of this program's first query row has bias 0 everywhere.
        if RELSKIP:
            if k_start + BLOCK_K > rel_lo:
                scores = _inkling_rel_bias_scores(
                    qk, dist, q_valid, q_rows, off_h, rel_ptr, q_start,
                    q_head_start, gqa_group_size, stride_r_t, stride_r_h,
                    stride_r_e, softmax_scale, rel_extent, BLOCK_QH, BLOCK_K,
                )
            else:
                # + 0.0 keeps the served rounding: without it the scale is
                # contracted into the exp2 argument (fma(qk, c, -m)).
                scores = qk * (softmax_scale * log2e) + 0.0
        else:
            scores = _inkling_rel_bias_scores(
                qk, dist, q_valid, q_rows, off_h, rel_ptr, q_start,
                q_head_start, gqa_group_size, stride_r_t, stride_r_h,
                stride_r_e, softmax_scale, rel_extent, BLOCK_QH, BLOCK_K,
            )
        score_valid = tl.reshape(
            score_valid[:, None, :] & (off_h[None, :, None] < gqa_group_size),
            (BLOCK_QH, BLOCK_K),
        )
        scores = tl.where(score_valid, scores, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        has_scores = tl.sum(score_valid.to(tl.int32), axis=1) > 0
        # A sliding-window query can have entire early KV tiles masked. Avoid
        # the -inf - -inf NaN in online softmax until its first live tile.
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

    acc /= l_i[:, None]
    acc = tl.reshape(acc, (BLOCK_Q, BLOCK_H, BLOCK_D))
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr + q_start * stride_o_t + q_head_start * stride_o_h,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_o_t, stride_o_h, stride_o_d),
        offsets=(q_block, 0, 0),
        block_shape=(BLOCK_Q, BLOCK_H, BLOCK_D),
        order=(2, 1, 0),
    )
    tl.store(
        out_block_ptr,
        acc.to(out_ptr.dtype.element_ty),
        boundary_check=(0, 1, 2),
    )


# (patch rocm_inkling_rel_attn_dec.py) INKLING_DEC_ATTN=1: rel-bias tile skip
# in the decode kernels, the INKLING_DEC_ATTN_G global verify tile, and an
# opt-in 2-way KV split for local verify at small batch.
_DEC_ATTN = os.getenv("INKLING_DEC_ATTN", "0") == "1"
_DEC_ATTN_LOCAL_SPLIT_B = int(os.getenv("INKLING_DEC_ATTN_LOCAL_SPLIT_B", "0"))
# global verify tile at q_len <= 2: BLOCK_Q, BLOCK_K, num_warps, HOIST_V, RELSKIP
_DEC_ATTN_G = tuple(
    int(v) for v in os.getenv("INKLING_DEC_ATTN_G", "2,64,2,0,1").split(",")
)


@triton.jit
def _inkling_rel_bias_scores(
    qk, dist, q_valid, q_rows, off_h, rel_ptr, q_start, q_head_start,
    gqa_group_size, stride_r_t, stride_r_h, stride_r_e, softmax_scale,
    rel_extent: tl.constexpr, BLOCK_QH: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # The served score expression (QK scale + relative bias), verbatim.
    log2e: tl.constexpr = 1.4426950408889634
    scores = qk * (softmax_scale * log2e)
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
    return scores


@triton.jit
def _kv_tile_load(
    page,
    k_pos,
    k_last,
    k_head_ptr,
    v_head_ptr,
    d_valid,
    stride_k_b,
    stride_k_p,
    page_size: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    safe_pos = tl.minimum(k_pos, k_last)
    kv_off = page * stride_k_b + (safe_pos % page_size) * stride_k_p
    if EVEN_D:
        k = tl.load(k_head_ptr + kv_off[None, :])
        v = tl.load(v_head_ptr + kv_off[:, None])
    else:
        k = tl.load(k_head_ptr + kv_off[None, :], mask=d_valid[:, None], other=0.0)
        v = tl.load(v_head_ptr + kv_off[:, None], mask=d_valid[None, :], other=0.0)
    return k, v


@triton.jit
def _rel_bias_load(
    k_pos,
    k_hi,
    rel_row_ptr,
    row_abs,
    row_valid,
    stride_r_e,
    rel_extent: tl.constexpr,
    window_left: tl.constexpr,
):
    dist = row_abs[:, None] - k_pos[None, :]
    valid = row_valid[:, None] & (k_pos[None, :] < k_hi) & (dist >= 0)
    if window_left >= 0:
        valid &= dist <= window_left
    if window_left >= 0 and window_left < rel_extent:
        # Every valid distance lies in [0, window_left], inside [0, rel_extent).
        bias = tl.load(rel_row_ptr[:, None] + dist * stride_r_e, mask=valid, other=0.0)
    else:
        bias = tl.load(
            rel_row_ptr[:, None]
            + tl.minimum(tl.maximum(dist, 0), rel_extent - 1) * stride_r_e,
            mask=valid & (dist < rel_extent),
            other=0.0,
        )
    return bias, valid


# MI355X (gfx950) single-pass kernel: the Hyperloom rewrite of
# _inkling_rel_attention_kernel (KV bounds, shared tile loads, EVEN_D / EVEN_H,
# XCD remap, Q_SPLIT for mixed steps), launched by _launch_rel_attention with
# the shapes tuned on MI355X. Other archs run _inkling_rel_attention_kernel and
# the v8 paths above it (flat grid, BLOCK_Q 16 prefill, INKLING_DEC_ATTN).
@triton.heuristics(
    {
        "BLOCK_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_QH": lambda args: (
            args["BLOCK_Q"] * triton.next_power_of_2(args["gqa_group_size"])
        ),
        "EVEN_D": lambda args: args["head_dim"] == triton.next_power_of_2(args["head_dim"]),
        "EVEN_H": lambda args: (
            args["gqa_group_size"] == triton.next_power_of_2(args["gqa_group_size"])
        ),
    }
)
@triton.jit(do_not_specialize_on_alignment=["cache_seqlens", "cu_seqlens_q"])
def _inkling_rel_attention_tuned_kernel(
    q_ptr,  # [total_q, Hq, D]
    k_ptr,  # [blocks, page, Hkv, D]
    v_ptr,  # [blocks, page, Hkv, D]
    rel_ptr,  # [total_q, Hq, rel_extent]
    out_ptr,  # [total_q, Hq, D]
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
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_bt_b,
    page_size: tl.constexpr,
    rel_extent: tl.constexpr,
    window_left: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_QH: tl.constexpr,
    EVEN_D: tl.constexpr,
    EVEN_H: tl.constexpr,
    XCD_REMAP: tl.constexpr,
    Q_SPLIT: tl.constexpr = 0,
    SHORT_Q: tl.constexpr = 16,
):
    # A program owns BLOCK_Q query positions and every query head sharing one
    # KV head. This makes the QK/PV operations MFMA-friendly on CDNA.
    pid_q = tl.program_id(0)
    if XCD_REMAP > 1:
        # Workgroups are dealt round-robin across XCDs; give each XCD a
        # contiguous run of query blocks so their overlapping KV windows hit
        # in that XCD's L2. Host guarantees grid[0] % XCD_REMAP == 0.
        n_q = tl.num_programs(0)
        pid_q = (pid_q % XCD_REMAP) * (n_q // XCD_REMAP) + pid_q // XCD_REMAP
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)

    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block = pid_q * BLOCK_Q
    if q_block >= q_len:
        return
    # Mixed steps launch twice: Q_SPLIT 1 takes the sequences longer than
    # SHORT_Q, Q_SPLIT 2 the short ones (decode / MTP verify).
    if Q_SPLIT == 1:
        if q_len <= SHORT_Q:
            return
    if Q_SPLIT == 2:
        if q_len > SHORT_Q:
            return

    kv_len = tl.load(cache_seqlens + pid_b)
    prefix_len = kv_len - q_len
    q_head_start = pid_kh * gqa_group_size
    bt_row = block_table_ptr + pid_b * stride_bt_b

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_start * stride_q_t + q_head_start * stride_q_h,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_q_t, stride_q_h, stride_q_d),
        offsets=(q_block, 0, 0),
        block_shape=(BLOCK_Q, BLOCK_H, BLOCK_D),
        order=(2, 1, 0),
    )
    q = tl.load(q_block_ptr, boundary_check=(0, 1, 2), padding_option="zero")
    q = tl.reshape(q, (BLOCK_QH, BLOCK_D))

    # Flattened (query row, GQA head) rows in QK's row order: i = r * BLOCK_H + h.
    off_qh = tl.arange(0, BLOCK_QH)
    row_q = q_block + off_qh // BLOCK_H
    row_h = off_qh % BLOCK_H
    row_valid = row_q < q_len
    if not EVEN_H:
        row_valid &= row_h < gqa_group_size
    row_abs = prefix_len + row_q
    rel_row_ptr = (
        rel_ptr + (q_start + row_q) * stride_r_t + (q_head_start + row_h) * stride_r_h
    )
    off_d = tl.arange(0, BLOCK_D)
    d_valid = off_d < head_dim
    offs_k = tl.arange(0, BLOCK_K)
    k_head_ptr = k_ptr + pid_kh * stride_k_h + off_d[:, None] * stride_k_d
    v_head_ptr = v_ptr + pid_kh * stride_v_h + off_d[None, :] * stride_v_d

    m_i = tl.full((BLOCK_QH,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_QH,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_QH, BLOCK_D), dtype=tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = softmax_scale * log2e

    # Only walk keys some query of this block can see: from the window start
    # of its first query (local layers) to the causal limit of its last one.
    # Every key outside [k_lo, k_hi) is masked for every row of this block.
    k_lo = 0
    if window_left >= 0:
        k_lo = tl.maximum(prefix_len + q_block - window_left, 0)
    k_hi = tl.minimum(kv_len, prefix_len + tl.minimum(q_block + BLOCK_Q, q_len))
    # Masked lanes read a clamped in-range key (finite cache data, p = 0 there).
    k_last = tl.maximum(k_hi - 1, 0)

    for k_start in range(k_lo, k_hi, BLOCK_K):
        k_pos = k_start + offs_k
        page = tl.load(bt_row + tl.minimum(k_pos, k_last) // page_size).to(tl.int64)
        k, v = _kv_tile_load(
            page, k_pos, k_last, k_head_ptr, v_head_ptr, d_valid,
            stride_k_b, stride_k_p, page_size, EVEN_D,
        )
        bias, valid = _rel_bias_load(
            k_pos, k_hi, rel_row_ptr, row_abs, row_valid, stride_r_e,
            rel_extent, window_left,
        )

        scores = tl.dot(q, k) * qk_scale + bias.to(tl.float32) * log2e
        scores = tl.where(valid, scores, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        # A sliding-window row can have whole early tiles masked; a finite
        # reference avoids -inf - -inf (alpha = 0 and p = 0 while m is -inf).
        m_safe = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(scores - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    acc /= l_i[:, None]
    acc = tl.reshape(acc, (BLOCK_Q, BLOCK_H, BLOCK_D))
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr + q_start * stride_o_t + q_head_start * stride_o_h,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_o_t, stride_o_h, stride_o_d),
        offsets=(q_block, 0, 0),
        block_shape=(BLOCK_Q, BLOCK_H, BLOCK_D),
        order=(2, 1, 0),
    )
    tl.store(
        out_block_ptr,
        acc.to(out_ptr.dtype.element_ty),
        boundary_check=(0, 1, 2),
    )


@triton.heuristics(
    {
        "BLOCK_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_QH": lambda args: (
            args["BLOCK_Q"] * triton.next_power_of_2(args["gqa_group_size"])
        ),
        "EVEN_D": lambda args: args["head_dim"] == triton.next_power_of_2(args["head_dim"]),
        "EVEN_H": lambda args: (
            args["gqa_group_size"] == triton.next_power_of_2(args["gqa_group_size"])
        ),
    }
)
@triton.jit(do_not_specialize_on_alignment=["cache_seqlens", "cu_seqlens_q"])
def _inkling_rel_attention_window_split_kernel(
    q_ptr,  # [total_q, Hq, D]
    k_ptr,  # [blocks, page, Hkv, D]
    v_ptr,  # [blocks, page, Hkv, D]
    rel_ptr,  # [total_q, Hq, rel_extent]
    mid_acc_ptr,  # [batch, Hkv, splits, BLOCK_QH, BLOCK_D] fp32
    mid_ml_ptr,  # [batch, Hkv, splits, 2, BLOCK_QH] fp32 (log2 row max, row sum)
    out_ptr,  # [total_q, Hq, D] (FUSED_COMBINE only)
    counter_ptr,  # [>= batch * Hkv] int32, zero between launches (FUSED_COMBINE only)
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
    stride_o_t,
    stride_o_h,
    stride_o_d,
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
    EVEN_D: tl.constexpr,
    EVEN_H: tl.constexpr,
    FUSED_COMBINE: tl.constexpr,
    PART_STORE_CM: tl.constexpr,
):
    # _inkling_rel_attention_tuned_kernel for one query block (host: q_len <= BLOCK_Q)
    # with the key range [k_lo, k_hi) cut into num_splits programs. Each writes
    # an unnormalised partial that _inkling_rel_attention_split_combine merges.
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

    off_qh = tl.arange(0, BLOCK_QH)
    row_q = off_qh // BLOCK_H
    row_h = off_qh % BLOCK_H
    row_valid = row_q < q_len
    if not EVEN_H:
        row_valid &= row_h < gqa_group_size
    row_abs = prefix_len + row_q
    rel_row_ptr = (
        rel_ptr + (q_start + row_q) * stride_r_t + (q_head_start + row_h) * stride_r_h
    )
    off_d = tl.arange(0, BLOCK_D)
    d_valid = off_d < head_dim
    offs_k = tl.arange(0, BLOCK_K)
    k_head_ptr = k_ptr + pid_kh * stride_k_h + off_d[:, None] * stride_k_d
    v_head_ptr = v_ptr + pid_kh * stride_v_h + off_d[None, :] * stride_v_d

    m_i = tl.full((BLOCK_QH,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_QH,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_QH, BLOCK_D), dtype=tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = softmax_scale * log2e

    k_lo = 0
    if window_left >= 0:
        k_lo = tl.maximum(prefix_len - window_left, 0)
    k_hi = tl.minimum(kv_len, prefix_len + tl.minimum(BLOCK_Q, q_len))
    k_last = tl.maximum(k_hi - 1, 0)
    split_len = tl.cdiv(tl.cdiv(k_hi - k_lo, num_splits), BLOCK_K) * BLOCK_K
    s_lo = k_lo + pid_s * split_len
    s_hi = tl.minimum(s_lo + split_len, k_hi)

    for k_start in range(s_lo, s_hi, BLOCK_K):
        k_pos = k_start + offs_k
        page = tl.load(bt_row + tl.minimum(k_pos, k_last) // page_size).to(tl.int64)
        k, v = _kv_tile_load(
            page, k_pos, k_last, k_head_ptr, v_head_ptr, d_valid,
            stride_k_b, stride_k_p, page_size, EVEN_D,
        )
        bias, valid = _rel_bias_load(
            k_pos, s_hi, rel_row_ptr, row_abs, row_valid, stride_r_e,
            rel_extent, window_left,
        )

        scores = tl.dot(q, k) * qk_scale + bias.to(tl.float32) * log2e
        scores = tl.where(valid, scores, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        m_safe = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(scores - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    # Unnormalised partial; an empty split stores (0, -inf, 0).
    part = (pid_b * tl.num_programs(1) + pid_kh) * num_splits + pid_s
    tl.store(
        mid_acc_ptr
        + part * (BLOCK_QH * BLOCK_D)
        + off_qh[:, None] * BLOCK_D
        + off_d[None, :],
        acc,
        cache_modifier=PART_STORE_CM,
    )
    tl.store(mid_ml_ptr + part * (2 * BLOCK_QH) + off_qh, m_i, cache_modifier=PART_STORE_CM)
    tl.store(mid_ml_ptr + part * (2 * BLOCK_QH) + BLOCK_QH + off_qh, l_i, cache_modifier=PART_STORE_CM)

    if FUSED_COMBINE:
        # Last-arriving split merges the partials. The partials were stored
        # write-through (sc0 sc1, PART_STORE_CM=".wt"); every wave drains them
        # (the workgroup barrier does not wait for global stores on gfx950)
        # before one relaxed agent-scope ticket, and the merge reads them back
        # with cache-bypassing ".cv" loads. An acq_rel ticket would instead
        # write back the whole L2 per CTA (buffer_wbl2), measured 2-3x slower.
        tl.inline_asm_elementwise(
            "s_waitcnt vmcnt(0)", "=v,v", [off_qh], dtype=tl.int32, is_pure=False, pack=1
        )
        tl.debug_barrier()
        cnt = counter_ptr + pid_b * tl.num_programs(1) + pid_kh
        ticket = tl.atomic_add(cnt, 1, sem="relaxed", scope="gpu")
        if ticket == num_splits - 1:
            # Self-reset for the next launch / graph replay.
            tl.atomic_xchg(cnt, 0, sem="relaxed", scope="gpu")
            base = (pid_b * tl.num_programs(1) + pid_kh) * num_splits
            m = tl.full((BLOCK_QH,), float("-inf"), dtype=tl.float32)
            for s in range(num_splits):
                m = tl.maximum(
                    m,
                    tl.load(mid_ml_ptr + (base + s) * (2 * BLOCK_QH) + off_qh,
                            cache_modifier=".cv"),
                )
            m_ref = tl.where(m == float("-inf"), 0.0, m)
            l_tot = tl.zeros((BLOCK_QH,), dtype=tl.float32)
            o = tl.zeros((BLOCK_QH, BLOCK_D), dtype=tl.float32)
            for s in range(num_splits):
                ml = mid_ml_ptr + (base + s) * (2 * BLOCK_QH)
                w = tl.exp2(tl.load(ml + off_qh, cache_modifier=".cv") - m_ref)
                l_tot += w * tl.load(ml + BLOCK_QH + off_qh, cache_modifier=".cv")
                o += w[:, None] * tl.load(
                    mid_acc_ptr
                    + (base + s) * (BLOCK_QH * BLOCK_D)
                    + off_qh[:, None] * BLOCK_D
                    + off_d[None, :],
                    cache_modifier=".cv",
                )
            o /= tl.where(l_tot > 0.0, l_tot, 1.0)[:, None]
            o = tl.reshape(o, (BLOCK_Q, BLOCK_H, BLOCK_D))
            out_block_ptr = tl.make_block_ptr(
                base=out_ptr + q_start * stride_o_t + q_head_start * stride_o_h,
                shape=(q_len, gqa_group_size, head_dim),
                strides=(stride_o_t, stride_o_h, stride_o_d),
                offsets=(0, 0, 0),
                block_shape=(BLOCK_Q, BLOCK_H, BLOCK_D),
                order=(2, 1, 0),
            )
            tl.store(
                out_block_ptr, o.to(out_ptr.dtype.element_ty), boundary_check=(0, 1, 2)
            )


# Per-device split-arrival counters for the fused combine; each launch leaves
# them at zero. Grown only outside graph capture (a capture would put them in
# the graph's private pool).
# Without counters (first sight of a larger batch during capture) the caller
# falls back to the separate combine kernel. Outgrown buffers are retained:
# a graph captured earlier may still reference them.
_SPLIT_COUNTERS: dict[torch.device, torch.Tensor] = {}
_RETIRED_SPLIT_COUNTERS: list[torch.Tensor] = []


def _split_counters(device: torch.device, n: int) -> torch.Tensor | None:
    buf = _SPLIT_COUNTERS.get(device)
    if buf is None or buf.numel() < n:
        if torch.cuda.is_current_stream_capturing():
            return None
        if buf is not None:
            _RETIRED_SPLIT_COUNTERS.append(buf)
        buf = torch.zeros(max(n, 4096), dtype=torch.int32, device=device)
        _SPLIT_COUNTERS[device] = buf
    return buf


def _mi355x_tuned_paths() -> bool:
    """Use the MI355X-tuned launch shapes, XCD remap and window-split decode.

    gfx950 only by default. ZENITH_INKLING_RAH=1/0 forces them on/off (e.g. to
    measure them on MI300A); unset leaves every other arch on v8's kernels
    and launch paths (the ones gfx90a was tuned with).
    """
    force = os.environ.get("ZENITH_INKLING_RAH", "")
    if force in ("0", "1"):
        return force == "1"
    return on_gfx950()


def _num_xcds() -> int:
    """XCDs the hardware deals workgroups across (round-robin)."""
    if on_gfx950():
        return 8  # MI355X / MI350X
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    if arch.startswith("gfx942"):
        # MI300A: 6 XCDs x 38 CUs; MI300X/MI325X: 8 x 38.
        cus = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        return 6 if cus == 228 else 8
    return 1  # gfx90a: one XCD per GCD


def decode_window_split_count(
    *, batch: int, num_kv_heads: int, window_left: int, max_seqlen_q: int, block_k: int
) -> int:
    """KV splits for windowed decode/verify (1 = single-pass kernel).

    Depends only on values fixed at CUDA-graph capture.
    """
    ctas = max(1, batch * num_kv_heads)
    if ctas >= 512:
        return 1
    # Measured on MI355X: c16 (32 CTAs) -> 6, c64 / draft (128) -> 4, c128 (256) -> 2.
    s = min(8, 512 // ctas)
    n_tiles = triton.cdiv(window_left + max_seqlen_q, block_k)
    s = max(1, min(s, n_tiles // 2))
    # Drop splits that the whole-tile split length would leave empty.
    return triton.cdiv(n_tiles, triton.cdiv(n_tiles, s))


def verify_split_count(*, batch: int, num_kv_heads: int, max_kv_len: int) -> int:
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
    RELSKIP: tl.constexpr = False,  # patch rocm_inkling_rel_attn_dec.py
    HOIST_V: tl.constexpr = False,
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
    rel_lo = prefix_len - rel_extent + 1

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
        if HOIST_V:
            # (patch rocm_inkling_rel_attn_dec.py) issue the V gather with K's
            v = tl.load(
                v_ptr
                + page[:, None] * stride_v_b
                + page_off[:, None] * stride_v_p
                + pid_kh * stride_v_h
                + off_d[None, :] * stride_v_d,
                mask=k_valid[:, None] & d_valid[None, :],
                other=0.0,
            )
        qk = tl.dot(q, k)

        dist = q_abs[:, None] - k_pos[None, :]
        score_valid = q_valid[:, None] & k_valid[None, :] & (dist >= 0)
        if window_left >= 0:
            score_valid &= dist <= window_left

        off_h = tl.arange(0, BLOCK_H)
        # (patch rocm_inkling_rel_attn_dec.py) RELSKIP: a tile ending before
        # rel_extent of this program's first query row has bias 0 everywhere.
        if RELSKIP:
            if k_start + BLOCK_K > rel_lo:
                scores = _inkling_rel_bias_scores(
                    qk, dist, q_valid, q_rows, off_h, rel_ptr, q_start,
                    q_head_start, gqa_group_size, stride_r_t, stride_r_h,
                    stride_r_e, softmax_scale, rel_extent, BLOCK_QH, BLOCK_K,
                )
            else:
                # + 0.0 keeps the served rounding: without it the scale is
                # contracted into the exp2 argument (fma(qk, c, -m)).
                scores = qk * (softmax_scale * log2e) + 0.0
        else:
            scores = _inkling_rel_bias_scores(
                qk, dist, q_valid, q_rows, off_h, rel_ptr, q_start,
                q_head_start, gqa_group_size, stride_r_t, stride_r_h,
                stride_r_e, softmax_scale, rel_extent, BLOCK_QH, BLOCK_K,
            )
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

        if not HOIST_V:
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
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    rel_extent: int,
    rel_logits: torch.Tensor,
    num_splits: int = 32,
    max_kv_len: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paged varlen attention with Inkling's relative score modification."""
    del num_splits
    if not causal or window_size[1] not in (0, -1):
        raise NotImplementedError("Inkling ROCm attention requires causal masking")
    if q.ndim != 3 or key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("expected q [T,H,D] and paged K/V [B,P,Hkv,D]")
    if rel_logits.shape != (q.shape[0], q.shape[1], rel_extent):
        raise ValueError(
            f"relative logits have shape {tuple(rel_logits.shape)}, expected "
            f"{(q.shape[0], q.shape[1], rel_extent)}"
        )

    num_kv_heads = key_cache.shape[2]
    if q.shape[1] % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if out is None:
        out = torch.empty_like(q)
    gqa_group_size = q.shape[1] // num_kv_heads
    batch = cu_seqlens_q.shape[0] - 1
    if max_kv_len is None:
        max_kv_len = key_cache.shape[0] * key_cache.shape[1]
    if use_gfx950_gluon_decode(
        max_query_len=max_seqlen_q,
        page_size=key_cache.shape[1],
        head_dim=q.shape[2],
    ):
        # Import only after the architecture guard. The implementation uses
        # gfx950-only CDNA4 Gluon layouts and async-copy operations.
        from vllm.models.inkling.amd.ops.gluon.rel_mha_decode_gfx950 import (
            gluon_rel_mha_decode_gfx950,
        )

        return gluon_rel_mha_decode_gfx950(
            q,
            key_cache,
            value_cache,
            block_table,
            cache_seqlens,
            max_kv_len,
            rel_logits,
            cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            window_left=window_size[0],
            softmax_scale=softmax_scale,
            out=out,
        )
    if use_gfx950_gluon_extend(
        max_query_len=max_seqlen_q,
        max_kv_len=max_kv_len,
        page_size=key_cache.shape[1],
        head_dim=q.shape[2],
        window_left=window_size[0],
    ):
        from vllm.models.inkling.amd.ops.gluon.rel_mha_extend_gfx950 import (
            gluon_rel_mha_extend_gfx950,
        )

        # The copied kernel retains TokenSpeed's cu_seqlens_kv parameter for
        # API compatibility but does not consume it.
        return cast(
            torch.Tensor,
            gluon_rel_mha_extend_gfx950(
                q,
                cu_seqlens_q,
                cu_seqlens_q,
                key_cache,
                value_cache,
                block_table,
                cache_seqlens,
                window_left=window_size[0],
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_kv_len,
                rel_logits=rel_logits,
                softmax_scale=softmax_scale,
                out=out,
            ),
        )
    if use_split_kv_decode(
        max_query_len=max_seqlen_q,
        max_kv_len=max_kv_len,
        page_size=key_cache.shape[1],
        window_left=window_size[0],
    ):
        return inkling_rel_attention_split_kv_decode(
            q,
            key_cache,
            value_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            window_left=window_size[0],
            rel_extent=rel_extent,
            rel_logits=rel_logits,
            max_kv_len=max_kv_len,
            out=out,
        )
    # Global-layer MTP verify: split the KV range across programs. Zenith
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
        # (patch rocm_inkling_rel_attn_dec.py) q_len <= 2: the
        # INKLING_DEC_ATTN_G tile (default 2-query tiles on 2 warps with the
        # rel-bias tile skip).
        dec2 = _DEC_ATTN and max_seqlen_q <= 2
        vq, vk, vw, vh, vr = _DEC_ATTN_G if dec2 else (4, 64, 4, 0, 0)
        part_shape = (batch, num_kv_heads, verify_splits)
        mid_acc = torch.empty(
            (*part_shape, vq * block_h, block_d), dtype=torch.float32, device=q.device
        )
        mid_ml = torch.empty(
            (*part_shape, 2, vq * block_h), dtype=torch.float32, device=q.device
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
            BLOCK_Q=vq,
            BLOCK_K=vk,
            num_warps=vw,
            num_stages=1,
            RELSKIP=bool(vr),
            HOIST_V=bool(vh),
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
            BLOCK_Q=vq,
            num_warps=4,
        )
        return out
    window_left = window_size[0]
    tuned = _mi355x_tuned_paths()
    if tuned and max_seqlen_q <= 4 and window_left >= 0:
        split_bk = 32
        win_splits = decode_window_split_count(
            batch=batch,
            num_kv_heads=num_kv_heads,
            window_left=window_left,
            max_seqlen_q=max_seqlen_q,
            block_k=split_bk,
        )
        if win_splits > 1:
            block_h = triton.next_power_of_2(gqa_group_size)
            block_d = triton.next_power_of_2(q.shape[2])
            part_shape = (batch, num_kv_heads, win_splits)
            mid_acc = torch.empty(
                (*part_shape, 4 * block_h, block_d), dtype=torch.float32, device=q.device
            )
            mid_ml = torch.empty(
                (*part_shape, 2, 4 * block_h), dtype=torch.float32, device=q.device
            )
            counters = _split_counters(q.device, batch * num_kv_heads)
            fused = counters is not None
            _inkling_rel_attention_window_split_kernel[(win_splits, num_kv_heads, batch)](
                q,
                key_cache,
                value_cache,
                rel_logits,
                mid_acc,
                mid_ml,
                out,
                counters if fused else mid_ml,
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
                out.stride(0),
                out.stride(1),
                out.stride(2),
                block_table.stride(0),
                page_size=key_cache.shape[1],
                rel_extent=rel_extent,
                window_left=window_left,
                num_splits=win_splits,
                BLOCK_Q=4,
                BLOCK_K=split_bk,
                FUSED_COMBINE=fused,
                PART_STORE_CM=".wt" if fused else "",
                num_warps=4,
                num_stages=3,
            )
            if not fused:
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
                num_splits=win_splits,
                BLOCK_Q=4,
                num_warps=4,
                )
            return out
    if tuned:
        # Launch shapes measured on MI355X (GQA 4, head_dim 128, window 511):
        # prefill wants 32 query rows x 4 heads per program with 32-key tiles and
        # no software pipelining (VGPR-bound occupancy); decode and MTP verify/draft
        # want 4 rows (draft padded from 1: 16-row MFMA tiles beat 4-row ones) with
        # 32-key tiles and a 3-stage pipeline to keep more KV bytes in flight.
        # A mixed step (prefill chunks plus 4-token decodes) gets one launch per
        # tile size: 32-row blocks would leave 28 of every 32 rows idle in each
        # decode program, which costs most on the global layers' full-length KV.
        short_q = 16
        if max_seqlen_q < 64:
            launches = [(4, 32, 4, 3, 0, max_seqlen_q)]
        elif batch == 1:
            launches = [(32, 32, 4, 1, 0, max_seqlen_q)]
        else:
            launches = [(32, 32, 4, 1, 1, max_seqlen_q), (4, 32, 4, 3, 2, short_q)]
        for block_q, block_k, nw, ns, q_split, span_q in launches:
            _launch_rel_attention(
                q, key_cache, value_cache, rel_logits, out, block_table,
                cache_seqlens, cu_seqlens_q, gqa_group_size, softmax_scale,
                rel_extent, window_size[0], num_kv_heads, batch,
                block_q, block_k, nw, ns, q_split, short_q, span_q,
            )
        return out
    # Other archs (MI250X gfx90a, MI300A gfx942): v8's kernels and launches.
    block_q = 1 if max_seqlen_q == 1 else 4
    grid = (triton.cdiv(max_seqlen_q, block_q), num_kv_heads, batch)
    bq16 = prefill_attn_bq16_mode(max_seqlen_q)
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
        # that exists instead of cdiv(max_seqlen_q, 4) per request, almost
        # all of which would exit at once. Same per-program work, so the
        # output is bit-identical. Zenith patch inkling_mixed_flat_grid.py.
        cu_units = torch.empty(batch + 1, dtype=torch.int32, device=q.device)
        _inkling_rel_attention_flat_units[(1,)](
            cu_seqlens_q,
            cu_units,
            batch,
            BLOCK_Q=block_q,
            BLOCK=triton.next_power_of_2(batch + 1),
            num_warps=4,
        )
        # sum_b cdiv(q_len_b, 4) <= (total_q + 3 * batch) // 4
        flat_grid = ((q.shape[0] + 3 * batch) // block_q, num_kv_heads)
        _inkling_rel_attention_flat_kernel[flat_grid](
            q,
            key_cache,
            value_cache,
            rel_logits,
            out,
            block_table,
            cache_seqlens,
            cu_seqlens_q,
            cu_units,
            batch,
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
            out.stride(0),
            out.stride(1),
            out.stride(2),
            block_table.stride(0),
            page_size=key_cache.shape[1],
            rel_extent=rel_extent,
            window_left=window_size[0],
            SEARCH=max(11, batch.bit_length()),
            BLOCK_Q=block_q,
            BLOCK_K=64,
            num_warps=4,
            num_stages=1,
        )
        return out
    if (
        _DEC_ATTN
        and 1 < max_seqlen_q <= 4
        and window_size[0] >= 0
        and batch * num_kv_heads <= _DEC_ATTN_LOCAL_SPLIT_B
    ):
        # (patch rocm_inkling_rel_attn_dec.py) local verify at small batch:
        # 2 KV splits per (request, KV head) + the served combine.
        block_h = triton.next_power_of_2(gqa_group_size)
        block_d = triton.next_power_of_2(q.shape[2])
        mid_acc = torch.empty(
            (batch, num_kv_heads, 2, 4 * block_h, block_d),
            dtype=torch.float32, device=q.device,
        )
        mid_ml = torch.empty(
            (batch, num_kv_heads, 2, 2, 4 * block_h),
            dtype=torch.float32, device=q.device,
        )
        _inkling_rel_attention_split_kernel[(2, num_kv_heads, batch)](
            q, key_cache, value_cache, rel_logits, mid_acc, mid_ml, block_table,
            cache_seqlens, cu_seqlens_q, gqa_group_size, q.shape[2], softmax_scale,
            q.stride(0), q.stride(1), q.stride(2),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            key_cache.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            value_cache.stride(3),
            rel_logits.stride(0), rel_logits.stride(1), rel_logits.stride(2),
            block_table.stride(0),
            page_size=key_cache.shape[1], rel_extent=rel_extent,
            window_left=window_size[0], num_splits=2, BLOCK_Q=4, BLOCK_K=64,
            num_warps=4, num_stages=1, RELSKIP=True,
        )
        _inkling_rel_attention_split_combine[(num_kv_heads, batch)](
            mid_acc, mid_ml, out, cu_seqlens_q, gqa_group_size, q.shape[2],
            out.stride(0), out.stride(1), out.stride(2),
            num_splits=2, BLOCK_Q=4, num_warps=4,
        )
        return out
    _inkling_rel_attention_kernel[grid](
        q,
        key_cache,
        value_cache,
        rel_logits,
        out,
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
        out.stride(0),
        out.stride(1),
        out.stride(2),
        block_table.stride(0),
        page_size=key_cache.shape[1],
        rel_extent=rel_extent,
        window_left=window_size[0],
        BLOCK_Q=block_q,
        BLOCK_K=64,
        num_warps=4,
        num_stages=1,
        RELSKIP=_DEC_ATTN,  # patch rocm_inkling_rel_attn_dec.py
    )
    return out


def _launch_rel_attention(
    q, key_cache, value_cache, rel_logits, out, block_table, cache_seqlens,
    cu_seqlens_q, gqa_group_size, softmax_scale, rel_extent, window_left,
    num_kv_heads, batch, block_q, block_k, nw, ns, q_split, short_q, span_q,
):
    grid = (triton.cdiv(span_q, block_q), num_kv_heads, batch)
    # Workgroups are dealt round-robin over the XCDs (8 on MI355X); remap only
    # when the query-block axis divides evenly (long prefill), else launch
    # order as is.
    xcd = _num_xcds()
    if grid[0] % xcd or grid[0] < 2 * xcd:
        xcd = 1
    _inkling_rel_attention_tuned_kernel[grid](
        q,
        key_cache,
        value_cache,
        rel_logits,
        out,
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
        out.stride(0),
        out.stride(1),
        out.stride(2),
        block_table.stride(0),
        page_size=key_cache.shape[1],
        rel_extent=rel_extent,
        window_left=window_left,
        BLOCK_Q=block_q,
        BLOCK_K=block_k,
        XCD_REMAP=xcd,
        Q_SPLIT=q_split,
        SHORT_Q=short_q,
        num_warps=nw,
        num_stages=ns,
    )
