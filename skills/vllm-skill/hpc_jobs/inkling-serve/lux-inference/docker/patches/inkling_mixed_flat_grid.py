#!/usr/bin/env python3
"""
Launch Inkling's single-pass attention on a flat grid of the (request, query
block) pairs that exist, for steps that mix prefill chunks with decode rows.

THE PROBLEM

Every step with max_query_len > 4 (any step carrying a prefill chunk) runs
_inkling_rel_attention_kernel on grid

    (cdiv(bucket_pow2(max_query_len), 4), num_kv_heads, batch)

so every request gets as many 4-query programs as the longest one. At c512
with MTP 1 a mixed step is 512 decode rows (q_len 2) plus a 7168-token chunk:
2048 x 1 x 513 = 1.05M programs, of which 512 x 2047 do nothing but load
cu_seqlens_q and exit at `q_block >= q_len`. The global-layer verify split
(inkling_verify_split_kv.py) needs max_query_len <= 4, so it never applies to
these steps either. On MI250X (job f61pMJKydXvf, PP stage 0, 21 layers, TP8,
MTP 1) the decode rows add 1.6 ms per local layer (they need ~0.08 ms alone)
and 3.95 ms per global layer: 41 ms of a 306 ms forward.

THE FIX

For max_query_len > 4 and batch > 1, launch (units, num_kv_heads) programs,
units = sum_b cdiv(q_len_b, 4). A one-program Triton kernel writes the unit
prefix sums cu_units (on the device, so no host sync and nothing changes
under graph capture); each attention program binary-searches cu_units for
its request b and query block (pid - cu_units[b]) * 4. The grid is sized
from the host-known bound (total_q + 3 * batch) // 4 >= units; programs past
cu_units[batch] exit at once. Everything after the row lookup is the served
kernel's body verbatim, including the KV-loop bounds of
inkling_single_pass_kv_bounds.py: every program owns the same request, query
block and key tiles as before and runs the same instructions, so the output
is bit-identical.

Pure decode / MTP verify (max_query_len <= 4, the CUDA-graph-captured steps),
the verify split, the split-KV decode path and single-request prefill
(batch 1, where the dense grid is already exact up to the power-of-two
bucket) are left alone.

EVIDENCE

Tioga, PP stage 0 (21 real-weight layers + MTP), TP8, MTP 1, eager, served
stack, natural-text prompts, 512 decode rows at 8.2-9k context plus fresh
8192-token prompts (job f61pmyoRDLPy, prof/sab/runs; the same kernel run from
a hook). Outputs compared with the served kernel on every attention call of
the probed mixed steps (all 21 backbone layers and the MTP layer): ndiff 0,
decode and prefill rows.

    attention core per step (21 layers)       served    flat
    mnbt 8192:  7168 chunk + 512 decode        59.0 ms   26.1 ms
    mnbt 8192:  1024 chunk + 512 decode        21.6      13.5
    mnbt 16384: 7168 + 8192 + 512 decode       78.6      45.7

    stage-0 forward per mixed step (median of 3 episodes, CUDA events)
    mnbt 8192:  7168 + 512 decode             278.2 ms  248.2 ms
    mnbt 8192:  1024 + 512 decode              99.3      92.7
    mnbt 16384: 16384 tokens + 512 decode     489.1     458.7

Per call: local 2.10 -> 0.63 ms (prefill rows alone 0.49), global at 7168
7.07 -> 4.92 ms (prefill rows alone 2.99). The rest of the global decode-row
cost is their ~8.4k-key walk, near the HBM floor once it overlaps the prefill
programs; splitting it (the verify-split partition) gained nothing more and
is not bit-identical.
"""
import argparse
import importlib.util
import os
import sys

TARGET = "vllm/models/inkling/amd/ops/fa4_rel_attention.py"
MARKER = "Zenith patch inkling_mixed_flat_grid.py"

# The flat-grid kernels go just before _inkling_rel_attention_kernel's
# decorators; the tail of inkling_fa4_num_splits is the anchor. Neither this
# nor the dispatch anchor below is touched by the other fa4_rel_attention.py
# fix-ups, and nothing added here matches their anchors or markers, so the
# result is the same whichever order they run in.
OLD_KERNEL_ANCHOR = '''        min(target_ctas // base_ctas, max_splits, (max_kv_len + 127) // 128),
    )


@triton.heuristics(
'''

NEW_KERNEL = '''        min(target_ctas // base_ctas, max_splits, (max_kv_len + 127) // 128),
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
'''

# The dense launch. inkling_verify_split_kv.py inserts its branch in front of
# the first two lines and keeps them; the flat branch goes after them.
OLD_DISPATCH = '''    block_q = 1 if max_seqlen_q == 1 else 4
    grid = (triton.cdiv(max_seqlen_q, block_q), num_kv_heads, batch)
    _inkling_rel_attention_kernel[grid](
'''

NEW_DISPATCH = '''    block_q = 1 if max_seqlen_q == 1 else 4
    grid = (triton.cdiv(max_seqlen_q, block_q), num_kv_heads, batch)
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
    _inkling_rel_attention_kernel[grid](
'''

SITES = [
    ("flat-grid kernels", OLD_KERNEL_ANCHOR, NEW_KERNEL),
    ("mixed-step dispatch", OLD_DISPATCH, NEW_DISPATCH),
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
                    help="restore the dense-grid launch")
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
        print("vLLM changed inkling_fa4_rel_attention; check whether steps with "
              "max_query_len > 4 still launch the dense (query block, KV head, "
              "request) grid, then re-derive or retire this patch",
              file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {path}")
        return 0

    with open(path, "w") as fh:
        fh.write(patched)
    print(f"patched: {path}  (mixed prefill + decode steps on a flat grid)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
