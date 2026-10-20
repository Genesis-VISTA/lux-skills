#!/usr/bin/env python3
"""
Faster MTP-verify decode attention for Inkling on MI250X (gfx90a).

THE COST AS IT PRESENTS

At T=512 (256 requests x q_len 2, ~8.2k context) a stage-1 decode step spends
~1.1 ms per global layer in _inkling_rel_attention_split_kernel (1.08 GB of
paged K|V at ~1.05 TB/s) and ~90-110 us per local layer (window 511) in
_inkling_rel_attention_kernel. A gather-only probe of the same pages reaches
~1.15-1.25 TB/s (f61zBFfeHCns), so the global kernel is ~10% off the
achievable paged-read bound, the local one ~30%.

Both kernels gather the relative-logit bias for every KV tile although it is
non-zero only within rel_extent of the query; at 8k context that is 1 tile in
130. The global split kernel also runs 16-row (BLOCK_Q 4 x 4 heads) MFMA
tiles for 8 live rows (q_len 2).

THE FIX (INKLING_DEC_ATTN=1; default off = served kernels and launches)

- RELSKIP (constexpr, both kernels): skip the bias gather for KV tiles that
  end before rel_extent of the program's first query row. Those tiles add a
  bias of exactly 0; the score is the served expression (+ 0.0 so that the
  scale is not contracted into the exp2 argument). Bit-identical. On for the
  local single-pass kernel.
- Global verify (q_len <= 2): the INKLING_DEC_ATTN_G tile (BLOCK_Q, BLOCK_K,
  num_warps, HOIST_V, RELSKIP), default 2,64,2,0,1: 2-query tiles on 2 warps
  with RELSKIP; same 64-key tiles, split count and combine.
- Local verify with batch * kv_heads <= INKLING_DEC_ATTN_LOCAL_SPLIT_B
  (default 0 = never): the split-KV kernel with 2 splits + the served combine.

Unit tests f61zSJVGXVKV / f61zamxaAbq1 (served LBNHC layout, 8.2-9.2k
context, graph replay, us; ulp = bf16 output vs the served kernel: fraction
> 0 / > 1 ulp):

                    served   patched                     ulp
  local  B=256        89.0     81.1  RELSKIP             bit-identical
  local  B=128        68.6     62.6  RELSKIP             bit-identical
  global B=256      1026.6    996.2  2,64,2,0,1          bit-identical
  global B=128       541.2    521.8  2,64,2,0,1          bit-identical
  opt-in:
  global B=256      1026.6    986.5  G=2,64,2,1,0        3e-5 / 0
  global B=128       541.2    513.3  G=2,64,2,1,0        5e-5 / 8e-6
  global B=256      1026.6    949.2  G=2,128,2,1,1       0.24 / 0.055
  global B=128       541.2    503.5  G=2,128,2,1,1       0.29 / 0.077
  local  B=128        68.6     51.8  LOCAL_SPLIT_B=128   0.30 / 0.087

Numerics: the defaults are bit-identical to the served kernels (B = 64, 128,
256). Without RELSKIP the BLOCK_Q 2 tile differs by <= 1 ulp on ~0.005% of
outputs (FMA contraction of the score differs). BLOCK_K 128 and the local
split change the online-softmax tile order (bf16 P tiles round against a
different running max): multi-ulp on 5-9% of outputs, mean |err| vs an fp64
reference within +2% / -2% of the served kernel's.

Apply after inkling_verify_split_kv.py and inkling_single_pass_kv_bounds.py
(alphabetical order in docker/patches does this).

    python rocm_inkling_rel_attn_dec.py [--check] [--revert] [--root DIR]
"""
import argparse
import importlib.util
import os
import sys

MARKER = "patch rocm_inkling_rel_attn_dec.py"
D = "vllm/models/inkling/amd/"

# ------------------------------------------------------------- module level
FLAG_OLD = '''def verify_split_count(*, batch: int, num_kv_heads: int, max_kv_len: int) -> int:
'''
FLAG_NEW = '''# (patch rocm_inkling_rel_attn_dec.py) INKLING_DEC_ATTN=1: rel-bias tile skip
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


def verify_split_count(*, batch: int, num_kv_heads: int, max_kv_len: int) -> int:
'''

# The served per-tile score block (identical in both kernels).
SCORE_OLD = '''        scores = tl.dot(q, k) * (softmax_scale * log2e)

        dist = q_abs[:, None] - k_pos[None, :]
        score_valid = q_valid[:, None] & k_valid[None, :] & (dist >= 0)
        if window_left >= 0:
            score_valid &= dist <= window_left
'''
BIAS_OLD = '''        rel_dist = dist[:, None, :]
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
'''
SCORE_NEW = '''        qk = tl.dot(q, k)

        dist = q_abs[:, None] - k_pos[None, :]
        score_valid = q_valid[:, None] & k_valid[None, :] & (dist >= 0)
        if window_left >= 0:
            score_valid &= dist <= window_left
'''
BIAS_NEW = '''        # (patch rocm_inkling_rel_attn_dec.py) RELSKIP: a tile ending before
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
'''

# ------------------------------------------------- _inkling_rel_attention_kernel
SINGLE_SIG_OLD = '''    BLOCK_QH: tl.constexpr,
):
'''
SINGLE_SIG_NEW = '''    BLOCK_QH: tl.constexpr,
    RELSKIP: tl.constexpr = False,  # patch rocm_inkling_rel_attn_dec.py
):
'''
SINGLE_RELLO_OLD = '''    log2e: tl.constexpr = 1.4426950408889634
'''
SINGLE_RELLO_NEW = '''    log2e: tl.constexpr = 1.4426950408889634
    # first key that is within rel_extent of the block's first query row
    rel_lo = prefix_len + q_block - rel_extent + 1
'''

# ------------------------------------------- _inkling_rel_attention_split_kernel
SPLIT_SIG_OLD = SINGLE_SIG_OLD
SPLIT_SIG_NEW = '''    BLOCK_QH: tl.constexpr,
    RELSKIP: tl.constexpr = False,  # patch rocm_inkling_rel_attn_dec.py
    HOIST_V: tl.constexpr = False,
):
'''
SPLIT_RELLO_OLD = SINGLE_RELLO_OLD
SPLIT_RELLO_NEW = '''    log2e: tl.constexpr = 1.4426950408889634
    rel_lo = prefix_len - rel_extent + 1
'''
SPLIT_SCORE_OLD = '''        k = tl.load(
            k_ptr
            + page[None, :] * stride_k_b
            + page_off[None, :] * stride_k_p
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d,
            mask=d_valid[:, None] & k_valid[None, :],
            other=0.0,
        )
''' + SCORE_OLD
SPLIT_SCORE_NEW = '''        k = tl.load(
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
''' + SCORE_NEW
SPLIT_V_OLD = '''        v = tl.load(
            v_ptr
            + page[:, None] * stride_v_b
            + page_off[:, None] * stride_v_p
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d,
            mask=k_valid[:, None] & d_valid[None, :],
            other=0.0,
        )
        acc += tl.dot(p.to(v.dtype), v)
'''
SPLIT_V_NEW = '''        if not HOIST_V:
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
'''

# ------------------------------------------------------------- launches
VERIFY_OLD = '''        block_h = triton.next_power_of_2(gqa_group_size)
        block_d = triton.next_power_of_2(q.shape[2])
        part_shape = (batch, num_kv_heads, verify_splits)
        mid_acc = torch.empty(
            (*part_shape, 4 * block_h, block_d), dtype=torch.float32, device=q.device
        )
        mid_ml = torch.empty(
            (*part_shape, 2, 4 * block_h), dtype=torch.float32, device=q.device
        )
'''
VERIFY_NEW = '''        block_h = triton.next_power_of_2(gqa_group_size)
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
'''
VERIFY_K_OLD = '''            num_splits=verify_splits,
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
'''
VERIFY_K_NEW = '''            num_splits=verify_splits,
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
'''
SINGLE_LAUNCH_OLD = '''        return out
    _inkling_rel_attention_kernel[grid](
'''
SINGLE_LAUNCH_NEW = '''        return out
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
'''
SINGLE_LAUNCH_K_OLD = '''        BLOCK_Q=block_q,
        BLOCK_K=64,
        num_warps=4,
        num_stages=1,
    )
    return out
'''
SINGLE_LAUNCH_K_NEW = '''        BLOCK_Q=block_q,
        BLOCK_K=64,
        num_warps=4,
        num_stages=1,
        RELSKIP=_DEC_ATTN,  # patch rocm_inkling_rel_attn_dec.py
    )
    return out
'''

# (name, function whose body scopes the match or None for whole file, old, new)
SITES = [
    ("flag + bias helper", None, FLAG_OLD, FLAG_NEW),
    ("single: signature", "_inkling_rel_attention_kernel", SINGLE_SIG_OLD, SINGLE_SIG_NEW),
    ("single: rel_lo", "_inkling_rel_attention_kernel", SINGLE_RELLO_OLD, SINGLE_RELLO_NEW),
    ("single: qk", "_inkling_rel_attention_kernel", SCORE_OLD, SCORE_NEW),
    ("single: bias", "_inkling_rel_attention_kernel", BIAS_OLD, BIAS_NEW),
    ("split: signature", "_inkling_rel_attention_split_kernel", SPLIT_SIG_OLD, SPLIT_SIG_NEW),
    ("split: rel_lo", "_inkling_rel_attention_split_kernel", SPLIT_RELLO_OLD, SPLIT_RELLO_NEW),
    ("split: qk + V hoist", "_inkling_rel_attention_split_kernel", SPLIT_SCORE_OLD,
     SPLIT_SCORE_NEW),
    ("split: bias", "_inkling_rel_attention_split_kernel", BIAS_OLD, BIAS_NEW),
    ("split: V load", "_inkling_rel_attention_split_kernel", SPLIT_V_OLD, SPLIT_V_NEW),
    ("verify buffers", "inkling_fa4_rel_attention", VERIFY_OLD, VERIFY_NEW),
    ("verify launch", "inkling_fa4_rel_attention", VERIFY_K_OLD, VERIFY_K_NEW),
    ("local split launch", "inkling_fa4_rel_attention", SINGLE_LAUNCH_OLD, SINGLE_LAUNCH_NEW),
    ("single launch", "inkling_fa4_rel_attention", SINGLE_LAUNCH_K_OLD, SINGLE_LAUNCH_K_NEW),
]
TARGET = D + "ops/fa4_rel_attention.py"


def func_span(src, func):
    """[start, end) of a top-level def (through the next top-level statement)."""
    if func is None:
        return 0, len(src)
    start = src.find(f"\ndef {func}(")
    if start < 0:
        return None
    start += 1
    nxt = len(src)
    for tok in ("\n@", "\ndef ", "\nclass ", "\n_", "\n#"):
        i = src.find(tok, start + 1)
        if 0 <= i < nxt:
            nxt = i
    return start, nxt


def apply(src, revert):
    problems = []
    for name, func, old, new in (reversed(SITES) if revert else SITES):
        a, b = (new, old) if revert else (old, new)
        span = func_span(src, func)
        if span is None:
            problems.append(f"{name}: def {func} not found")
            continue
        s, e = span
        body = src[s:e]
        n = body.count(a)
        if n != 1:
            problems.append(f"{name} (matched {n}x in {func or 'file'}, expected 1)")
            continue
        src = src[:s] + body.replace(a, b, 1) + src[e:]
    return src, problems


def find_root():
    """site-packages dir holding vllm/, without importing vllm."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        sys.exit("vllm is not installed; run this inside the container")
    return os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report status without writing")
    ap.add_argument("--revert", action="store_true", help="undo the patch")
    ap.add_argument("--root", help="patch the vllm/ tree under this dir "
                    "instead of the installed vLLM's")
    args = ap.parse_args()

    path = os.path.join(args.root or find_root(), TARGET)
    if not os.path.exists(path):
        print(f"ERROR: {path} missing; vLLM layout changed", file=sys.stderr)
        return 1
    with open(path) as fh:
        src = fh.read()
    if (MARKER in src) != args.revert:
        print(f"already {'reverted' if args.revert else 'patched'}: {path}")
        return 0
    patched, problems = apply(src, args.revert)
    if problems:
        print("ERROR: patch target not found:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print("Inkling's fa4_rel_attention.py changed (are "
              "inkling_verify_split_kv.py and inkling_single_pass_kv_bounds.py "
              "applied first?); re-derive or retire this patch", file=sys.stderr)
        return 1
    if args.check:
        print(f"would {'revert' if args.revert else 'patch'}: {path}")
        return 0
    with open(path, "w") as fh:
        fh.write(patched)
    print(f"{'reverted' if args.revert else 'patched'}: {path}  "
          "(Inkling decode attention, INKLING_DEC_ATTN)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
