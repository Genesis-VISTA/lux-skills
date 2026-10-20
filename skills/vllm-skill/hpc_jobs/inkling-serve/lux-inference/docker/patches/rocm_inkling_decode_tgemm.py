#!/usr/bin/env python3
"""
Triton GEMMs for Inkling's decode-sized dense projections on MI250X.

THE COST AS IT PRESENTS

In a stage-1 decode step (T = 256 / 512 tokens, HIP graphs) each layer runs
five small dense GEMMs per rank through hipBLASLt / the zero-copy bmm: qkvr
(832 x 4096), wo (4096 x 512, slabs), the sink w13 (1024 x 4096) and w2
(4096 x 512) and the fp32 router gate (264 x 4096). At these M hipBLASLt
picks tiles that leave the GCD under-occupied: sweep f61z2r7ujrDM (isolated,
cold weights, us):

          M=256 BLAS -> Triton      M=512 BLAS -> Triton
  qkvr      37.4 -> 25.7              48.0 -> 44.3
  sink_g1   43.3 -> 36.9              56.5 (kept)
  sink_g2   26.0 -> 19.9              38.2 -> 31.5
  wo (bmm)  25.1 -> 19.2              38.6 -> 31.9
  gate      22.9 -> 18.7              43.4 -> 27.8

In graphs the sink branch does not overlap the routed experts (its exposure,
125 us/layer at T=512, exceeds its kernel sum), so its savings land in full.

THE FIX

ops/decode_tgemm.py: one Triton GEMM (grouped tile order, optional split-K
with fp32 partials and a fixed-order reduce, optional rank-major slab store
for wo) with per-op tile tables, one bucket per CUDA-graph capture size from
128 to 1024 (pick f61zJJbGmcw9; M rounds up to the next bucket); sizes where
hipBLASLt is within 2% keep the served call (qkvr above 512, sink_g1 from 448
except 768). Hooked at the sink GEMMs
(InklingSinkExperts.forward), the router gate (_linear_with_fp32_out, after
the T <= 64 split-K path), qkvr and the zero-copy wo slabs (attention.py).
Only bf16 unquantized weights of the tuned shapes and 0 < M <= 1024; anything
else falls through to the served call.

Numerics: not bit-identical. bf16 outputs differ from hipBLASLt at rounding
level (0.01-0.05% of elements, as hipBLASLt differs from an fp64 reference);
the gate's fp32 logits differ at fp32-ulp level (accumulation order), which
is why the gate is opt-in. Deterministic and run-to-run stable.

INKLING_DEC_TGEMM: unset / "0" = off (served behaviour), "1" = qkvr,wo,sink,
or a comma list of qkvr, wo, sink, gate. Read once at import.

Apply after inkling_gate_fp32_splitk.py and inkling_sconv_zero_copy.py
(alphabetical order in docker/patches does this).

    python rocm_inkling_decode_tgemm.py [--check] [--revert] [--root DIR]
"""
import argparse
import importlib.util
import os
import sys

MARKER = "patch rocm_inkling_decode_tgemm.py"
D = "vllm/models/inkling/amd/"
NEW_FILE = D + "ops/decode_tgemm.py"

MODULE_SRC = r'''# SPDX-License-Identifier: Apache-2.0
# (patch rocm_inkling_decode_tgemm.py) Triton GEMMs for Inkling's decode-sized
# dense projections on MI250X (gfx90a).
#
# At decode M (128..1024 tokens per stage step) hipBLASLt runs the per-rank
# qkvr / wo / sink / router GEMMs 1.1-2x slower than a tuned Triton tile
# (zv7 dec sweep f61z2r7ujrDM; e.g. M=512: sink_g2 38.2 -> 31.5 us, wo bmm
# 38.6 -> 31.9 us, gate 43.4 -> 27.8 us; M=256: qkvr 37.4 -> 25.7 us).
# y = x @ w.T with fp32 accumulation. Split-K configs write fp32 partials and
# a second kernel sums them in fixed order, so results are deterministic and
# batch-invariant per (shape, M bucket). Accumulation order differs from
# hipBLASLt: bf16 outputs change at the rounding level (~0.04% of elements,
# 1 ulp), the fp32 gate logits at the fp32-ulp level.
#
# INKLING_DEC_TGEMM: "0"/unset = off, "1" = qkvr,wo,sink, or a comma list of
# qkvr, wo, sink, gate. Shapes, dtypes or M outside the tuned table fall back
# to the served call.
from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton

_SPEC = os.environ.get("INKLING_DEC_TGEMM", "0").strip()
if _SPEC in ("", "0"):
    ENABLED = frozenset()
elif _SPEC == "1":
    ENABLED = frozenset(("qkvr", "wo", "sink"))
else:
    ENABLED = frozenset(s.strip() for s in _SPEC.split(",") if s.strip())

# (N, K) per rank of the tuned shapes (Inkling-Small, TP8).
_SHAPES = {
    "qkvr": (832, 4096),
    "sink_g1": (1024, 4096),
    "sink_g2": (4096, 512),
    "wo": (4096, 512),
    "gate": (264, 4096),
}

# Per op: (max M, config) buckets, ascending, one per CUDA-graph capture size
# (pick f61zJJbGmcw9: fastest of the swept tiles per size, None = served call
# within 2%). Config: BM, BN, BK, split-K, num_warps, num_stages,
# matrix_instr_nonkdim, waves_per_eu, kpack.
_CFG = {
    "qkvr": (
        (112, None),
        (128, (128, 64, 128, 8, 8, 2, 16, 1, 1)),
        (144, (64, 64, 128, 8, 4, 2, 16, 0, 1)),
        (160, (64, 128, 128, 4, 8, 2, 16, 0, 1)),
        (176, (64, 128, 128, 4, 8, 2, 16, 1, 1)),
        (192, (64, 128, 128, 4, 8, 2, 16, 1, 1)),
        (224, (128, 64, 128, 4, 8, 2, 16, 0, 1)),
        (256, (128, 64, 128, 4, 8, 2, 16, 0, 1)),
        (272, (64, 64, 64, 4, 4, 2, 16, 0, 1)),
        (288, (64, 64, 64, 4, 4, 2, 16, 0, 1)),
        (320, (64, 64, 64, 4, 4, 2, 16, 0, 1)),
        (352, (64, 64, 64, 4, 4, 2, 16, 0, 1)),
        (384, (64, 64, 64, 4, 4, 2, 16, 0, 1)),
        (448, (64, 128, 128, 2, 8, 2, 16, 0, 1)),
        (512, (128, 64, 128, 2, 8, 2, 16, 0, 1)),
        (528, None),
        (544, None),
        (576, None),
        (640, None),
        (768, None),
        (1024, None),
    ),
    "sink_g1": (
        (112, None),
        (128, (64, 128, 128, 4, 8, 2, 16, 1, 1)),
        (144, (64, 128, 128, 4, 8, 2, 16, 1, 1)),
        (160, (64, 128, 128, 4, 8, 2, 16, 1, 1)),
        (176, (64, 128, 128, 4, 8, 2, 16, 1, 1)),
        (192, (64, 128, 128, 4, 8, 2, 16, 1, 1)),
        (224, (64, 64, 128, 4, 4, 1, 16, 0, 1)),
        (256, (64, 64, 128, 4, 4, 1, 16, 0, 1)),
        (272, (64, 64, 64, 4, 4, 2, 16, 0, 1)),
        (288, (64, 64, 64, 4, 4, 2, 16, 0, 1)),
        (320, (64, 64, 64, 4, 4, 2, 16, 0, 1)),
        (352, (64, 128, 128, 2, 8, 2, 16, 0, 1)),
        (384, (64, 128, 128, 2, 8, 2, 16, 0, 1)),
        (448, None),
        (512, None),
        (528, None),
        (544, None),
        (576, None),
        (640, None),
        (768, (128, 128, 64, 2, 8, 2, 16, 0, 1)),
        (1024, None),
    ),
    "sink_g2": (
        (112, None),
        (128, (64, 128, 128, 1, 8, 2, 16, 1, 1)),
        (144, (64, 128, 128, 1, 8, 2, 16, 1, 1)),
        (160, (64, 128, 128, 1, 8, 2, 16, 1, 1)),
        (176, (64, 128, 128, 1, 8, 2, 16, 1, 1)),
        (192, (64, 128, 128, 1, 8, 2, 16, 1, 1)),
        (224, (64, 64, 128, 1, 4, 1, 16, 0, 1)),
        (256, (64, 64, 64, 1, 4, 2, 16, 0, 1)),
        (272, (64, 64, 64, 1, 4, 2, 16, 0, 1)),
        (288, (64, 128, 128, 1, 4, 1, 16, 2, 1)),
        (320, (64, 128, 128, 1, 4, 1, 16, 2, 1)),
        (352, (128, 128, 64, 1, 8, 2, 16, 0, 1)),
        (384, (128, 128, 64, 1, 8, 2, 16, 0, 1)),
        (448, (64, 64, 128, 1, 4, 1, 16, 0, 1)),
        (512, (64, 64, 64, 1, 4, 2, 16, 0, 1)),
        (528, (64, 64, 64, 1, 4, 2, 16, 0, 1)),
        (544, (64, 64, 64, 1, 4, 2, 16, 0, 1)),
        (576, (64, 64, 64, 1, 4, 2, 16, 0, 1)),
        (640, (64, 128, 128, 1, 8, 1, 16, 1, 1)),
        (768, (128, 128, 64, 1, 8, 2, 16, 0, 1)),
        (1024, (64, 128, 128, 1, 8, 1, 16, 0, 1)),
    ),
    "wo": (
        (112, None),
        (128, (64, 128, 128, 1, 8, 2, 16, 1, 1)),
        (144, (64, 128, 128, 1, 8, 2, 16, 1, 1)),
        (160, (64, 128, 128, 1, 8, 2, 16, 0, 1)),
        (176, (64, 128, 128, 1, 8, 2, 16, 0, 1)),
        (192, (64, 128, 128, 1, 8, 2, 16, 0, 1)),
        (224, (64, 64, 64, 1, 4, 2, 16, 0, 1)),
        (256, (64, 64, 64, 1, 4, 2, 16, 0, 1)),
        (272, (64, 128, 128, 1, 4, 1, 16, 2, 1)),
        (288, (64, 64, 64, 1, 4, 2, 16, 0, 1)),
        (320, (64, 64, 64, 1, 4, 2, 16, 0, 1)),
        (352, (128, 128, 64, 1, 8, 2, 16, 0, 1)),
        (384, (128, 128, 64, 1, 8, 2, 16, 0, 1)),
        (448, (64, 64, 128, 1, 4, 1, 16, 0, 1)),
        (512, (64, 64, 64, 1, 4, 2, 16, 0, 1)),
        (528, (64, 64, 64, 1, 4, 2, 16, 0, 1)),
        (544, (64, 64, 64, 1, 4, 2, 16, 0, 1)),
        (576, (64, 64, 64, 1, 4, 2, 16, 0, 1)),
        (640, (128, 128, 64, 1, 8, 2, 16, 0, 1)),
        (768, (128, 128, 64, 1, 8, 2, 16, 0, 1)),
        (1024, (64, 128, 128, 1, 8, 1, 16, 0, 1)),
    ),
    "gate": (
        (112, None),
        (128, (64, 32, 128, 4, 4, 2, 16, 0, 1)),
        (144, (64, 32, 128, 4, 4, 2, 16, 0, 1)),
        (160, (64, 32, 128, 4, 4, 2, 16, 0, 1)),
        (176, (64, 32, 128, 4, 4, 2, 16, 0, 1)),
        (192, (64, 32, 128, 4, 4, 2, 16, 0, 1)),
        (224, (64, 32, 128, 4, 4, 2, 16, 0, 1)),
        (256, (64, 64, 128, 4, 4, 2, 16, 0, 2)),
        (272, (64, 64, 128, 4, 4, 2, 16, 0, 2)),
        (288, (64, 32, 128, 4, 4, 2, 16, 0, 1)),
        (320, (64, 32, 128, 4, 4, 2, 16, 0, 1)),
        (352, (64, 32, 128, 4, 4, 2, 16, 0, 1)),
        (384, (64, 32, 128, 4, 4, 2, 16, 0, 1)),
        (448, (64, 128, 128, 4, 8, 2, 16, 1, 1)),
        (512, (64, 128, 128, 4, 8, 2, 16, 0, 1)),
        (528, (64, 128, 128, 4, 8, 2, 16, 1, 1)),
        (544, (128, 64, 128, 4, 8, 2, 16, 0, 1)),
        (576, (128, 64, 128, 4, 8, 2, 16, 0, 1)),
        (640, (128, 64, 128, 4, 8, 2, 16, 0, 1)),
        (768, (128, 32, 128, 2, 4, 2, 32, 0, 2)),
        (1024, (128, 64, 128, 2, 8, 2, 32, 0, 2)),
    ),
}
_MAX_M = 1024


@triton.jit
def _dec_mm_kernel(
    x_ptr, w_ptr, y_ptr, M, N, K_SPLIT,
    sxm, swn, sys,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, SLAB: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group = GROUP_M * num_n
    first_m = (pid // group) * GROUP_M
    gsize = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % gsize
    pid_n = (pid % group) // gsize
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    k0 = pid_k * K_SPLIT
    rmc = tl.where(rm < M, rm, 0)
    rnc = tl.where(rn < N, rn, 0)
    xp = x_ptr + rmc[:, None] * sxm + (k0 + rk)[None, :]
    wp = w_ptr + rnc[None, :] * swn + (k0 + rk)[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K_SPLIT, BK):
        acc += tl.dot(tl.load(xp), tl.load(wp))
        xp += BK
        wp += BK
    msk = (rm < M)[:, None] & (rn < N)[None, :]
    if SPLIT_K == 1:
        if SLAB > 0:
            # rank-major (N // SLAB, M, SLAB) slabs: (m, n) at
            # [n // SLAB, m, n % SLAB]
            yp = (y_ptr + (rn // SLAB)[None, :] * (M * SLAB) + rm[:, None] * SLAB
                  + (rn % SLAB)[None, :])
        else:
            yp = y_ptr + rm[:, None] * N + rn[None, :]
        tl.store(yp, acc.to(y_ptr.dtype.element_ty), mask=msk)
    else:
        tl.store(y_ptr + pid_k * sys + rm[:, None] * N + rn[None, :], acc, mask=msk)


@triton.jit
def _dec_mm_reduce_kernel(p_ptr, y_ptr, M, N, SPLIT_K: tl.constexpr,
                          SLAB: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mn = M * N
    m = offs < mn
    acc = tl.load(p_ptr + offs, mask=m, other=0.0)
    for s in tl.static_range(1, SPLIT_K):
        acc += tl.load(p_ptr + s * mn + offs, mask=m, other=0.0)
    if SLAB > 0:
        row = offs // N
        col = offs % N
        dst = (col // SLAB) * (M * SLAB) + row * SLAB + col % SLAB
    else:
        dst = offs
    tl.store(y_ptr + dst, acc.to(y_ptr.dtype.element_ty), mask=m)


def _config(op: str, m: int):
    for hi, c in _CFG[op]:
        if m <= hi:
            return c
    return None


def dec_tgemm(
    x: torch.Tensor,
    w: torch.Tensor,
    op: str,
    out_dtype: torch.dtype | None = None,
    slab_tp: int = 0,
) -> torch.Tensor | None:
    """``x @ w.T`` with the tuned Triton tile for ``op``, or None if ``op`` is
    disabled / untuned for this shape (caller runs the served GEMM).

    slab_tp > 0 returns the output rank-major as (slab_tp, M, N // slab_tp).
    """
    if op.split("_")[0] not in ENABLED:
        return None
    if x.dim() != 2 or w.dim() != 2:
        return None
    m, k = x.shape
    n = w.shape[0]
    if not (0 < m <= _MAX_M) or (n, k) != _SHAPES[op] or w.shape[1] != k:
        return None
    if (
        x.dtype != torch.bfloat16
        or w.dtype != torch.bfloat16
        or x.stride(1) != 1
        or w.stride(1) != 1
        or not x.is_cuda
    ):
        return None
    cfg = _config(op, m)
    if cfg is None:
        return None
    bm, bn, bk, sk, nw, ns, nk, wpe, kp = cfg
    if k % sk or (k // sk) % bk:
        return None
    slab = n // slab_tp if slab_tp > 0 else 0
    if slab and (n % slab_tp or slab % bn):
        return None
    odt = out_dtype or x.dtype
    if slab:
        y = torch.empty((slab_tp, m, slab), dtype=odt, device=x.device)
    else:
        y = torch.empty((m, n), dtype=odt, device=x.device)
    k_split = k // sk
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn), sk)
    if sk == 1:
        _dec_mm_kernel[grid](
            x, w, y, m, n, k_split, x.stride(0), w.stride(0), 0,
            BM=bm, BN=bn, BK=bk, GROUP_M=4, SPLIT_K=1, SLAB=slab,
            num_warps=nw, num_stages=ns, matrix_instr_nonkdim=nk,
            waves_per_eu=wpe, kpack=kp,
        )
        return y
    part = torch.empty((sk, m, n), dtype=torch.float32, device=x.device)
    _dec_mm_kernel[grid](
        x, w, part, m, n, k_split, x.stride(0), w.stride(0), m * n,
        BM=bm, BN=bn, BK=bk, GROUP_M=4, SPLIT_K=sk, SLAB=0,
        num_warps=nw, num_stages=ns, matrix_instr_nonkdim=nk,
        waves_per_eu=wpe, kpack=kp,
    )
    _dec_mm_reduce_kernel[(triton.cdiv(m * n, 1024),)](
        part, y, m, n, SPLIT_K=sk, SLAB=slab, BLOCK=1024, num_warps=4
    )
    return y
'''

# ---------------------------------------------------------------- moe.py
MOE_IMPORT_OLD = '''from .ops.norm import add_to_slabs
'''
MOE_IMPORT_NEW = '''from .ops.decode_tgemm import dec_tgemm  # patch rocm_inkling_decode_tgemm.py
from .ops.norm import add_to_slabs
'''

MOE_GATE_OLD = '''    if _gate_splitk_ok(flat, weight):
        return _gate_fp32_splitk(flat, weight).view(*leading, weight.shape[0])
'''
MOE_GATE_NEW = '''    if _gate_splitk_ok(flat, weight):
        return _gate_fp32_splitk(flat, weight).view(*leading, weight.shape[0])
    # (patch rocm_inkling_decode_tgemm.py) tuned Triton tile at decode M.
    out = dec_tgemm(flat, weight, "gate", torch.float32)
    if out is not None:
        return out.view(*leading, weight.shape[0])
'''

MOE_SINK_OLD = '''        raw = x @ self.w13_weight.view(-1, x.shape[-1]).T  # (T, S*2F)
        h = sink_silu_mul_epilogue(
            raw, self._unit, gammas, self._unit, self.n_experts, x.dtype
        )
        return h @ self.w2_weight.T  # (T, D)
'''
MOE_SINK_NEW = '''        w13 = self.w13_weight.view(-1, x.shape[-1])
        # (patch rocm_inkling_decode_tgemm.py) tuned Triton tiles at decode M.
        raw = dec_tgemm(x, w13, "sink_g1")
        if raw is None:
            raw = x @ w13.T  # (T, S*2F)
        h = sink_silu_mul_epilogue(
            raw, self._unit, gammas, self._unit, self.n_experts, x.dtype
        )
        out = dec_tgemm(h, self.w2_weight, "sink_g2")
        return out if out is not None else h @ self.w2_weight.T  # (T, D)
'''

# ---------------------------------------------------------------- attention.py
ATTN_IMPORT_OLD = '''from .ops.qkvr_prep import fused_qkvr_prep
'''
ATTN_IMPORT_NEW = '''from .ops.decode_tgemm import dec_tgemm  # patch rocm_inkling_decode_tgemm.py
from .ops.qkvr_prep import fused_qkvr_prep
'''

ATTN_QKVR_OLD = '''        qkvr, _ = self.qkvr(hidden_states)

        attn_metadata = get_forward_context().attn_metadata
'''
ATTN_QKVR_NEW = '''        # (patch rocm_inkling_decode_tgemm.py) tuned Triton tile at decode M
        # for a bias-free bf16 qkvr; None -> the served linear.
        qkvr = None
        w_qkvr = getattr(self.qkvr, "weight", None)
        if w_qkvr is not None and getattr(self.qkvr, "bias", None) is None:
            qkvr = dec_tgemm(hidden_states, w_qkvr, "qkvr")
        if qkvr is None:
            qkvr, _ = self.qkvr(hidden_states)

        attn_metadata = get_forward_context().attn_metadata
'''

ATTN_WO_OLD = '''            w = self.wo_ud.weight
            tp = get_tensor_model_parallel_world_size()
            return torch.bmm(
'''
ATTN_WO_NEW = '''            w = self.wo_ud.weight
            tp = get_tensor_model_parallel_world_size()
            # (patch rocm_inkling_decode_tgemm.py) the same (tp, T, C) slabs
            # from a tuned Triton tile at decode M.
            slabs = dec_tgemm(flat, w, "wo", slab_tp=tp)
            if slabs is not None:
                return slabs
            return torch.bmm(
'''

TARGETS = [
    (D + "moe.py", [("import", MOE_IMPORT_OLD, MOE_IMPORT_NEW),
                    ("gate", MOE_GATE_OLD, MOE_GATE_NEW),
                    ("sink GEMMs", MOE_SINK_OLD, MOE_SINK_NEW)]),
    (D + "attention.py", [("import", ATTN_IMPORT_OLD, ATTN_IMPORT_NEW),
                          ("qkvr", ATTN_QKVR_OLD, ATTN_QKVR_NEW),
                          ("wo slabs", ATTN_WO_OLD, ATTN_WO_NEW)]),
]


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
    ap.add_argument("--revert", action="store_true",
                    help="undo the patch")
    ap.add_argument("--root", help="patch the vllm/ tree under this dir "
                    "instead of the installed vLLM's")
    args = ap.parse_args()

    root = args.root or find_root()
    plans = []
    problems = []
    for rel, sites in TARGETS:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            problems.append(f"{path} missing; vLLM layout changed")
            continue
        with open(path) as fh:
            src = fh.read()
        if (MARKER in src) != args.revert:
            print(f"already {'reverted' if args.revert else 'patched'}: {path}")
            continue
        patched = src
        for name, old, new in sites:
            a, b = (new, old) if args.revert else (old, new)
            n = patched.count(a)
            if n != 1:
                problems.append(f"{rel}: {name} (matched {n}x, expected 1)")
                continue
            patched = patched.replace(a, b, 1)
        plans.append((path, patched))
    if not os.path.isdir(os.path.join(root, os.path.dirname(NEW_FILE))):
        problems.append(f"{os.path.dirname(NEW_FILE)} missing; vLLM layout changed")

    if problems:
        print("ERROR: patch target not found:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print("Inkling's moe.py / attention.py changed (are "
              "inkling_gate_fp32_splitk.py and inkling_sconv_zero_copy.py "
              "applied first?); re-derive or retire this patch",
              file=sys.stderr)
        return 1

    new_path = os.path.join(root, NEW_FILE)
    if args.check:
        for path, _ in plans:
            print(f"would {'revert' if args.revert else 'patch'}: {path}")
        print(f"would {'remove' if args.revert else 'write'}: {new_path}")
        return 0
    if args.revert:
        if os.path.exists(new_path):
            os.remove(new_path)
    else:
        with open(new_path, "w") as fh:
            fh.write(MODULE_SRC)
    for path, patched in plans:
        with open(path, "w") as fh:
            fh.write(patched)
        print(f"{'reverted' if args.revert else 'patched'}: {path}  "
              "(Inkling decode Triton GEMMs, INKLING_DEC_TGEMM)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
