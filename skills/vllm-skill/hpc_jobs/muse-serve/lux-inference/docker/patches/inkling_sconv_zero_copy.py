#!/usr/bin/env python3
"""
Drop the row-block copies that inkling_sconv_rsag_blocks.py leaves around
Inkling's sconv reduce-scatter / all-gather. Apply after that patch.

THE COST AS IT PRESENTS

After rsag_blocks, each _sconv_add_norm site (42 per stage pass) still does two
(T, 4096) bf16 copies: delta (T, tp, C) -> rank-major slabs (tp, T, C) before
the RS, and the AG output (tp, T, C) -> (T, hidden) before add_rmsnorm. In the
car2 microbench (f61kom7jidEw) they cost 55 us per site at T=512 and 370 us at
T=8192 over flat RCCL RS+AG: 2.3 ms per c256 decode step, 15.6 ms per 8k
prefill chunk.

THE FIX

- Post-AG: add_rmsnorm_slabs reads the AG output in its (tp, T, C) layout
  directly (element (t, r*C + c) sits at slabs[r, t, c]), so the residual add
  (+ rmsnorm) needs no permute. All sites, including the norm-less PP / MTP
  tails.
- Pre-RS, MoE layers: InklingMoE already ends with out.add_(sink_out), a full
  read + write of the TP-partial output. add_to_slabs does that add and writes
  the result rank-major (tp, T, C), which _sconv_add_norm recognises by its
  3-D shape and reduce-scatters without a copy.
- Pre-RS, attention (opt-in, INKLING_SCONV_ZC_ATTN_MIN_T=<T>, default 0 = off):
  for T >= that threshold, wo_ud runs as a batched GEMM over the tp output
  slabs, bmm(flat.expand(tp, T, K), W.view(tp, C, K).transpose(1, 2)), which
  writes (tp, T, C) directly. Only for an unquantized, bias-free wo_ud.

Numerics: the add / add+rmsnorm kernels do the same fp32 math and rounding as
the copy path, so with the attention bmm off the result is bit-identical to
rsag_blocks. The bmm may pick a different GEMM kernel (rounding-level change).

INKLING_SCONV_ZC=0 restores the rsag_blocks behaviour (strict 0/1).
"""
import argparse
import importlib.util
import os
import sys

MARKER = "patch inkling_sconv_zero_copy.py"

# ---------------------------------------------------------------- ops/norm.py
NORM_OLD = '''@triton.jit
def _embed_rmsnorm_kernel(
'''

NORM_NEW = '''# (patch inkling_sconv_zero_copy.py) Rank-major ("slab") variants for the
# sconv RS/AG path: a [T, N] tensor held as [tp, T, C] slabs, N = tp * C, where
# element (t, r*C + c) sits at slabs[r, t, c].
@triton.jit
def _add_rmsnorm_slabs_fwd_kernel(
    res_ptr,  # [T, N] residual (read)
    slab_ptr,  # [tp, T, C] rank-major delta (read)
    weight_ptr,  # [N] (HAS_NORM only)
    y_ptr,  # [T, N] normed output (HAS_NORM only)
    res_out_ptr,  # [T, N] updated residual output
    eps,
    res_stride_0,
    slab_stride_0,
    slab_stride_1,
    y_stride_0,
    res_out_stride_0,
    n_cols,
    seg: tl.constexpr,
    block_size_n: tl.constexpr,
    HAS_NORM: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    offs_n = tl.arange(0, block_size_n)
    mask_n = offs_n < n_cols
    d_offs = (
        (offs_n // seg).to(tl.int64) * slab_stride_0
        + pid_m * slab_stride_1
        + offs_n % seg
    )
    r = tl.load(res_ptr + pid_m * res_stride_0 + offs_n, mask=mask_n, other=0.0).to(
        tl.float32
    )
    d = tl.load(slab_ptr + d_offs, mask=mask_n, other=0.0).to(tl.float32)
    # Same rounding as _add_rmsnorm_fwd_kernel / eager `residual + delta`.
    s = (r + d).to(res_out_ptr.dtype.element_ty)
    tl.store(res_out_ptr + pid_m * res_out_stride_0 + offs_n, s, mask=mask_n)
    if HAS_NORM:
        weight = tl.load(weight_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        x = s.to(tl.float32)
        row_var = tl.sum(x * x, axis=0) / n_cols
        rstd = tl.math.rsqrt(row_var + eps)
        y = x * rstd * weight
        tl.store(y_ptr + pid_m * y_stride_0 + offs_n, y, mask=mask_n)


def add_rmsnorm_slabs(
    residual: torch.Tensor,
    slabs: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """``add_rmsnorm`` with the delta given rank-major as ``[tp, T, C]`` slabs
    (the layout a dim-0 all-gather of channel shards returns).

    Returns ``(y, res)``; ``y`` is None when ``weight`` is None (plain
    residual add). Both are fresh tensors.
    """
    assert residual.ndim == 2 and slabs.ndim == 3, (residual.shape, slabs.shape)
    n_rows, n_cols = residual.shape
    tp, t, seg = slabs.shape
    assert t == n_rows and tp * seg == n_cols, (residual.shape, slabs.shape)
    assert slabs.stride(2) == 1 and residual.stride(1) == 1
    has_norm = weight is not None
    res_out = torch.empty_like(residual)
    y = torch.empty_like(residual) if has_norm else None
    if n_rows == 0:
        return y, res_out

    block_size_n = triton.next_power_of_2(n_cols)
    max_block_size_n = _MAX_FUSED_SIZE // residual.element_size()
    if max_block_size_n < block_size_n:
        raise RuntimeError(f"Large {n_cols=} is not supported")
    num_warps = _get_num_warps_from_block_size(block_size_n)
    y_arg = y if has_norm else res_out
    _add_rmsnorm_slabs_fwd_kernel[(n_rows,)](
        residual,
        slabs,
        weight if has_norm else residual,
        y_arg,
        res_out,
        eps,
        residual.stride(0),
        slabs.stride(0),
        slabs.stride(1),
        y_arg.stride(0),
        res_out.stride(0),
        n_cols,
        seg,
        block_size_n,
        has_norm,
        num_warps=num_warps,
    )
    return y, res_out


@triton.jit
def _add_to_slabs_kernel(
    a_ptr,  # [T, N]
    b_ptr,  # [T, N]
    out_ptr,  # [tp, T, C] rank-major a + b
    a_stride_0,
    b_stride_0,
    out_stride_0,
    out_stride_1,
    n_cols,
    seg: tl.constexpr,
    block_size_n: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    offs_n = tl.arange(0, block_size_n)
    mask_n = offs_n < n_cols
    a = tl.load(a_ptr + pid_m * a_stride_0 + offs_n, mask=mask_n, other=0.0)
    b = tl.load(b_ptr + pid_m * b_stride_0 + offs_n, mask=mask_n, other=0.0)
    s = (a.to(tl.float32) + b.to(tl.float32)).to(out_ptr.dtype.element_ty)
    o_offs = (
        (offs_n // seg).to(tl.int64) * out_stride_0
        + pid_m * out_stride_1
        + offs_n % seg
    )
    tl.store(out_ptr + o_offs, s, mask=mask_n)


def add_to_slabs(a: torch.Tensor, b: torch.Tensor, tp: int) -> torch.Tensor:
    """``a + b`` (rounded to ``a.dtype``, like ``a.add_(b)``) for ``[T, N]``
    inputs, returned rank-major as fresh ``[tp, T, N // tp]`` slabs."""
    assert a.ndim == 2 and a.shape == b.shape, (a.shape, b.shape)
    assert a.stride(1) == 1 and b.stride(1) == 1
    n_rows, n_cols = a.shape
    assert n_cols % tp == 0, (n_cols, tp)
    seg = n_cols // tp
    out = torch.empty((tp, n_rows, seg), dtype=a.dtype, device=a.device)
    if n_rows == 0:
        return out
    block_size_n = triton.next_power_of_2(n_cols)
    num_warps = _get_num_warps_from_block_size(block_size_n)
    _add_to_slabs_kernel[(n_rows,)](
        a,
        b,
        out,
        a.stride(0),
        b.stride(0),
        out.stride(0),
        out.stride(1),
        n_cols,
        seg,
        block_size_n,
        num_warps=num_warps,
    )
    return out


@triton.jit
def _embed_rmsnorm_kernel(
'''

# --------------------------------------------------------------------- moe.py
MOE_OLD = '''        self._routed_sel = None

        return out.add_(sink_out)
'''

MOE_NEW = '''        self._routed_sel = None

        # (patch inkling_sconv_zero_copy.py) Fold the sink add into a
        # rank-major (tp, T, C) write so _sconv_add_norm can reduce-scatter
        # it without a permute copy.
        tp = get_tensor_model_parallel_world_size()
        if (
            _SCONV_ZC
            and tp > 1
            and out.dim() == 2
            and out.shape == sink_out.shape
            and out.shape[1] % tp == 0
            and out.stride(1) == 1
            and sink_out.stride(1) == 1
        ):
            return add_to_slabs(out, sink_out, tp)
        return out.add_(sink_out)
'''

MOE_IMPORT_OLD = '''from ..configs import InklingModelConfig
'''

MOE_IMPORT_NEW = '''from ..configs import InklingModelConfig
from .ops.norm import add_to_slabs

# (patch inkling_sconv_zero_copy.py)
_SCONV_ZC_RAW = __import__("os").environ.get("INKLING_SCONV_ZC", "1")
if _SCONV_ZC_RAW not in ("0", "1"):
    raise ValueError(f"INKLING_SCONV_ZC must be 0 or 1, got {_SCONV_ZC_RAW!r}")
_SCONV_ZC = _SCONV_ZC_RAW == "1"
'''

# ------------------------------------------------------------------- model.py
MODEL_IMPORT_OLD = '''from .ops.norm import add_rmsnorm, embed_rmsnorm
'''

MODEL_IMPORT_NEW = '''from .ops.norm import add_rmsnorm, add_rmsnorm_slabs, embed_rmsnorm
'''

MODEL_OLD = '''    tp = get_tensor_model_parallel_world_size()
    if tp > 1 and delta.dim() == 2:
        t, h = delta.shape
        c = h // tp
        slabs = delta.view(t, tp, c).transpose(0, 1).reshape(tp * t, c)
        shard = tensor_model_parallel_reduce_scatter(slabs, dim=0)
        shard = sconv(shard, positions)
        full = tensor_model_parallel_all_gather(shard, dim=0)
        full = full.view(tp, t, c).transpose(0, 1).reshape(t, h)
    else:
'''

MODEL_NEW = '''    tp = get_tensor_model_parallel_world_size()
    if tp > 1 and delta.dim() == 3:
        # (patch inkling_sconv_zero_copy.py) The sublayer already wrote its
        # partial output rank-major (tp, T, C): no copy on either side.
        _, t, c = delta.shape
        shard = tensor_model_parallel_reduce_scatter(delta.view(tp * t, c), dim=0)
        shard = sconv(shard, positions)
        full = tensor_model_parallel_all_gather(shard, dim=0)
        return add_rmsnorm_slabs(hidden, full.view(tp, t, c), norm_w, eps)
    if tp > 1 and delta.dim() == 2:
        t, h = delta.shape
        c = h // tp
        slabs = delta.view(t, tp, c).transpose(0, 1).reshape(tp * t, c)
        shard = tensor_model_parallel_reduce_scatter(slabs, dim=0)
        shard = sconv(shard, positions)
        full = tensor_model_parallel_all_gather(shard, dim=0)
        if _SCONV_ZC:
            # (patch inkling_sconv_zero_copy.py) Read the AG slabs in place.
            return add_rmsnorm_slabs(hidden, full.view(tp, t, c), norm_w, eps)
        full = full.view(tp, t, c).transpose(0, 1).reshape(t, h)
    else:
'''

MODEL_FLAG_OLD = '''def _sconv_add_norm(
'''

MODEL_FLAG_NEW = '''# (patch inkling_sconv_zero_copy.py)
_SCONV_ZC_RAW = __import__("os").environ.get("INKLING_SCONV_ZC", "1")
if _SCONV_ZC_RAW not in ("0", "1"):
    raise ValueError(f"INKLING_SCONV_ZC must be 0 or 1, got {_SCONV_ZC_RAW!r}")
_SCONV_ZC = _SCONV_ZC_RAW == "1"


def _sconv_add_norm(
'''

# --------------------------------------------------------------- attention.py
ATTN_OLD = '''        flat = attn_output.view(num_tokens, -1)
        output, _ = self.wo_ud(flat)
        return output
'''

ATTN_NEW = '''        flat = attn_output.view(num_tokens, -1)
        # (patch inkling_sconv_zero_copy.py) Large T: emit the TP-partial
        # output rank-major (tp, T, C) as one batched GEMM over the output
        # slabs, so the sconv reduce-scatter needs no permute copy.
        if 0 < _ZC_ATTN_MIN_T <= num_tokens and self._zc_slab_ok():
            w = self.wo_ud.weight
            tp = get_tensor_model_parallel_world_size()
            return torch.bmm(
                flat.unsqueeze(0).expand(tp, -1, -1),
                w.view(tp, w.shape[0] // tp, w.shape[1]).transpose(1, 2),
            )
        output, _ = self.wo_ud(flat)
        return output

    def _zc_slab_ok(self) -> bool:
        ok = getattr(self, "_zc_ok", None)
        if ok is None:
            w = getattr(self.wo_ud, "weight", None)
            tp = get_tensor_model_parallel_world_size()
            ok = (
                tp > 1
                and w is not None
                and w.dim() == 2
                and w.dtype in (torch.bfloat16, torch.float16)
                and w.is_contiguous()
                and w.shape[0] % tp == 0
                and getattr(self.wo_ud, "bias", None) is None
                and type(self.wo_ud.quant_method).__name__
                == "UnquantizedLinearMethod"
            )
            self._zc_ok = ok
        return ok
'''

ATTN_IMPORT_OLD = '''from ..configs import InklingModelConfig
'''

ATTN_IMPORT_NEW = '''from ..configs import InklingModelConfig

# (patch inkling_sconv_zero_copy.py) 0 = off.
_ZC_ATTN_MIN_T = int(__import__("os").environ.get("INKLING_SCONV_ZC_ATTN_MIN_T", "0"))
if __import__("os").environ.get("INKLING_SCONV_ZC", "1") == "0":
    _ZC_ATTN_MIN_T = 0
'''

D = "vllm/models/inkling/amd/"
TARGETS = [
    (D + "ops/norm.py", [("slab kernels", NORM_OLD, NORM_NEW)]),
    (D + "moe.py", [("flag/import", MOE_IMPORT_OLD, MOE_IMPORT_NEW),
                    ("sink add -> slabs", MOE_OLD, MOE_NEW)]),
    (D + "model.py", [("import", MODEL_IMPORT_OLD, MODEL_IMPORT_NEW),
                      ("flag", MODEL_FLAG_OLD, MODEL_FLAG_NEW),
                      ("_sconv_add_norm slabs", MODEL_OLD, MODEL_NEW)]),
    (D + "attention.py", [("flag", ATTN_IMPORT_OLD, ATTN_IMPORT_NEW),
                          ("wo_ud slabs", ATTN_OLD, ATTN_NEW)]),
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
        if MARKER in src:
            print(f"already patched: {path}")
            continue
        patched = src
        for name, old, new in sites:
            n = patched.count(old)
            if n != 1:
                problems.append(f"{rel}: {name} (matched {n}x, expected 1)")
                continue
            patched = patched.replace(old, new, 1)
        plans.append((path, patched))

    if problems:
        print("ERROR: patch target not found:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print("Inkling's sconv RS/AG path changed (is inkling_sconv_rsag_blocks"
              ".py applied first?); re-derive or retire this patch",
              file=sys.stderr)
        return 1

    for path, patched in plans:
        if args.check:
            print(f"would patch: {path}")
            continue
        with open(path, "w") as fh:
            fh.write(patched)
        print(f"patched: {path}  (Inkling sconv RS/AG zero-copy slabs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
