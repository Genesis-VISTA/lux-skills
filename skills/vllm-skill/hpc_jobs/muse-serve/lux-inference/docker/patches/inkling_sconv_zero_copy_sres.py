#!/usr/bin/env python3
"""
Keep Inkling's residual stream channel-sharded across the sconv RS/AG sites
for large T, so the per-site add + rmsnorm stops reading and writing the full
replicated (T, hidden) residual. Apply after inkling_sconv_zero_copy_ca.py
(alphabetical in the container patch loop).

THE COST AS IT PRESENTS

Every _sconv_add_norm site (42 per stage pass) ends with add_rmsnorm_slabs on
the full (T, 4096) row on all 8 TP ranks: read the residual + the AG'd sconv
output, write the new residual + the normed y, 4 x T x 8 KiB. At an 8k prefill
chunk that is 207 us per site, 8.7 ms per stage-0 chunk (job f61reugSbS8f),
near HBM bandwidth, replicated on every rank.

THE FIX

The residual only feeds the next site's add and rmsnorm, so each rank keeps
only its own channel shard [r*C, (r+1)*C) of it (C = hidden / tp, the sconv
shard). Per site:
  RS (unchanged) -> sconv on the shard with the residual add fused into the
  sconv kernel's epilogue (s = residual_shard + sconv_out, rounded exactly like
  add_rmsnorm_slabs) -> AG of s (same bytes as before) -> rmsnorm_slabs reads
  the AG'd rank-major slabs, writes y.
The new residual shard is s itself. The full-width pass now reads 1 and writes
1 (T, hidden) tensor instead of 2 + 2; the shard add costs 1/tp of a pass.

InklingModel.forward switches to the sharded residual when T >=
INKLING_SCONV_SRES_MIN_T (default 1 = always; 0 = off): it computes the first
layer's attn_norm on the full row first (as the layer would), then slices the
rank's channel shard. A norm-less site (PP-stage end) returns the full
residual row-major. The MTP layer always keeps the full residual.

Measured (TP8, same node, served env): stage 0 with 21 real layers, eager
prefill fwd 8k chunk 223.2 -> 219.4 ms, 16k 421.0 -> 412.9 ms (job
f61we1oupAbq); 22-layer graphs decode step T=256 22.01 -> 21.78, T=512
30.08 -> 29.83, T=1024 46.07 -> 45.58 ms, mixed 4x4096 step 433.7 -> 426.4 ms
(job f61wxbHqi6Nb).

Numerics: bit-identical. The add is the same fp32 add + bf16 rounding of the
same operands (sconv output rounded to bf16 first), and rmsnorm_slabs is
add_rmsnorm_slabs's kernel without the residual load, so it reduces the same
row in the same layout.
"""
import argparse
import importlib.util
import os
import sys

MARKER = "patch inkling_sconv_zero_copy_sres.py"

# ------------------------------------------------------------- ops/sconv.py
SCONV_SIG_OLD = '''    MAX_BLOCKS,
    N,  # block_size
    W: tl.constexpr,
    USE_SILU: tl.constexpr,
    USE_RESIDUAL: tl.constexpr,
'''

SCONV_SIG_NEW = '''    MAX_BLOCKS,
    N,  # block_size
    res_ptr,  # [T, H*WS] addend (ADD_RES only; patch inkling_sconv_zero_copy_sres.py)
    stride_r_t,
    W: tl.constexpr,
    USE_SILU: tl.constexpr,
    USE_RESIDUAL: tl.constexpr,
    ADD_RES: tl.constexpr,
'''

SCONV_EPI_OLD = '''    if USE_RESIDUAL:
        acc += xv.to(tl.float32)

    tl.store(
'''

SCONV_EPI_NEW = '''    if USE_RESIDUAL:
        acc += xv.to(tl.float32)
    if ADD_RES:
        # (patch inkling_sconv_zero_copy_sres.py) out = res + bf16(conv), with
        # the rounding of add_rmsnorm_slabs (`(r + d).to(dtype)`).
        acc = acc.to(out_ptr.dtype.element_ty).to(tl.float32)
        rv = tl.load(
            res_ptr + toff[:, None] * stride_r_t + coff[None, :],
            mask=tc_mask,
            other=0.0,
        ).to(tl.float32)
        acc = rv + acc

    tl.store(
'''

SCONV_WRAP_SIG_OLD = '''    activation: str | None = None,
    use_residual: bool = True,
) -> torch.Tensor:
    """Single-launch insert + depthwise causal conv1d over the paged cache.
'''

SCONV_WRAP_SIG_NEW = '''    activation: str | None = None,
    use_residual: bool = True,
    add_res: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single-launch insert + depthwise causal conv1d over the paged cache.

    ``add_res`` ([T, H*ws], unit channel stride): return ``add_res + out``
    (out rounded to the output dtype first) instead of ``out``.
'''

SCONV_WRAP_CALL_OLD = '''        block_table.shape[1],
        block_size,
        W=W,
        USE_SILU=activation in ("silu", "swish"),
        USE_RESIDUAL=use_residual,
'''

SCONV_WRAP_CALL_NEW = '''        block_table.shape[1],
        block_size,
        add_res if add_res is not None else x,
        add_res.stride(0) if add_res is not None else 0,
        W=W,
        USE_SILU=activation in ("silu", "swish"),
        USE_RESIDUAL=use_residual,
        ADD_RES=add_res is not None,
'''

SCONV_WRAP_ASSERT_OLD = '''    assert x.is_contiguous()
    assert cache.stride(3) == 1, "cache D-dim must be contiguous"
'''

SCONV_WRAP_ASSERT_NEW = '''    assert x.is_contiguous()
    assert cache.stride(3) == 1, "cache D-dim must be contiguous"
    if add_res is not None:
        assert add_res.shape == x.shape and add_res.stride(1) == 1
'''

# ------------------------------------------------------------- short_conv.py
SC_FWD_OLD = '''    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: (num_tokens, dim); positions: (num_tokens,) absolute positions.
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            # Memory-profiling / no metadata: identity (residual).
            return x
        m = attn_metadata.get(self.owner.prefix)
        if m is None:
            return x
        assert isinstance(m, InklingSconvMetadata)
        cache = self.owner.kv_cache
        if cache.numel() == 0:
            # Cache not yet bound (profiling before KV alloc): identity.
            return x
'''

SC_FWD_NEW = '''    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        add_res: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (num_tokens, dim); positions: (num_tokens,) absolute positions.
        # (patch inkling_sconv_zero_copy_sres.py) add_res: return add_res + out.
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            # Memory-profiling / no metadata: identity (residual).
            return x if add_res is None else add_res + x
        m = attn_metadata.get(self.owner.prefix)
        if m is None:
            return x if add_res is None else add_res + x
        assert isinstance(m, InklingSconvMetadata)
        cache = self.owner.kv_cache
        if cache.numel() == 0:
            # Cache not yet bound (profiling before KV alloc): identity.
            return x if add_res is None else add_res + x
'''

SC_CALL_OLD = '''            activation=None,
            use_residual=True,
        )
'''

SC_CALL_NEW = '''            activation=None,
            use_residual=True,
            add_res=add_res,
        )
'''

# ------------------------------------------------------------- ops/norm.py
NORM_OLD = '''@triton.jit
def _add_to_slabs_kernel(
'''

NORM_NEW = '''# (patch inkling_sconv_zero_copy_sres.py) add_rmsnorm_slabs's kernel without
# the residual: the slabs already hold the summed residual. Same row layout
# and reduction, so y is bit-identical to add_rmsnorm_slabs on (res, delta).
@triton.jit
def _rmsnorm_slabs_fwd_kernel(
    slab_ptr,  # [tp, T, C] rank-major input (read)
    weight_ptr,  # [N]
    y_ptr,  # [T, N] normed output
    eps,
    slab_stride_0,
    slab_stride_1,
    y_stride_0,
    n_cols,
    seg: tl.constexpr,
    block_size_n: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    offs_n = tl.arange(0, block_size_n)
    mask_n = offs_n < n_cols
    d_offs = (
        (offs_n // seg).to(tl.int64) * slab_stride_0
        + pid_m * slab_stride_1
        + offs_n % seg
    )
    s = tl.load(slab_ptr + d_offs, mask=mask_n, other=0.0)
    weight = tl.load(weight_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    x = s.to(tl.float32)
    row_var = tl.sum(x * x, axis=0) / n_cols
    rstd = tl.math.rsqrt(row_var + eps)
    y = x * rstd * weight
    tl.store(y_ptr + pid_m * y_stride_0 + offs_n, y, mask=mask_n)


def rmsnorm_slabs(
    slabs: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """rmsnorm of a ``[T, N]`` tensor held rank-major as ``[tp, T, C]`` slabs;
    returns a fresh row-major ``[T, N]``."""
    assert slabs.ndim == 3 and slabs.stride(2) == 1, slabs.shape
    tp, n_rows, seg = slabs.shape
    n_cols = tp * seg
    assert weight.shape[0] == n_cols
    y = torch.empty((n_rows, n_cols), dtype=slabs.dtype, device=slabs.device)
    if n_rows == 0:
        return y
    block_size_n = triton.next_power_of_2(n_cols)
    max_block_size_n = _MAX_FUSED_SIZE // slabs.element_size()
    if max_block_size_n < block_size_n:
        raise RuntimeError(f"Large {n_cols=} is not supported")
    num_warps = _get_num_warps_from_block_size(block_size_n)
    _rmsnorm_slabs_fwd_kernel[(n_rows,)](
        slabs,
        weight,
        y,
        eps,
        slabs.stride(0),
        slabs.stride(1),
        y.stride(0),
        n_cols,
        seg,
        block_size_n,
        num_warps=num_warps,
    )
    return y


@triton.jit
def _add_to_slabs_kernel(
'''

# ------------------------------------------------------------- model.py
MODEL_IMPORT_OLD = '''from .ops.norm import add_rmsnorm, add_rmsnorm_slabs, embed_rmsnorm
'''

MODEL_IMPORT_NEW = '''from .ops.norm import add_rmsnorm, add_rmsnorm_slabs, embed_rmsnorm, rmsnorm_slabs
'''

MODEL_FLAG_OLD = '''_CA_RSAG_MAX_T = int(__import__("os").environ.get("INKLING_SCONV_CA_RSAG_MAX_T", "0"))
'''

MODEL_FLAG_NEW = '''_CA_RSAG_MAX_T = int(__import__("os").environ.get("INKLING_SCONV_CA_RSAG_MAX_T", "0"))

# (patch inkling_sconv_zero_copy_sres.py) channel-sharded residual for
# forwards of at least this many tokens; 0 = off.
_SRES_MIN_T = int(__import__("os").environ.get("INKLING_SCONV_SRES_MIN_T", "1"))
'''

HELPER_OLD = '''    tp: int,
    t: int,
    c: int,
) -> torch.Tensor:
    """RS(dim 0) of contiguous rank-major ``[tp * t, c]`` slabs -> sconv on the
    ``[t, c]`` shard -> AG(dim 0); returns ``[tp * t, c]``. Small T goes
    through the custom (IPC) RS/AG kernels when the TP group has them."""
'''

HELPER_NEW = '''    tp: int,
    t: int,
    c: int,
    add_res: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RS(dim 0) of contiguous rank-major ``[tp * t, c]`` slabs -> sconv on the
    ``[t, c]`` shard -> AG(dim 0); returns ``([tp * t, c], [t, c] shard)``.
    Small T goes through the custom (IPC) RS/AG kernels when the TP group has
    them. (patch inkling_sconv_zero_copy_sres.py) ``add_res`` (``[t, c]``) is
    added to the sconv output before the AG."""
'''

HELPER_CA_OLD = '''            shard = sconv(shard.view(t, c), positions).contiguous()
'''

HELPER_CA_NEW = '''            shard = sconv(shard.view(t, c), positions, add_res).contiguous()
'''

HELPER_CA_RET_OLD = '''            return full.view(tp * t, c)
    shard = tensor_model_parallel_reduce_scatter(slabs, dim=0)
    shard = sconv(shard, positions)
    return tensor_model_parallel_all_gather(shard, dim=0)
'''

HELPER_CA_RET_NEW = '''            return full.view(tp * t, c), shard
    shard = tensor_model_parallel_reduce_scatter(slabs, dim=0)
    shard = sconv(shard, positions, add_res)
    return tensor_model_parallel_all_gather(shard, dim=0), shard
'''

SITE_OLD = '''    tp = get_tensor_model_parallel_world_size()
    if tp > 1 and delta.dim() == 3:
        # (patch inkling_sconv_zero_copy.py) The sublayer already wrote its
        # partial output rank-major (tp, T, C): no copy on either side.
        _, t, c = delta.shape
        # (patch inkling_sconv_zero_copy_ca.py) small T: custom RS/AG kernels.
        full = _slab_rs_sconv_ag(delta.view(tp * t, c), sconv, positions, tp, t, c)
        return add_rmsnorm_slabs(hidden, full.view(tp, t, c), norm_w, eps)
    if tp > 1 and delta.dim() == 2:
        t, h = delta.shape
        c = h // tp
        slabs = delta.view(t, tp, c).transpose(0, 1).reshape(tp * t, c)
        full = _slab_rs_sconv_ag(slabs, sconv, positions, tp, t, c)
'''

SITE_NEW = '''    tp = get_tensor_model_parallel_world_size()
    width = delta.shape[0] * delta.shape[2] if delta.dim() == 3 else delta.shape[-1]
    if tp > 1 and hidden.shape[-1] * tp == width:
        # (patch inkling_sconv_zero_copy_sres.py) hidden is this rank's
        # [T, C] channel shard of the residual: add it to the sconv shard
        # before the AG; the AG'd slabs are the new full residual.
        t, c = hidden.shape
        if delta.dim() == 3:
            slabs = delta.view(tp * t, c)
        else:
            slabs = delta.view(t, tp, c).transpose(0, 1).reshape(tp * t, c)
        full, shard = _slab_rs_sconv_ag(slabs, sconv, positions, tp, t, c, hidden)
        full = full.view(tp, t, c)
        if norm_w is None:
            return None, full.transpose(0, 1).reshape(t, tp * c)
        return rmsnorm_slabs(full, norm_w, eps), shard
    if tp > 1 and delta.dim() == 3:
        # (patch inkling_sconv_zero_copy.py) The sublayer already wrote its
        # partial output rank-major (tp, T, C): no copy on either side.
        _, t, c = delta.shape
        # (patch inkling_sconv_zero_copy_ca.py) small T: custom RS/AG kernels.
        full = _slab_rs_sconv_ag(delta.view(tp * t, c), sconv, positions, tp, t, c)[0]
        return add_rmsnorm_slabs(hidden, full.view(tp, t, c), norm_w, eps)
    if tp > 1 and delta.dim() == 2:
        t, h = delta.shape
        c = h // tp
        slabs = delta.view(t, tp, c).transpose(0, 1).reshape(tp * t, c)
        full = _slab_rs_sconv_ag(slabs, sconv, positions, tp, t, c)[0]
'''

FWD_OLD = '''        pending: tuple[torch.Tensor | None, InklingShortConv] | None = None
        for layer in self.layers[self.start_layer : self.end_layer]:
'''

FWD_NEW = '''        # (patch inkling_sconv_zero_copy_sres.py) Large T: carry only this
        # rank's channel shard of the residual between the sconv sites.
        tp = get_tensor_model_parallel_world_size()
        if (
            tp > 1
            and 0 < _SRES_MIN_T <= hidden_states.shape[0]
            and self.start_layer < self.end_layer
            and hidden_states.shape[-1] % tp == 0
            and hidden_states.stride(-1) == 1
        ):
            if attn_in0 is None:
                attn_in0 = self.layers[self.start_layer].attn_norm(hidden_states)
            c = hidden_states.shape[-1] // tp
            r = get_tensor_model_parallel_rank()
            hidden_states = hidden_states[:, r * c : (r + 1) * c]

        pending: tuple[torch.Tensor | None, InklingShortConv] | None = None
        for layer in self.layers[self.start_layer : self.end_layer]:
'''

D = "vllm/models/inkling/amd/"
TARGETS = [
    (D + "ops/sconv.py", [("kernel signature", SCONV_SIG_OLD, SCONV_SIG_NEW),
                          ("kernel epilogue", SCONV_EPI_OLD, SCONV_EPI_NEW),
                          ("wrapper signature", SCONV_WRAP_SIG_OLD, SCONV_WRAP_SIG_NEW),
                          ("wrapper assert", SCONV_WRAP_ASSERT_OLD, SCONV_WRAP_ASSERT_NEW),
                          ("wrapper launch", SCONV_WRAP_CALL_OLD, SCONV_WRAP_CALL_NEW)]),
    (D + "short_conv.py", [("forward", SC_FWD_OLD, SC_FWD_NEW),
                           ("fused_sconv call", SC_CALL_OLD, SC_CALL_NEW)]),
    (D + "ops/norm.py", [("rmsnorm_slabs", NORM_OLD, NORM_NEW)]),
    (D + "model.py", [("import", MODEL_IMPORT_OLD, MODEL_IMPORT_NEW),
                      ("flag", MODEL_FLAG_OLD, MODEL_FLAG_NEW),
                      ("_slab_rs_sconv_ag signature", HELPER_OLD, HELPER_NEW),
                      ("_slab_rs_sconv_ag CA sconv", HELPER_CA_OLD, HELPER_CA_NEW),
                      ("_slab_rs_sconv_ag returns", HELPER_CA_RET_OLD, HELPER_CA_RET_NEW),
                      ("_sconv_add_norm sharded", SITE_OLD, SITE_NEW),
                      ("InklingModel.forward", FWD_OLD, FWD_NEW)]),
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
        print("Inkling's sconv slab path changed (are inkling_sconv_zero_copy"
              ".py and inkling_sconv_zero_copy_ca.py applied first?); "
              "re-derive or retire this patch", file=sys.stderr)
        return 1

    for path, patched in plans:
        if args.check:
            print(f"would patch: {path}")
            continue
        with open(path, "w") as fh:
            fh.write(patched)
        print(f"patched: {path}  (Inkling channel-sharded residual, "
              f"INKLING_SCONV_SRES_MIN_T)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
