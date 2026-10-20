#!/usr/bin/env python3
"""
IPC reduce-scatter / all-gather for Inkling's per-site sconv chain at decode T
(TP8, gfx90a).

THE COST AS IT PRESENTS

Each attention/MLP site runs RS([8T, 512] bf16 slabs) -> shard sconv -> AG;
44 sites per stage-0 pass. At T <= 256 (INKLING_SCONV_CA_RSAG_MAX_T) this is
the image's custom RS/AG ops, each preceded by a hipMemcpy of its input into
the IPC buffer. Graph replay,
TP8 (jobs f61yHZPKk82s / f61yQcArGTqq), us per site RS+AG:
  T=64 37.7, 128 67.4, 192 91.3, 256 134.3   (RCCL above: 384 147, 512 165)
Eager decode profile at T=256 (f61y7opCT1iP): 6.4 ms of a 23.2 ms stage-0
kernel sum, plus 0.32 ms of the copies.

THE FIX

ops/xrsag.py: the same algorithm (8-rank per-block flag barrier, 16-byte peer
loads, fp32 sum in rank order 0..7, RNE to bf16, input copied into the IPC
buffer first) with no trailing barrier (RS and AG alternate on these buffers,
so each op's start barrier already proves every peer finished the previous
op's peer reads) and 8 blocks of 512 threads. Built with torch's load_inline
at patch time (ops/_xrsag_build, ~1 min) and loaded by each worker; buffers
are allocated over the TP group on the first forward (the eager profile run).
Microbench, graph replay, us per site RS+AG including both copies
(f61yQcArGTqq, f61ySjnQuJ9V; custom ops -> this):
  T=64 37.7 -> 35.0, 128 67.4 -> 53.7, 192 91.3 -> 70.8, 256 134.3 -> 87.0
  (RCCL at 384 / 512 / 768: 147 / 165 / 239, LL128 tuner rules 122 / 162 /
  206; this path 121 / 154 / 217)
Race test: 16 distinct inputs per graph with rank-skewed spin kernels between
the ops, 40 replays, every output torch.equal to the custom ops: 0 mismatches
at T=64..1024. Fusing the copies into the kernels (8 blocks copying) was
slower (T=256 97 us).

INKLING_SCONV_XRSAG_MAX_T (default 256; 0 = off) is the largest forward that
takes this path; INKLING_SCONV_XRSAG_BLOCKS (default 8) the grid.

INKLING_SCONV_XRSAG_NORM (default 1): where the AG only feeds rmsnorm_slabs
(channel-sharded residual sites), copy the shard into the IPC buffer, run a
1-block barrier, and have a Triton rmsnorm read the 8 peer shards directly
(same layout and reduction as _rmsnorm_slabs_fwd_kernel, bit-identical).
Microbench (f61yynFBhbwV): -5.5..-8 us per site at T=64..512.

INKLING_SCONV_XRSAG_DIRECT (default 1): the sublayer outputs are written
straight into the RS input buffer, so the RS skips its input copy: attention's
rank-major o_proj bmm, the MoE routed+sink add_to_slabs, and the [T, H] ->
rank-major permute of other (dense MLP) outputs.

Numerics: bit-identical to the custom RS/AG ops it replaces (T <= 256).
Above INKLING_SCONV_CA_RSAG_MAX_T it replaces RCCL, whose bf16 ring sum
differs at rounding level; keep MAX_T <= CA_RSAG_MAX_T for a bit-exact path.
"""
import argparse
import importlib.util
import os
import sys

MARKER = "patch inkling_sconv_zero_copy_xrsag.py"
D = "vllm/models/inkling/amd/"
MODULE = D + "ops/xrsag.py"

MODULE_SRC = '''# SPDX-License-Identifier: Apache-2.0
# (patch inkling_sconv_zero_copy_xrsag.py)
"""IPC reduce-scatter / all-gather for Inkling's sconv slabs on TP8 gfx90a.

Same algorithm and fp32 rank-order sum as the image's custom RS/AG kernels
(bit-identical outputs), restructured for the per-site RS -> sconv -> AG chain:

* no trailing barrier: RS and AG always alternate on these buffers, and each
  op's start barrier proves every peer finished the previous op's peer reads
  (stream order), so its buffers may be rewritten;
* 8 blocks of 512 threads (6-12 swept; more blocks are slower);
* all_gather_rmsnorm: when the AG only feeds rmsnorm_slabs, a barrier kernel
  then the same Triton rmsnorm reading the rows straight from the peers' g
  buffers (no [8T, C] AG output written and re-read; bit-identical).

Buffers are allocated on the first call (an eager forward on every rank of
the TP group, before any graph capture) and kept for the process lifetime.
"""
from __future__ import annotations

import os

import torch

MAXB = 128
_BLOCKS = int(os.environ.get("INKLING_SCONV_XRSAG_BLOCKS", "8"))

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

struct Ptrs { void* p[8]; };

static Ptrs mk(const std::vector<int64_t>& v) {
  Ptrs p;
  for (int i = 0; i < 8; ++i) p.p[i] = (void*)(i < (int)v.size() ? v[i] : 0);
  return p;
}

// Per-block flag barrier over the 8 ranks (vLLM custom all-reduce design):
// thread k < 8 stores the block's next epoch into peer k's slot [block][rank]
// with system-scope release, then spins on its own slot [block][k].
__device__ __forceinline__ void xbarrier(const Ptrs& sig, uint32_t* epoch, int rank) {
  __syncthreads();
  uint32_t flag = epoch[blockIdx.x] + 1;
  if (threadIdx.x < 8) {
    uint32_t* peer = (uint32_t*)sig.p[threadIdx.x] + blockIdx.x * 8 + rank;
    __hip_atomic_store(peer, flag, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    uint32_t* self = (uint32_t*)sig.p[rank] + blockIdx.x * 8 + threadIdx.x;
    while (__hip_atomic_load(self, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < flag) {}
  }
  __syncthreads();
  if (threadIdx.x == 0) epoch[blockIdx.x] = flag;
}

__device__ __forceinline__ float bf2f(uint16_t v) { return __uint_as_float(((uint32_t)v) << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {
  // round-to-nearest-even, NaN-preserving (as __float2bfloat16)
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

// out[i] = sum_{k=0..7} in_k[rank * n16 + i] (16-byte rows), fp32 in rank
// order 0..7, RNE to bf16.  The caller copied this rank's [8, n16] slabs into
// in_rank first.
__global__ void __launch_bounds__(512, 1) xrs_kernel(Ptrs in, Ptrs sig, uint32_t* epoch,
                                                     uint4* __restrict__ out, int rank, int n16) {
  xbarrier(sig, epoch, rank);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x) {
    uint4 v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = ((const uint4*)in.p[j])[(size_t)rank * n16 + i];
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint16_t* h = (const uint16_t*)&v[j];
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] = (j == 0) ? bf2f(h[e]) : acc[e] + bf2f(h[e]);
    }
    uint4 o;
    uint16_t* oh = (uint16_t*)&o;
#pragma unroll
    for (int e = 0; e < 8; ++e) oh[e] = f2bf(acc[e]);
    out[i] = o;
  }
}

// out[k * n16 + i] = g_k[i]; the caller copied this rank's shard into g_rank.
__global__ void __launch_bounds__(512, 1) xag_kernel(Ptrs g, Ptrs sig, uint32_t* epoch,
                                                     uint4* __restrict__ out, int rank, int n16) {
  xbarrier(sig, epoch, rank);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x) {
    uint4 v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = ((const uint4*)g.p[(rank + j) & 7])[i];
#pragma unroll
    for (int j = 0; j < 8; ++j) out[(size_t)((rank + j) & 7) * n16 + i] = v[j];
  }
}

// Barrier only (block slot 0): the peers' g buffers hold their shards once it
// returns, for a following kernel that reads them.
__global__ void xbar_kernel(Ptrs sig, uint32_t* epoch, int rank) {
  xbarrier(sig, epoch, rank);
}

static int nblocks(int64_t blocks, int n16) {
  return (int)std::max<int64_t>(1, std::min<int64_t>(blocks, (n16 + 511) / 512));
}

void xrs(std::vector<int64_t> in, std::vector<int64_t> sig, torch::Tensor epoch,
         torch::Tensor out, int64_t rank, int64_t blocks) {
  int n16 = out.numel() * out.element_size() / 16;
  hipLaunchKernelGGL(xrs_kernel, dim3(nblocks(blocks, n16)), dim3(512), 0,
                     at::cuda::getCurrentCUDAStream(), mk(in), mk(sig),
                     (uint32_t*)epoch.data_ptr(), (uint4*)out.data_ptr(), (int)rank, n16);
}

void xag(std::vector<int64_t> g, std::vector<int64_t> sig, torch::Tensor epoch,
         torch::Tensor out, int64_t rank, int64_t blocks) {
  int n16 = out.numel() * out.element_size() / 16 / 8;
  hipLaunchKernelGGL(xag_kernel, dim3(nblocks(blocks, n16)), dim3(512), 0,
                     at::cuda::getCurrentCUDAStream(), mk(g), mk(sig),
                     (uint32_t*)epoch.data_ptr(), (uint4*)out.data_ptr(), (int)rank, n16);
}

void xbar(std::vector<int64_t> sig, torch::Tensor epoch, int64_t rank) {
  hipLaunchKernelGGL(xbar_kernel, dim3(1), dim3(64), 0, at::cuda::getCurrentCUDAStream(),
                     mk(sig), (uint32_t*)epoch.data_ptr(), (int)rank);
}

torch::Tensor view(int64_t ptr, int64_t numel, torch::Tensor like) {
  return torch::from_blob((void*)ptr, {numel}, like.options());
}

void zero(int64_t ptr, int64_t nbytes) {
  TORCH_CHECK(hipMemset((void*)ptr, 0, nbytes) == hipSuccess, "hipMemset failed");
  TORCH_CHECK(hipDeviceSynchronize() == hipSuccess, "hipDeviceSynchronize failed");
}
"""

_CPP = r"""
#include <torch/extension.h>
void xrs(std::vector<int64_t> in, std::vector<int64_t> sig, torch::Tensor epoch,
         torch::Tensor out, int64_t rank, int64_t blocks);
void xag(std::vector<int64_t> g, std::vector<int64_t> sig, torch::Tensor epoch,
         torch::Tensor out, int64_t rank, int64_t blocks);
void xbar(std::vector<int64_t> sig, torch::Tensor epoch, int64_t rank);
torch::Tensor view(int64_t ptr, int64_t numel, torch::Tensor like);
void zero(int64_t ptr, int64_t nbytes);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("xrs", &xrs); m.def("xag", &xag); m.def("view", &view); m.def("zero", &zero);
  m.def("xbar", &xbar);
}
"""


def _norm_kernel():
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _rmsnorm_peers_fwd_kernel(
        addr_ptr,  # [tp] int64 byte addresses of the ranks' g buffers ([T, seg] shards)
        weight_ptr,  # [N]
        y_ptr,  # [T, N] normed output
        eps,
        y_stride_0,
        n_cols,
        seg: tl.constexpr,
        block_size_n: tl.constexpr,
    ):
        # ops/norm.py _rmsnorm_slabs_fwd_kernel with the slab rows read from
        # the peers: same row layout and reduction, so y is bit-identical.
        pid_m = tl.program_id(0).to(tl.int64)
        offs_n = tl.arange(0, block_size_n)
        mask_n = offs_n < n_cols
        a = tl.load(addr_ptr + offs_n // seg) + (pid_m * seg + offs_n % seg) * 2
        p = tl.max_contiguous(tl.multiple_of(a.to(tl.pointer_type(tl.bfloat16)), 16), seg)
        s = tl.load(p, mask=mask_n, other=0.0)
        weight = tl.load(weight_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        x = s.to(tl.float32)
        row_var = tl.sum(x * x, axis=0) / n_cols
        rstd = tl.math.rsqrt(row_var + eps)
        y = x * rstd * weight
        tl.store(y_ptr + pid_m * y_stride_0 + offs_n, y, mask=mask_n)

    return _rmsnorm_peers_fwd_kernel


_NORM = None


def build_dir() -> str:
    return os.environ.get(
        "INKLING_SCONV_XRSAG_BUILD",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "_xrsag_build"),
    )


def load_ext(verbose: bool = False):
    """Compile (first use per build dir, ~1 min) or load the extension."""
    from torch.utils.cpp_extension import load_inline

    d = build_dir()
    os.makedirs(d, exist_ok=True)
    return load_inline(
        "inkling_xrsag_ext",
        cpp_sources=[_CPP],
        cuda_sources=[_SRC],
        build_directory=d,
        verbose=verbose,
        extra_cuda_cflags=["-O3", "--offload-arch=gfx90a"],
    )


class _XRsAg:
    def __init__(self, tp_group, nbytes: int) -> None:
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        self.ext = load_ext()
        cpu = tp_group.cpu_group
        self.rank = tp_group.rank_in_group
        self.capacity = nbytes  # bytes of the [8, t * c] slab input
        self.in_ptrs = CustomAllreduce.create_shared_buffer(nbytes, group=cpu)
        self.g_ptrs = CustomAllreduce.create_shared_buffer(nbytes // 8, group=cpu)
        self.sig_ptrs = CustomAllreduce.create_shared_buffer(MAXB * 8 * 4, group=cpu)
        self.ext.zero(self.sig_ptrs[self.rank], MAXB * 8 * 4)
        dev = torch.cuda.current_device()
        self.epoch = torch.zeros(MAXB, dtype=torch.int32, device=dev)
        like = torch.empty(1, dtype=torch.bfloat16, device=dev)
        self.in_view = self.ext.view(self.in_ptrs[self.rank], nbytes // 2, like)
        self.g_addr = torch.tensor(self.g_ptrs, dtype=torch.int64, device=dev)
        self.g_view = self.ext.view(self.g_ptrs[self.rank], nbytes // 16, like)
        torch.cuda.synchronize()
        torch.distributed.barrier(group=cpu)

    def reduce_scatter(self, slabs: torch.Tensor, out: torch.Tensor) -> None:
        """bf16 [8 * t, c] contiguous slabs -> out [t, c] (slab `rank` summed).
        Slabs already in in_view (written there by their producer) are not
        copied."""
        if slabs.data_ptr() != self.in_view.data_ptr():
            self.in_view[: slabs.numel()].copy_(slabs.view(-1))
        self.ext.xrs(self.in_ptrs, self.sig_ptrs, self.epoch, out, self.rank, _BLOCKS)

    def all_gather(self, shard: torch.Tensor, out: torch.Tensor) -> None:
        """bf16 [t, c] contiguous shard -> out [8 * t, c], rank-major."""
        self.g_view[: shard.numel()].copy_(shard.view(-1))
        self.ext.xag(self.g_ptrs, self.sig_ptrs, self.epoch, out, self.rank, _BLOCKS)

    def all_gather_rmsnorm(self, shard: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """rmsnorm_slabs(AG(shard)) without materializing the AG: bf16 [t, c]
        contiguous shard (c % 8 == 0) -> fresh [t, 8 * c] normed rows."""
        global _NORM
        if _NORM is None:
            _NORM = _norm_kernel()
        from vllm.triton_utils import triton

        t, c = shard.shape
        self.g_view[: shard.numel()].copy_(shard.view(-1))
        self.ext.xbar(self.sig_ptrs, self.epoch, self.rank)
        n_cols = 8 * c
        y = shard.new_empty(t, n_cols)
        block_size_n = triton.next_power_of_2(n_cols)
        # ops/norm.py _get_num_warps_from_block_size
        num_warps = (32 if block_size_n >= 32768 else 16 if block_size_n >= 8192
                     else 8 if block_size_n >= 2048 else 4)
        _NORM[(t,)](self.g_addr, weight, y, eps, y.stride(0), n_cols, c, block_size_n,
                    num_warps=num_warps)
        return y


_INSTANCES: dict[int, _XRsAg | None] = {}


def get(tp_group, nbytes: int) -> _XRsAg | None:
    """The group's instance, created on first call (collective over the TP
    group: every rank must call this at the same point, outside graph
    capture). None when the path does not apply."""
    key = id(tp_group)
    if key not in _INSTANCES:
        inst = None
        if (
            tp_group.world_size == 8
            and torch.version.hip is not None
            and torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.startswith(
                "gfx90a"
            )
            and not torch.cuda.is_current_stream_capturing()
            and 1 <= _BLOCKS <= MAXB
        ):
            inst = _XRsAg(tp_group, nbytes)
        _INSTANCES[key] = inst
    return _INSTANCES[key]


_MAX_T = int(os.environ.get("INKLING_SCONV_XRSAG_MAX_T", "256"))
_DIRECT = os.environ.get("INKLING_SCONV_XRSAG_DIRECT", "1") != "0"


def direct_slabs(tp: int, t: int, c: int, dtype: torch.dtype) -> torch.Tensor | None:
    """The TP group's RS input buffer as [tp, t, c] when this forward's sconv
    RS goes through it (same conditions as model._xrsag_for), else None. A
    producer writing its rank-major partial output there saves the RS input
    copy. Rewriting the buffer is safe: every RS on it is followed by its
    site's AG (or AG+norm) start barrier before the next producer runs, so no
    peer still reads it."""
    if not _DIRECT or dtype != torch.bfloat16 or not 0 < t <= _MAX_T:
        return None
    from vllm.distributed import get_tp_group

    xr = get(get_tp_group(), tp * _MAX_T * c * 2)
    if xr is None or tp * t * c * 2 > xr.capacity or (t * c * 2) % 16:
        return None
    return xr.in_view[: tp * t * c].view(tp, t, c)
'''

ENV_OLD = """_SRES_MIN_T = int(__import__("os").environ.get("INKLING_SCONV_SRES_MIN_T", "1"))
"""

ENV_NEW = """_SRES_MIN_T = int(__import__("os").environ.get("INKLING_SCONV_SRES_MIN_T", "1"))

# (patch inkling_sconv_zero_copy_xrsag.py) largest forward whose sconv RS/AG
# goes through ops/xrsag.py (IPC RS/AG kernels, TP8 gfx90a); 0 = off.
_XRSAG_MAX_T = int(__import__("os").environ.get("INKLING_SCONV_XRSAG_MAX_T", "256"))
# ... and with the AG fused into the following rmsnorm (no [8T, C] AG output).
_XRSAG_NORM = __import__("os").environ.get("INKLING_SCONV_XRSAG_NORM", "1") != "0"


def _xrsag_for(slabs: torch.Tensor, tp: int, t: int, c: int):
    # The TP group's xrsag instance when this site's RS/AG fits it, else None.
    # The first call (the eager profile run, same point on every rank)
    # allocates the IPC buffers.
    if _XRSAG_MAX_T <= 0:
        return None
    from .ops import xrsag

    nbytes = slabs.numel() * slabs.element_size()
    xr = xrsag.get(get_tp_group(), tp * _XRSAG_MAX_T * c * slabs.element_size())
    if (
        xr is not None
        and 0 < t <= _XRSAG_MAX_T
        and slabs.dtype == torch.bfloat16
        and slabs.is_contiguous()
        and nbytes <= xr.capacity
        and (nbytes // tp) % 16 == 0
    ):
        return xr
    return None
"""

HOOK_OLD = """    added to the sconv output before the AG.\"\"\"
    if 0 < t <= _CA_RSAG_MAX_T:
"""

HOOK_NEW = """    added to the sconv output before the AG.\"\"\"
    # (patch inkling_sconv_zero_copy_xrsag.py) IPC RS/AG kernels.
    xr = _xrsag_for(slabs, tp, t, c)
    if xr is not None:
        shard = slabs.new_empty(t, c)
        xr.reduce_scatter(slabs, shard)
        shard = sconv(shard, positions, add_res).contiguous()
        full = slabs.new_empty(tp * t, c)
        xr.all_gather(shard, full)
        return full, shard
    if 0 < t <= _CA_RSAG_MAX_T:
"""

NORM_OLD = """        full, shard = _slab_rs_sconv_ag(slabs, sconv, positions, tp, t, c, hidden)
"""

NORM_NEW = """        if norm_w is not None and _XRSAG_NORM and c % 8 == 0:
            # (patch inkling_sconv_zero_copy_xrsag.py) the AG only feeds the
            # norm: read the peers' shards straight into it.
            xr = _xrsag_for(slabs, tp, t, c)
            if xr is not None:
                shard = slabs.new_empty(t, c)
                xr.reduce_scatter(slabs, shard)
                shard = sconv(shard, positions, hidden).contiguous()
                return xr.all_gather_rmsnorm(shard, norm_w, eps), shard
        full, shard = _slab_rs_sconv_ag(slabs, sconv, positions, tp, t, c, hidden)
"""

SLABS_OLD = """        if delta.dim() == 3:
            slabs = delta.view(tp * t, c)
        else:
            slabs = delta.view(t, tp, c).transpose(0, 1).reshape(tp * t, c)
"""

SLABS_NEW = """        if delta.dim() == 3:
            slabs = delta.view(tp * t, c)
        else:
            # (patch inkling_sconv_zero_copy_xrsag.py) permute straight into
            # the xrsag RS input buffer when the RS reads it from there.
            from .ops import xrsag

            dst = xrsag.direct_slabs(tp, t, c, delta.dtype)
            if dst is None:
                slabs = delta.view(t, tp, c).transpose(0, 1).reshape(tp * t, c)
            else:
                dst.copy_(delta.view(t, tp, c).transpose(0, 1))
                slabs = dst.view(tp * t, c)
"""

ATTN_IMP_OLD = """from .ops.qkvr_prep import fused_qkvr_prep
"""

ATTN_IMP_NEW = """from .ops.qkvr_prep import fused_qkvr_prep
from .ops import xrsag as _xrsag  # (patch inkling_sconv_zero_copy_xrsag.py)
"""

ATTN_OLD = """                w.view(tp, w.shape[0] // tp, w.shape[1]).transpose(1, 2),
            )
"""

ATTN_NEW = """                w.view(tp, w.shape[0] // tp, w.shape[1]).transpose(1, 2),
                # (patch inkling_sconv_zero_copy_xrsag.py) straight into the
                # xrsag RS input buffer when the sconv RS reads it from there.
                out=_xrsag.direct_slabs(tp, num_tokens, w.shape[0] // tp, flat.dtype),
            )
"""

MOE_OLD = """            return add_to_slabs(out, sink_out, tp)
        return out.add_(sink_out)
"""

MOE_NEW = """            # (patch inkling_sconv_zero_copy_xrsag.py) straight into the
            # xrsag RS input buffer when the sconv RS reads it from there.
            from .ops import xrsag

            dst = xrsag.direct_slabs(tp, out.shape[0], out.shape[1] // tp, out.dtype)
            return add_to_slabs(out, sink_out, tp, dst=dst)
        return out.add_(sink_out)
"""

ADD_OLD = """def add_to_slabs(a: torch.Tensor, b: torch.Tensor, tp: int) -> torch.Tensor:
    \"\"\"``a + b`` (rounded to ``a.dtype``, like ``a.add_(b)``) for ``[T, N]``
    inputs, returned rank-major as fresh ``[tp, T, N // tp]`` slabs.\"\"\"
"""

ADD_NEW = """def add_to_slabs(
    a: torch.Tensor, b: torch.Tensor, tp: int, dst: torch.Tensor | None = None
) -> torch.Tensor:
    \"\"\"``a + b`` (rounded to ``a.dtype``, like ``a.add_(b)``) for ``[T, N]``
    inputs, returned rank-major as fresh ``[tp, T, N // tp]`` slabs.
    (patch inkling_sconv_zero_copy_xrsag.py) Written into ``dst`` (contiguous
    ``[tp, T, N // tp]``) instead when given.\"\"\"
"""

ADD_OUT_OLD = """    out = torch.empty((tp, n_rows, seg), dtype=a.dtype, device=a.device)
"""

ADD_OUT_NEW = """    out = torch.empty((tp, n_rows, seg), dtype=a.dtype, device=a.device) if dst is None else dst
"""

TARGETS = [
    (D + "model.py", [("env switch", ENV_OLD, ENV_NEW), ("RS/AG hook", HOOK_OLD, HOOK_NEW),
                      ("sres slabs", SLABS_OLD, SLABS_NEW),
                      ("sres norm site", NORM_OLD, NORM_NEW)]),
    (D + "attention.py", [("ops import", ATTN_IMP_OLD, ATTN_IMP_NEW),
                          ("o_proj slabs", ATTN_OLD, ATTN_NEW)]),
    (D + "moe.py", [("routed+sink slabs", MOE_OLD, MOE_NEW)]),
    (D + "ops/norm.py", [("add_to_slabs dst", ADD_OLD, ADD_NEW), ("add_to_slabs out", ADD_OUT_OLD, ADD_OUT_NEW)]),
]


def find_root():
    """site-packages dir holding vllm/, without importing vllm."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        sys.exit("vllm is not installed; run this inside the container")
    return os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))


def prebuild(path):
    """Compile the extension now so workers only load it."""
    try:
        spec = importlib.util.spec_from_file_location("_inkling_xrsag_build", path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        m.load_ext()
        print(f"built: {m.build_dir()}")
    except Exception as e:  # workers retry the build on first use
        print(f"WARNING: xrsag prebuild failed ({e!r}); workers will build on first use",
              file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report status without writing")
    ap.add_argument("--root", help="patch the vllm/ tree under this dir "
                    "instead of the installed vLLM's (no prebuild)")
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
    mod_path = os.path.join(root, MODULE)
    if not os.path.isdir(os.path.dirname(mod_path)):
        problems.append(f"{os.path.dirname(mod_path)} missing; vLLM layout changed")
    elif os.path.exists(mod_path):
        with open(mod_path) as fh:
            if fh.read() == MODULE_SRC:
                print(f"already written: {mod_path}")
            else:
                plans.append((mod_path, MODULE_SRC))
    else:
        plans.append((mod_path, MODULE_SRC))

    if problems:
        print("ERROR: patch target not found:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print("Inkling's sconv RS/AG path changed (needs inkling_sconv_zero_copy_ca.py "
              "and _sres.py applied first); re-derive or retire this patch",
              file=sys.stderr)
        return 1

    for path, patched in plans:
        if args.check:
            print(f"would write: {path}")
            continue
        with open(path, "w") as fh:
            fh.write(patched)
        print(f"patched: {path}  (Inkling sconv IPC RS/AG, INKLING_SCONV_XRSAG_MAX_T / _NORM / _DIRECT)")
    if not args.check and not args.root and os.environ.get("INKLING_SCONV_XRSAG_MAX_T", "256") != "0":
        prebuild(mod_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
