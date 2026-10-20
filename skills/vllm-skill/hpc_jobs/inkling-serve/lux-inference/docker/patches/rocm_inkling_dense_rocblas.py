#!/usr/bin/env python3
"""
Inkling dense (non-MoE) prefill GEMMs on gfx90a: fixed rocBLAS solutions
instead of the hipBLASLt heuristic pick, for T >= 2048.

THE PROBLEM
Every Inkling dense GEMM goes through torch (F.linear / matmul / bmm), and
on MI250X torch takes hipBLASLt's heuristic kernel. At prefill sizes those
kernels reach only 34-60% of the 191.5 TF/s bf16 peak per GCD. In the eager
stage-0 profile of tree_m512_bss_pbwg_al_nm (job f61yVCnLFTK5, TP8, 22
layers), the dense GEMMs cost 48.6 ms of the 222 ms 8k chunk and 92.9 ms of
the 413 ms 16k chunk. Per TP rank, (N, K):
  qkvr 832x4096 (22/stage), wo_ud 4096x512 as the zero-copy (tp, T, C) bmm
  (22), sink w13 1024x4096 (20), sink w2 4096x512 (20), dense gate_up
  4096x4096 (2), dense down 4096x2048 (2).

THE FIX
ops/rocblas_dense.py calls rocblas_gemm_ex / rocblas_gemm_strided_batched_ex
through ctypes on the librocblas torch already has loaded, with one fixed
solution index per (N, K) and T range. The indices come from enumerating
every rocBLAS and hipBLASLt solution at T = 8192 / 16384 (moe_dg2, job
f61ybNTV3qMh) and from a T sweep of the best ones from 2048 to 16384 (moe_dg3,
job f61yhBNKjd3m). Single-GCD us, served -> this patch:

  GEMM        T=2048       T=8192       T=16384
  qkvr        (served)     540 -> 528   1009 -> 984   (from T=4096)
  sink w13    (served)     641 -> 596   1253 -> 1196  (from T=3072)
  sink w2     131 -> 105   396 -> 339    750 -> 653
  wo_ud bmm   132 -> 103   400 -> 334    755 -> 648
  gate_up     636 -> 600  2396 -> 2313  4795 -> 4448  (2048-4095 sol 569)
  down        376 -> 331  1408 -> 1189  2586 -> 2320

In situ (stage 0, 22 layers, TP8, real weights, served env), patch on vs
INKLING_DENSE_ROCBLAS_MIN_T=0 in the same job:
  - GEMM kernels on TP rank 0 (eager profile, job f61z2SBXpRZh): -4.6 ms
    per 8k chunk, -8.4 ms per 16k chunk.
  - Unprofiled stage forward, median of 7 steps (job f61zKD1qxfmR):
    8k 222.3 -> 218.4 ms (-1.75%), 16k 412.1 -> 404.5 ms (-1.84%).
  - INKLING_DENSE_ROCBLAS_CHECK=1 on real activations: max 1 ulp for
    every GEMM, rank and T.

The call sites (attention qkvr and wo_ud, dense MLP, sink experts) try
the rocBLAS path first and fall back to their own GEMM when it declines.
It declines for: any GPU but gfx90a; an (N, K) not in the table, i.e.
any other TP size or model; T below the threshold; non-bf16 or
non-unit-stride operands; quantized or biased layers; stream capture; or
a nonzero rocBLAS status. A nonzero status turns the path off for the
process, which covers a solution index that no longer exists after an
image rebuild. The arch check is there because the solution indices are
MI250X measurements: on gfx950 an index that happens to exist would run
an untuned kernel (seen on lux, job 229389), so the path is off there.
The indices were measured on librocblas.so.5 md5 c796431b, the copy in
the ROCm 7.14 SDK of the gfx90a image. Rerun the sweep after an image
rebuild: a valid but different kernel stays correct, only the speed
changes.

INKLING_DENSE_ROCBLAS_MIN_T: smallest T that takes the path (default 2048,
and never below each GEMM's own floor in the table); 0 turns it off.
INKLING_DENSE_ROCBLAS_CHECK=1 (debug, slow): also run the torch GEMM and
print the max bf16 ulp distance once per GEMM shape, T and solution.
FULL_DECODE_ONLY graphs are captured at T <= 1024, so they never see it.

Exactness: this is not bit-exact. Each GEMM is the same fp32-accumulated
dot product, but a different kernel sums K in a different order. Against
the served kernel, in every case swept (6 GEMMs x T in 2048..16384, 8191
and 13001 included, random weights), the result is within 1 bf16 ulp:
0.01-0.04% of elements differ, by exactly 1 ulp. The hipBLASLt solutions
that are bit-equal to served save at most 2.7% at T = 8192 / 16384
(dense down at 8192: 8.3%; job f61ybNTV3qMh), and Triton tl.dot is
slower on every GEMM but dense down at 8192.

Upstream status
---------------
Present on the pinned build 0.2.1.dev32+gbc0f47cd0 (vllm_py_src).
"""
import argparse
import importlib.util
import os
import sys

MARKER = "Zenith patch rocm_inkling_dense_rocblas.py"

D = "vllm/models/inkling/amd/"
HELPER = D + "ops/rocblas_dense.py"

HELPER_SRC = '''"""(Zenith patch rocm_inkling_dense_rocblas.py) Inkling dense prefill GEMMs
through fixed rocBLAS solutions; see the patch docstring. Every entry point
returns None when it declines, and the caller then runs its own GEMM."""
import ctypes
import glob
import os

import torch

_OP_N, _OP_T = 111, 112  # rocblas_operation_none / _transpose
_BF16, _F32 = 168, 151  # rocblas_datatype_bf16_r / _f32_r
_ALGO_STANDARD, _ALGO_SOLUTION_INDEX = 0, 1  # rocblas_gemm_algo

# Per-TP-rank weight (N, K) -> ((min T, rocBLAS solution index), ...), the
# first entry whose min T <= T wins.
_MM = {
    (832, 4096): ((4096, -606610598),),  # attention qkvr
    (1024, 4096): ((3072, -606610569),),  # sink experts w13
    (4096, 512): ((2048, -606610598),),  # sink experts w2
    (4096, 4096): ((4096, -606610594), (2048, -606610569)),  # dense gate_up
    (4096, 2048): ((2048, -606610598),),  # dense down
}
# wo_ud as the zero-copy bmm: (N, K, tp) -> as above
_SLABS = {(4096, 512, 8): ((2048, -606610597),)}

_MIN_T = max(0, int(os.environ.get("INKLING_DENSE_ROCBLAS_MIN_T", "2048") or 0))
_CHECK = os.environ.get("INKLING_DENSE_ROCBLAS_CHECK", "0") == "1"
_checked = set()
_primed = set()
_lib = None
_off = _MIN_T == 0
_handles = {}
_one = ctypes.c_float(1.0)
_zero = ctypes.c_float(0.0)


def _rocblas_path():
    """The librocblas torch uses: the mapped one, else the one in torch's ROCm
    SDK wheel (torch can load it lazily, after the first call here)."""
    with open("/proc/self/maps") as fh:
        for line in fh:
            p = line.split()[-1]
            if "/librocblas.so" in p:
                return p
    site = os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__)))
    for path in sorted(glob.glob(os.path.join(site, "_rocm_sdk*", "lib", "librocblas.so*"))):
        if os.path.isfile(path):
            return path
    return "librocblas.so.5"


def _load():
    global _lib
    i, p = ctypes.c_int, ctypes.c_void_p
    s = ctypes.c_int64
    lib = ctypes.CDLL(_rocblas_path(), mode=os.RTLD_GLOBAL)
    lib.rocblas_create_handle.argtypes = [ctypes.POINTER(p)]
    lib.rocblas_set_stream.argtypes = [p, p]
    lib.rocblas_gemm_ex.argtypes = [
        p, i, i, i, i, i, p, p, i, i, p, i, i, p, p, i, i, p, i, i, i, i, i,
        ctypes.c_uint32]
    lib.rocblas_gemm_ex.restype = i
    lib.rocblas_gemm_strided_batched_ex.argtypes = [
        p, i, i, i, i, i, p, p, i, i, s, p, i, i, s, p, p, i, i, s, p, i, i, s, i,
        i, i, i, ctypes.c_uint32]
    lib.rocblas_gemm_strided_batched_ex.restype = i
    _lib = lib
    return lib


def _handle(x):
    """rocBLAS handle for x's device, bound to the current stream."""
    dev = x.device.index
    h = _handles.get(dev)
    if h is None:
        h = ctypes.c_void_p()
        with torch.cuda.device(x.device):
            rc = _lib.rocblas_create_handle(ctypes.byref(h))
        if rc != 0:
            _failed(rc)
            return None
        _handles[dev] = h
        _lib.rocblas_set_stream(h, torch.cuda.current_stream(x.device).cuda_stream)
        _prime(h, x.device)
        if _off:
            return None
    _lib.rocblas_set_stream(h, torch.cuda.current_stream(x.device).cuda_stream)
    return h


def _prime(h, dev):
    """Run every solution in the tables once, on zeros at its smallest T.
    rocBLAS loads kernels lazily: an indexed call fails (status 6) until a
    heuristic call of the same problem has loaded them (job f61yyUbERe23),
    and loading costs seconds (2.7 s for the whole table in job
    f61zKD1qxfmR, then < 5 ms for the first call at any other T). This
    does it all at the first use, which is vLLM's startup profile run
    (a max_num_batched_tokens dummy batch)."""
    for (n, k), entries in _MM.items():
        w = torch.zeros((n, k), dtype=torch.bfloat16, device=dev)
        for lo, sol in entries:
            x = torch.zeros((lo, k), dtype=torch.bfloat16, device=dev)
            if _gemm(h, x, w, torch.empty((lo, n), dtype=x.dtype, device=dev), sol):
                return
    for (n, k, tp), entries in _SLABS.items():
        w = torch.zeros((n, k), dtype=torch.bfloat16, device=dev)
        for lo, sol in entries:
            x = torch.zeros((lo, k), dtype=torch.bfloat16, device=dev)
            out = torch.empty((tp, lo, n // tp), dtype=x.dtype, device=dev)
            if _gemm_slabs(h, x, w, out, tp, sol):
                return
    torch.cuda.synchronize(dev)


def _solution(table, key, t):
    for lo, sol in table.get(key, ()):
        if t >= max(lo, _MIN_T):
            return sol
    return None


def _ready(x, w):
    global _off
    if not (
        x.dim() == 2
        and x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and x.is_cuda
        and x.stride(1) == 1
        and x.stride(0) >= x.shape[1]
        and w.is_contiguous()
    ) or torch.cuda.is_current_stream_capturing():
        return None
    if _lib is None:
        if not torch.cuda.get_device_properties(x.device).gcnArchName.startswith("gfx90a"):
            _off = True
            print("[rocblas_dense] not gfx90a (the solutions are MI250X ones); "
                  "INKLING_DENSE_ROCBLAS off for this process", flush=True)
            return None
        try:
            _load()
        except (OSError, AttributeError) as e:
            _off = True
            print(f"[rocblas_dense] cannot load rocBLAS ({e}); INKLING_DENSE_ROCBLAS "
                  "off for this process", flush=True)
            return None
    return _handle(x)


def _call(key, run, sol):
    """run(algo, sol) -> rocBLAS status. A first indexed call can fail when the
    kernel library holding sol is not loaded yet; one heuristic call for the
    same problem loads it, then the indexed call is retried."""
    rc = run(_ALGO_SOLUTION_INDEX, sol)
    if rc != 0 and key not in _primed:
        _primed.add(key)
        if run(_ALGO_STANDARD, 0) == 0:
            rc = run(_ALGO_SOLUTION_INDEX, sol)
    if rc != 0:
        _failed(rc)
    return rc


def _gemm(h, x, w, out, sol):
    """out[T, N] = x[T, K] @ w[N, K].T with solution sol -> rocBLAS status."""
    t, k = x.shape
    n = w.shape[0]

    def run(algo, s):
        return _lib.rocblas_gemm_ex(
            h, _OP_T, _OP_N, n, t, k, ctypes.byref(_one), w.data_ptr(), _BF16, k,
            x.data_ptr(), _BF16, x.stride(0), ctypes.byref(_zero), out.data_ptr(), _BF16, n,
            out.data_ptr(), _BF16, n, _F32, algo, s, 0)

    return _call(("mm", n, k), run, sol)


def _gemm_slabs(h, x, w, out, tp, sol):
    """out[r] = x @ w[r*C:(r+1)*C].T for r < tp, out [tp, T, C] -> status."""
    t, k = x.shape
    c = w.shape[0] // tp

    def run(algo, s):
        return _lib.rocblas_gemm_strided_batched_ex(
            h, _OP_T, _OP_N, c, t, k, ctypes.byref(_one), w.data_ptr(), _BF16, k, c * k,
            x.data_ptr(), _BF16, x.stride(0), 0, ctypes.byref(_zero), out.data_ptr(), _BF16,
            c, t * c, out.data_ptr(), _BF16, c, t * c, tp, _F32, algo, s, 0)

    return _call(("bmm", w.shape[0], k, tp), run, sol)


def _failed(rc):
    global _off
    _off = True
    print(f"[rocblas_dense] rocBLAS status {rc}; INKLING_DENSE_ROCBLAS off for "
          "this process", flush=True)


def _check(key, out, ref):
    if key in _checked:
        return
    _checked.add(key)

    def ordered(t):
        i = t.view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)

    # ulp over |ref| >= 1e-3 rms with matching sign (a 1-ulp change of a value
    # next to zero is a large step in this ordering); rel = max |diff| / rms
    rf = ref.float()
    rms = rf.pow(2).mean().sqrt()
    d = (ordered(out) - ordered(ref)).abs()
    sel = (rf.abs() >= 1e-3 * rms) & (torch.sign(out.float()) == torch.sign(rf))
    rel = float((out.float() - rf).abs().max() / rms)
    print(f"[rocblas_dense] check {key}: max_ulp {int(d[sel].max())} "
          f"frac_diff {float((d > 0).float().mean()):.6f} rel {rel:.5f}", flush=True)


def mm(x, w):
    """x [T, K] @ w[N, K].T -> [T, N], or None."""
    if _off:
        return None
    t, k = x.shape[0], x.shape[-1]
    n = w.shape[0]
    sol = _solution(_MM, (n, k), t)
    if sol is None:
        return None
    h = _ready(x, w)
    if h is None:
        return None
    out = torch.empty((t, n), dtype=x.dtype, device=x.device)
    if _gemm(h, x, w, out, sol) != 0:
        return None
    if _CHECK:
        _check(("mm", n, k, t, sol), out, x @ w.T)
    return out


def linear(layer, x):
    """layer(x)[0] for an unquantized, bias-free vLLM linear layer whose
    forward is just the local GEMM (no gather / reduce), or None."""
    if _off or getattr(layer, "bias", None) is not None:
        return None
    if type(getattr(layer, "quant_method", None)).__name__ != "UnquantizedLinearMethod":
        return None
    return mm(x, layer.weight)


def bmm_slabs(x, w, tp):
    """bmm(x.expand(tp, T, K), w.view(tp, C, K).mT) -> [tp, T, C], or None."""
    if _off:
        return None
    t, k = x.shape[0], x.shape[-1]
    n = w.shape[0]
    sol = _solution(_SLABS, (n, k, tp), t)
    if sol is None:
        return None
    h = _ready(x, w)
    if h is None:
        return None
    c = n // tp
    out = torch.empty((tp, t, c), dtype=x.dtype, device=x.device)
    if _gemm_slabs(h, x, w, out, tp, sol) != 0:
        return None
    if _CHECK:
        ref = torch.bmm(x.unsqueeze(0).expand(tp, -1, -1), w.view(tp, c, k).transpose(1, 2))
        _check(("bmm", n, k, t, sol), out, ref)
    return out
'''

# (file, OLD, NEW): each OLD occurs exactly once in the file as left by the
# served patch stack (inkling_sconv_zero_copy.py adds the wo_ud bmm).
HUNKS = [
    (D + "attention.py",
     '''        num_tokens = hidden_states.shape[0]
        qkvr, _ = self.qkvr(hidden_states)
''',
     '''        num_tokens = hidden_states.shape[0]
        # Zenith patch rocm_inkling_dense_rocblas.py
        qkvr = _rocblas_dense.linear(self.qkvr, hidden_states)
        if qkvr is None:
            qkvr, _ = self.qkvr(hidden_states)
'''),
    (D + "attention.py",
     '''            w = self.wo_ud.weight
            tp = get_tensor_model_parallel_world_size()
            return torch.bmm(
''',
     '''            w = self.wo_ud.weight
            tp = get_tensor_model_parallel_world_size()
            # Zenith patch rocm_inkling_dense_rocblas.py
            out = _rocblas_dense.bmm_slabs(flat, w, tp)
            if out is not None:
                return out
            return torch.bmm(
'''),
    (D + "attention.py",
     '''from .ops.qkvr_prep import fused_qkvr_prep
''',
     '''from .ops.qkvr_prep import fused_qkvr_prep
from .ops import rocblas_dense as _rocblas_dense  # Zenith patch rocm_inkling_dense_rocblas.py
'''),
    (D + "mlp.py",
     '''        gate_up, _ = self.gate_up_proj(x)
        x = silu_and_mul_triton(gate_up)
        x, _ = self.down_proj(x)
''',
     '''        from .ops import rocblas_dense  # Zenith patch rocm_inkling_dense_rocblas.py

        gate_up = rocblas_dense.linear(self.gate_up_proj, x)
        if gate_up is None:
            gate_up, _ = self.gate_up_proj(x)
        x = silu_and_mul_triton(gate_up)
        y = rocblas_dense.linear(self.down_proj, x)
        if y is None:
            y, _ = self.down_proj(x)
        x = y
'''),
    (D + "moe.py",
     '''        raw = x @ self.w13_weight.view(-1, x.shape[-1]).T  # (T, S*2F)
        h = sink_silu_mul_epilogue(
            raw, self._unit, gammas, self._unit, self.n_experts, x.dtype
        )
        return h @ self.w2_weight.T  # (T, D)
''',
     '''        from .ops import rocblas_dense  # Zenith patch rocm_inkling_dense_rocblas.py

        raw = rocblas_dense.mm(x, self.w13_weight.view(-1, x.shape[-1]))
        if raw is None:
            raw = x @ self.w13_weight.view(-1, x.shape[-1]).T  # (T, S*2F)
        h = sink_silu_mul_epilogue(
            raw, self._unit, gammas, self._unit, self.n_experts, x.dtype
        )
        out = rocblas_dense.mm(h, self.w2_weight)
        return out if out is not None else h @ self.w2_weight.T  # (T, D)
'''),
]

# Alternative (OLD, NEW) for a HUNKS entry, used when the primary OLD is absent
# because another patch rewrote that call site first. rocm_inkling_decode_tgemm.py
# (applied before this one) routes qkvr, the wo_ud slabs and the sink GEMMs
# through dec_tgemm at decode M. Each declines outside its T range (dec_tgemm
# above decode M, rocBLAS below INKLING_DENSE_ROCBLAS_MIN_T), then torch runs.
ALT_HUNKS = {
    0: (
     '''        if qkvr is None:
            qkvr, _ = self.qkvr(hidden_states)
''',
     '''        if qkvr is None:
            # Zenith patch rocm_inkling_dense_rocblas.py
            qkvr = _rocblas_dense.linear(self.qkvr, hidden_states)
        if qkvr is None:
            qkvr, _ = self.qkvr(hidden_states)
'''),
    1: (
     '''            slabs = dec_tgemm(flat, w, "wo", slab_tp=tp)
            if slabs is not None:
                return slabs
            return torch.bmm(
''',
     '''            slabs = dec_tgemm(flat, w, "wo", slab_tp=tp)
            if slabs is not None:
                return slabs
            # Zenith patch rocm_inkling_dense_rocblas.py
            out = _rocblas_dense.bmm_slabs(flat, w, tp)
            if out is not None:
                return out
            return torch.bmm(
'''),
    4: (
     '''        raw = dec_tgemm(x, w13, "sink_g1")
        if raw is None:
            raw = x @ w13.T  # (T, S*2F)
        h = sink_silu_mul_epilogue(
            raw, self._unit, gammas, self._unit, self.n_experts, x.dtype
        )
        out = dec_tgemm(h, self.w2_weight, "sink_g2")
        return out if out is not None else h @ self.w2_weight.T  # (T, D)
''',
     '''        from .ops import rocblas_dense  # Zenith patch rocm_inkling_dense_rocblas.py

        raw = rocblas_dense.mm(x, w13)
        if raw is None:
            raw = dec_tgemm(x, w13, "sink_g1")
        if raw is None:
            raw = x @ w13.T  # (T, S*2F)
        h = sink_silu_mul_epilogue(
            raw, self._unit, gammas, self._unit, self.n_experts, x.dtype
        )
        out = rocblas_dense.mm(h, self.w2_weight)
        if out is None:
            out = dec_tgemm(h, self.w2_weight, "sink_g2")
        return out if out is not None else h @ self.w2_weight.T  # (T, D)
'''),
}


def resolve_hunks(srcs):
    """HUNKS with each entry swapped for its ALT_HUNKS form when only that matches."""
    hunks = []
    for i, (f, old, new) in enumerate(HUNKS):
        alt = ALT_HUNKS.get(i)
        if alt is not None and new not in srcs[f] and srcs[f].count(old) != 1:
            old2, new2 = alt
            if new2 in srcs[f] or srcs[f].count(old2) == 1:
                old, new = old2, new2
        hunks.append((f, old, new))
    return hunks


def find_root():
    """Locate site-packages without `import vllm` (that touches the GPU)."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        sys.exit("vllm is not installed; run this inside the container")
    return os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report status without writing")
    ap.add_argument("--revert", action="store_true",
                    help="restore the torch dense GEMMs and remove ops/rocblas_dense.py")
    ap.add_argument("--path", "--root", dest="root",
                    help="directory holding the vllm package to patch instead of "
                    "the installed one (e.g. vllm_py_src)")
    args = ap.parse_args()

    root = args.root or find_root()
    files = sorted({f for f, _, _ in HUNKS})
    srcs = {}
    for f in files:
        path = os.path.join(root, f)
        if not os.path.exists(path):
            print(f"ERROR: expected {path} to exist; vLLM layout changed",
                  file=sys.stderr)
            return 1
        with open(path) as fh:
            srcs[f] = fh.read()
    helper = os.path.join(root, HELPER)
    helper_src = open(helper).read() if os.path.exists(helper) else None
    hunks = resolve_hunks(srcs)

    applied = [new in srcs[f] for f, _, new in hunks]
    marked = any(MARKER in s for s in srcs.values()) or helper_src is not None

    if args.revert:
        if not marked:
            print(f"not patched: {root}")
            return 0
        if not all(applied) or helper_src != HELPER_SRC:
            print(f"ERROR: {root} carries a different version of this patch; "
                  f"restore {', '.join(files)} from the image and delete {HELPER}",
                  file=sys.stderr)
            return 1
        if args.check:
            print(f"would revert: {root}")
            return 0
        for f, old, new in reversed(hunks):
            srcs[f] = srcs[f].replace(new, old, 1)
        for f in files:
            with open(os.path.join(root, f), "w") as fh:
                fh.write(srcs[f])
        os.remove(helper)
        print(f"reverted: {root}")
        return 0

    if all(applied) and helper_src == HELPER_SRC:
        print(f"already patched: {root}")
        return 0
    if all(applied) and helper_src is not None and MARKER in helper_src:
        # an image that baked in an earlier version of this patch: the call
        # sites are the same, only ops/rocblas_dense.py changed
        if args.check:
            print(f"would update {HELPER}: {root}")
            return 0
        with open(helper, "w") as fh:
            fh.write(HELPER_SRC)
        print(f"patched: {root}  (updated {HELPER} from an earlier version)")
        return 0
    if marked:
        print(f"ERROR: {root} is partly patched or carries a different version "
              f"of this patch; restore {', '.join(files)} from the image and "
              f"delete {HELPER}", file=sys.stderr)
        return 1

    missing = [(f, old) for f, old, _ in hunks if srcs[f].count(old) != 1]
    if missing:
        for f, old in missing:
            print(f"ERROR: patch anchor not found once in {f}:\n{old}",
                  file=sys.stderr)
        print("vLLM changed the Inkling dense GEMM call sites (or "
              "inkling_sconv_zero_copy.py is not applied); re-derive or retire "
              "this patch", file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {root}")
        return 0

    for f, old, new in hunks:
        srcs[f] = srcs[f].replace(old, new, 1)
    for f in files:
        with open(os.path.join(root, f), "w") as fh:
            fh.write(srcs[f])
    with open(helper, "w") as fh:
        fh.write(HELPER_SRC)
    print(f"patched: {root}  (Inkling dense prefill GEMMs -> fixed rocBLAS "
          "solutions for T >= INKLING_DENSE_ROCBLAS_MIN_T, default 2048; 0 to "
          "disable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
