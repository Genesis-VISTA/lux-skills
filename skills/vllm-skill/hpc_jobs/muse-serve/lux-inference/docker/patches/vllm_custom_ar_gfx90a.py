#!/usr/bin/env python3
"""
Let vLLM's CustomAllreduce run on MI250X (gfx90a), behind an env flag.

THE GATE

RocmPlatform.use_custom_allreduce() (vllm/platforms/rocm.py) returns True only
for gfx94*/gfx95*. ParallelConfig then forces disable_custom_all_reduce=True,
so the TP communicator never builds ca_comm on tioga. Nothing else blocks it:
a tioga node is fully XGMI-connected (amdsmi hops=1 for every GCD pair, so
CustomAllreduce.fully_connected is True), the kernels are built for gfx90a,
and the car microbench (f61kGXbdrzuD) measured ca all-reduce bit-exact against
RCCL (err 0) at 7.3 us (T=4) / 56.8 us (T=256) vs RCCL 33 / 158.5 for
T x 4096 bf16 under graph replay.

THE CHANGE

1. platforms/rocm.py: VLLM_ROCM_CUSTOM_AR_GFX90A=1 makes use_custom_allreduce()
   return True on gfx90a. Default 0 keeps today's behaviour, so applying the
   patch alone changes nothing. Values other than 0/1 raise.
2. custom_all_reduce.py: VLLM_CUSTOM_AR_MAX_SIZE_MB (default 8, today's
   max_size) sets the CustomAllreduce default max_size. should_custom_ar needs
   bytes < max_size, so T=1024 x 4096 bf16 (exactly 8 MiB) needs 16. Cost per
   rank: the uncached meta buffer grows by (max_size - 8 MiB); the IPC buffer
   is max(max_size, 16 MiB) and is unchanged up to 16.

WHAT STAYS OFF ON gfx90a

- AITER custom AR: rocm_aiter_ops.is_custom_all_reduce_enabled() is wrapped in
  @if_aiter_supported, which requires get_cdna_version() > 2 (gfx90a is 2), so
  it returns None regardless of VLLM_ROCM_USE_AITER*. AITER also isn't
  importable in the gfx90a image.
- QuickAllReduce: cuda_communicator builds qr_comm whenever custom AR is on,
  but QuickAllReduce._rocm_arch_available() keeps its own gfx94/95 gate and
  leaves it disabled.

EFFECT ON INKLING

None by itself: Inkling issues no tensor_model_parallel_all_reduce (MoE skips
the final reduce, attention uses reduce_results=False, sconv uses RS/AG). It
pairs with inkling_sconv_ar.py (INKLING_SCONV_AR=1). Graph capture also
registers the ca buffers (ca_comm.capture()), so expect a few MiB more and a
slightly longer capture.
"""
import argparse
import importlib.util
import os
import sys

MARKER = "patch vllm_custom_ar_gfx90a.py"

ROCM_OLD = '''    def use_custom_allreduce(cls) -> bool:
        # We only enable custom allreduce for MI300 series
        return any(gfx in _GCN_ARCH for gfx in ["gfx94", "gfx95"])
'''

ROCM_NEW = '''    def use_custom_allreduce(cls) -> bool:
        # We only enable custom allreduce for MI300 series
        # (patch vllm_custom_ar_gfx90a.py) ...and on MI250X when
        # VLLM_ROCM_CUSTOM_AR_GFX90A=1: a node is fully XGMI-connected and the
        # kernels are built for gfx90a. AITER and quick AR keep their own
        # CDNA3+ gates.
        flag = os.environ.get("VLLM_ROCM_CUSTOM_AR_GFX90A", "0")
        if flag not in ("0", "1"):
            raise ValueError(
                f"VLLM_ROCM_CUSTOM_AR_GFX90A must be 0 or 1, got {flag!r}"
            )
        if flag == "1" and "gfx90a" in _GCN_ARCH:
            return True
        return any(gfx in _GCN_ARCH for gfx in ["gfx94", "gfx95"])
'''

CA_CONST_OLD = '''from vllm.distributed.utils import is_weak_contiguous  # noqa: E402
'''

CA_CONST_NEW = '''from vllm.distributed.utils import is_weak_contiguous  # noqa: E402

# (patch vllm_custom_ar_gfx90a.py) default max all-reduce size; 8 MiB unless
# VLLM_CUSTOM_AR_MAX_SIZE_MB says otherwise (bytes must be < max_size).
_CA_DEFAULT_MAX_SIZE = int(
    __import__("os").environ.get("VLLM_CUSTOM_AR_MAX_SIZE_MB", "8")
) * 1024 * 1024
'''

CA_SIG_OLD = '''        max_size=8192 * 1024,
'''

CA_SIG_NEW = '''        max_size=_CA_DEFAULT_MAX_SIZE,
'''

TARGETS = [
    ("vllm/platforms/rocm.py",
     [("RocmPlatform.use_custom_allreduce", ROCM_OLD, ROCM_NEW)]),
    ("vllm/distributed/device_communicators/custom_all_reduce.py",
     [("max_size constant", CA_CONST_OLD, CA_CONST_NEW),
      ("CustomAllreduce.__init__ max_size", CA_SIG_OLD, CA_SIG_NEW)]),
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
        print("vLLM changed the ROCm custom-AR gate or CustomAllreduce's "
              "signature; re-derive or retire this patch", file=sys.stderr)
        return 1

    for path, patched in plans:
        if args.check:
            print(f"would patch: {path}")
            continue
        with open(path, "w") as fh:
            fh.write(patched)
        print(f"patched: {path}  (custom AR on gfx90a behind "
              "VLLM_ROCM_CUSTOM_AR_GFX90A, VLLM_CUSTOM_AR_MAX_SIZE_MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
