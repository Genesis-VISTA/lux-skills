#!/usr/bin/env python3
"""
Retile Inkling's relative-logits projection (qkvr_rel_proj) for gfx90a.

THE COST AS IT PRESENTS

Every attention layer's fused_qkvr_prep writes rel = q_r @ rel_proj,
(T, 4 heads, extent) bf16 per TP8 rank, extent 512 on local layers and 1024
(log-scaled) on global ones. K is 16, so the kernel is a pure store stream,
but it runs at ~450 GB/s: 74.8 us per local layer at an 8k chunk against a
26 us store roofline (out.zero_() of the same tensor), 2.0 ms per stage-0 8k
chunk (job f61reugSbS8f). The default 32x32 MFMA output layout leaves each
lane a column strip, so the row-major store is poorly coalesced.

THE FIX

Launch the same two kernels with matrix_instr_nonkdim=16 (16x16 MFMA layout)
and per-size tiles, gfx90a only. Single-GCD, CUDA-graph timed (job
f61xBiFTuNi3 / f61xEVNpGBjm), us served -> patched:
  extent 512:  T=256 5.4->2.7, 512 9.1->3.8, 1024 14.5->6.0, 4096 34.2->20.3,
               8192 74.8->41.1, 16384 145.5->86.8
  extent 1024: T=256 6.1->4.0, 1024 9.9->9.5, 8192 97.4->75.6, 16384 195->151
Stage 0 (19 local + 3 global layers): ~0.7 ms per 8k chunk, ~1.25 ms per 16k,
~0.16 ms per T=1024 decode step.

INKLING_REL_PROJ_MFMA16=0 restores the served launches.

Numerics: bit-identical. Same bf16 products, fp32 sum of 16 terms, same
rounding and log scaling; every variant torch.equal to the served kernel at
T=128..16384, both extents.
"""
import argparse
import importlib.util
import os
import sys

MARKER = "patch inkling_rel_proj_mfma16.py"
D = "vllm/models/inkling/amd/"

IMPORT_OLD = '''import torch

from vllm.triton_utils import tl, triton
'''

IMPORT_NEW = '''import functools
import os

import torch

from vllm.triton_utils import tl, triton
'''

HELPER_OLD = '''def qkvr_rel_proj(
'''

HELPER_NEW = '''# (patch inkling_rel_proj_mfma16.py) gfx90a: K=16 makes rel a store stream;
# the 16x16 MFMA output layout stores ~2x faster than the default 32x32 one.
# Bit-identical. INKLING_REL_PROJ_MFMA16=0 restores the served launches.
REL_PROJ_MFMA16 = os.environ.get("INKLING_REL_PROJ_MFMA16", "1") != "0"


@functools.cache
def _rel_proj_is_gfx90a(index: int) -> bool:
    return torch.cuda.get_device_properties(index).gcnArchName.startswith("gfx90a")


def rel_proj_mfma16_config(
    rows: int, rel_extent: int, device: torch.device
) -> tuple[bool, int, int, int, int] | None:
    """(throughput kernel, BLOCK_M, BLOCK_N, GROUP_M, num_warps), or None for
    the served launch."""
    if not REL_PROJ_MFMA16 or rel_extent not in (512, 1024):
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    if not _rel_proj_is_gfx90a(index):
        return None
    if rel_extent == 512:
        if rows < 4096:
            return False, 32, 64, 1, 4
        if rows < 65536:
            return True, 64, 128, 1, 4
        return True, 32, 256, 1, 4
    if rows < 2048:
        return False, 32, 64, 1, 4
    if rows < 32768:
        return True, 64, 128, 1, 4
    return True, 32, 512, 1, 8


def qkvr_rel_proj(
'''

LAUNCH_OLD = '''    if use_rel_proj_throughput(rows, rel_extent):
        grid = (
            triton.cdiv(rows, THROUGHPUT_BLOCK_M * THROUGHPUT_GROUP_M),
'''

LAUNCH_NEW = '''    tuned = rel_proj_mfma16_config(rows, rel_extent, qkvr.device)
    if tuned is not None:
        throughput, block_m, block_n, group_m, num_warps = tuned
        if throughput:
            _rel_proj_throughput_kernel[
                (triton.cdiv(rows, block_m * group_m), triton.cdiv(rel_extent, block_n))
            ](
                qkvr,
                rel_proj,
                rel_out,
                log_scaling_ptr,
                rows,
                qkvr.stride(0),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                GROUP_M=group_m,
                num_warps=num_warps,
                matrix_instr_nonkdim=16,
                **common,
            )
        else:
            _rel_proj_low_latency_kernel[
                (triton.cdiv(rows, block_m), triton.cdiv(rel_extent, block_n))
            ](
                qkvr,
                rel_proj,
                rel_out,
                log_scaling_ptr,
                rows,
                qkvr.stride(0),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=num_warps,
                matrix_instr_nonkdim=16,
                **common,
            )
        return

    if use_rel_proj_throughput(rows, rel_extent):
        grid = (
            triton.cdiv(rows, THROUGHPUT_BLOCK_M * THROUGHPUT_GROUP_M),
'''

TARGETS = [
    (D + "ops/qkvr_prep.py", [("imports", IMPORT_OLD, IMPORT_NEW),
                              ("config helper", HELPER_OLD, HELPER_NEW),
                              ("launch", LAUNCH_OLD, LAUNCH_NEW)]),
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
        print("Inkling's qkvr_rel_proj changed; re-derive or retire this patch",
              file=sys.stderr)
        return 1

    for path, patched in plans:
        if args.check:
            print(f"would patch: {path}")
            continue
        with open(path, "w") as fh:
            fh.write(patched)
        print(f"patched: {path}  (Inkling rel projection 16x16 MFMA tiles, "
              f"INKLING_REL_PROJ_MFMA16)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
