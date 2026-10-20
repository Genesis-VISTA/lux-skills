#!/usr/bin/env python3
"""
Small-T sconv RS/AG through vLLM's custom (IPC, one-shot) reduce-scatter and
all-gather kernels instead of RCCL. Apply after inkling_sconv_zero_copy.py
(alphabetical in the container patch loop); needs vllm_custom_ar_gfx90a.py with
VLLM_ROCM_CUSTOM_AR_GFX90A=1 so the TP group builds a CustomAllreduce on gfx90a.

THE COST AS IT PRESENTS

At decode-sized T the sconv RS+AG pair is latency-bound: flat RCCL RS+AG takes
42 us at T=4 and 118 us at T=128 per site (car, f61kGXbdrzuD), 42 sites per
stage pass. vLLM's custom RS/AG kernels take 24 us and 66 us (err 0, HIP graph
replay), and break even with RCCL near T=256.

THE FIX

INKLING_SCONV_CA_RSAG_MAX_T=<T> (default 0 = off): when the site's T is at most
that and the TP group has a live CustomAllreduce, _sconv_add_norm runs the
reduce-scatter and all-gather of the rank-major slabs with ops.
custom_reduce_scatter / ops.custom_all_gather (the python wrappers are
is_cuda()-gated, so the ops are called directly, as in car_bench.py). Shard
layout, sconv weights and conv state are unchanged; only the summation order
of the reduce differs from RCCL's.
"""
import argparse
import importlib.util
import os
import sys

MARKER = "patch inkling_sconv_zero_copy_ca.py"

MODEL_OLD = '''        _, t, c = delta.shape
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
'''

MODEL_NEW = '''        _, t, c = delta.shape
        # (patch inkling_sconv_zero_copy_ca.py) small T: custom RS/AG kernels.
        full = _slab_rs_sconv_ag(delta.view(tp * t, c), sconv, positions, tp, t, c)
        return add_rmsnorm_slabs(hidden, full.view(tp, t, c), norm_w, eps)
    if tp > 1 and delta.dim() == 2:
        t, h = delta.shape
        c = h // tp
        slabs = delta.view(t, tp, c).transpose(0, 1).reshape(tp * t, c)
        full = _slab_rs_sconv_ag(slabs, sconv, positions, tp, t, c)
'''

HELPER_OLD = '''def _sconv_add_norm(
'''

HELPER_NEW = '''# (patch inkling_sconv_zero_copy_ca.py) 0 = off.
_CA_RSAG_MAX_T = int(__import__("os").environ.get("INKLING_SCONV_CA_RSAG_MAX_T", "0"))


def _slab_rs_sconv_ag(
    slabs: torch.Tensor,
    sconv: InklingShortConv,
    positions: torch.Tensor,
    tp: int,
    t: int,
    c: int,
) -> torch.Tensor:
    """RS(dim 0) of contiguous rank-major ``[tp * t, c]`` slabs -> sconv on the
    ``[t, c]`` shard -> AG(dim 0); returns ``[tp * t, c]``. Small T goes
    through the custom (IPC) RS/AG kernels when the TP group has them."""
    if 0 < t <= _CA_RSAG_MAX_T:
        ca = getattr(get_tp_group().device_communicator, "ca_comm", None)
        nbytes = slabs.numel() * slabs.element_size()
        if (
            ca is not None
            and not ca.disabled
            and slabs.is_contiguous()
            and nbytes <= ca.max_reduce_scatter_size
            and nbytes // tp <= ca.max_all_gather_size
            and (nbytes // tp) % 16 == 0
        ):
            from vllm import _custom_ops as ops

            shard = slabs.new_empty(1, t * c)
            ops.custom_reduce_scatter(
                ca._ptr,
                slabs.view(tp, t * c),
                shard,
                ca.buffer_ptrs[ca.rank],
                ca.max_reduce_scatter_size,
            )
            shard = sconv(shard.view(t, c), positions).contiguous()
            full = slabs.new_empty(tp, t * c)
            ops.custom_all_gather(
                ca._ptr,
                shard.view(1, t * c),
                full,
                ca.buffer_ptrs[ca.rank],
                ca.max_all_gather_size,
            )
            return full.view(tp * t, c)
    shard = tensor_model_parallel_reduce_scatter(slabs, dim=0)
    shard = sconv(shard, positions)
    return tensor_model_parallel_all_gather(shard, dim=0)


def _sconv_add_norm(
'''

IMPORT_OLD = '''from vllm.distributed import (
    get_pp_group,
'''

IMPORT_NEW = '''from vllm.distributed import (
    get_pp_group,
    get_tp_group,
'''

D = "vllm/models/inkling/amd/"
TARGETS = [
    (D + "model.py", [("import", IMPORT_OLD, IMPORT_NEW),
                      ("helper", HELPER_OLD, HELPER_NEW),
                      ("_sconv_add_norm slabs", MODEL_OLD, MODEL_NEW)]),
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
        print("Inkling's sconv slab path changed (is inkling_sconv_zero_copy.py "
              "applied first?); re-derive or retire this patch", file=sys.stderr)
        return 1

    for path, patched in plans:
        if args.check:
            print(f"would patch: {path}")
            continue
        with open(path, "w") as fh:
            fh.write(patched)
        print(f"patched: {path}  (Inkling sconv small-T custom RS/AG)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
