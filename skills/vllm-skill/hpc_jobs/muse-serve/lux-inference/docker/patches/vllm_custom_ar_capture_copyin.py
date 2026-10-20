#!/usr/bin/env python3
"""
Graph capture without IPC-registered buffers for vLLM's custom all-reduce, behind an env flag.

THE PROBLEM

Under HIP graph capture, CudaCommunicator builds CustomAllreduce with register=True
(unless the cuMem graph pool is on): captured all-reduces read the graph's own
buffers, IPC-registered after capture (get_graph_buffer_ipc_meta). On MI250X
(gfx90a, with vllm_custom_ar_gfx90a.py's VLLM_ROCM_CUSTOM_AR_GFX90A=1) that path
corrupts Muse-Spark-Open's outputs at every capture size: probe 6/30, fluent but
wrong text (tioga f63aCjuJYcas, capture 1..256 and 1..8).

THE CHANGE

device_communicators/cuda_communicator.py: VLLM_CUSTOM_AR_CAPTURE_REGISTER=0 sets
register=False, so a captured all-reduce copies its input into the pre-registered
IPC buffer, as an eager one does. Default 1 keeps today's behaviour, so applying
the patch alone changes nothing. Values other than 0/1 raise.

On tioga (Muse bf16, TP8 x PP6, graphs FULL_DECODE_ONLY 1..256 + custom AR) this
is the configuration that scored curve 6.91 with probe 30/30 and gsm8k 0.98
(f63bSzuwRVS7), vs 4.70 for graphs with RCCL and 3.00 eager. It was served from
the muse_hl_overlay_tioga overlay tree, whose only change to the installed vLLM
was this one; this patch makes the same edit in place.
"""
import argparse
import importlib.util
import os
import sys

MARKER = "patch vllm_custom_ar_capture_copyin.py"

IMPORT_OLD = '''

import torch
from torch.distributed import ProcessGroup
'''

IMPORT_NEW = '''

import os  # (patch vllm_custom_ar_capture_copyin.py)

import torch
from torch.distributed import ProcessGroup
'''

GATE_OLD = '''        # cuMem graph buffers cannot be IPC-registered; capture copies them instead.
        config = get_current_vllm_config_or_none()
        register = config is None or not config.use_cumem_cudagraph_pool
'''

GATE_NEW = '''        # cuMem graph buffers cannot be IPC-registered; capture copies them instead.
        config = get_current_vllm_config_or_none()
        register = config is None or not config.use_cumem_cudagraph_pool
        # (patch vllm_custom_ar_capture_copyin.py) VLLM_CUSTOM_AR_CAPTURE_REGISTER=0
        # copies under capture too (the eager path): on gfx90a, IPC-registered
        # graph buffers gave corrupted outputs.
        _flag = os.environ.get("VLLM_CUSTOM_AR_CAPTURE_REGISTER", "1")
        if _flag not in ("0", "1"):
            raise ValueError(
                f"VLLM_CUSTOM_AR_CAPTURE_REGISTER must be 0 or 1, got {_flag!r}"
            )
        if _flag == "0":
            register = False
'''

TARGETS = [
    ("vllm/distributed/device_communicators/cuda_communicator.py",
     [("import os", IMPORT_OLD, IMPORT_NEW),
      ("CudaCommunicator.__init__ register", GATE_OLD, GATE_NEW)]),
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
        print("vLLM changed CudaCommunicator's custom-AR registration; "
              "re-derive or retire this patch", file=sys.stderr)
        return 1

    for path, patched in plans:
        if args.check:
            print(f"would patch: {path}")
            continue
        with open(path, "w") as fh:
            fh.write(patched)
        print(f"patched: {path}  (custom AR capture copy-in behind "
              "VLLM_CUSTOM_AR_CAPTURE_REGISTER)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
