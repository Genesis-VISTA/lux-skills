#!/usr/bin/env python3
"""
Run MRv2's greedy rejection-sampler warmup pass on every PP rank, not just the
last one.

THE PROBLEM
vLLM main c1407a7e8 (#59260, 2026-10-14) added a second kernel-warmup pass to
vllm/v1/worker/gpu/warmup.py::warmup_kernels for speculative decoding:

    _warmup_kernels(model_runner, worker_execute_model, worker_sample_tokens)
    if rejection_sampler is not None:
        _warmup_kernels(..., greedy=True)

Each pass drives worker.execute_model, which under pipeline parallelism sends
(non-last ranks) or receives (non-first ranks) the intermediate tensors. But
model_runner.rejection_sampler is only built on the last PP rank (the sampler
block in GPUModelRunner.__init__ is under `if self.is_last_pp_rank`). With
PP > 1 and MTP, the last stage runs the greedy pass and blocks in
irecv_tensor_dict on its first prefill, while every earlier stage has already
returned from compile_or_warm_up_model. Startup hangs forever after the
"Capturing CUDA graphs" lines, with EngineCore logging "No available shared
memory broadcast block found in 60 seconds".

Seen on tioga (Inkling-Small-bf16, TP8 x PP2, MTP-1, vLLM 0e939af8 = muse +
main 642c18d64): every job hung that way, py-spy showing PP1 in
recv_object <- irecv_tensor_dict <- execute_model <- _warmup_kernels and PP0
idle in worker_busy_loop. bc0f47cd (the Inkling/DSV4 images) has no greedy pass.

THE FIX
Gate the greedy pass on model_runner.speculative_config, which every rank
has, so all stages run the same number of execute_model steps. On the
non-last stages `greedy` only changes the SamplingParams of the dummy
requests; the last stage behaves as before (it has a rejection sampler
whenever spec decode is on, unless a model's custom_sampler returns none, in
which case the extra pass only re-warms the plain sampler).

On a vLLM without the greedy pass (bc0f47cd) the anchor is absent and this is
a no-op, so the production pin builds unchanged.

Usage (inside the container):
    python3 spec_decode_pp_greedy_warmup.py [--check]
"""

import argparse
import importlib.util
import os
import sys

TARGET = "vllm/v1/worker/gpu/warmup.py"

OLD = '''        if rejection_sampler is not None:
            # Greedy batches skip logits processing, so rejection kernels
            # also see model-dtype logits instead of FP32.
            _warmup_kernels(
'''

NEW = '''        # zenith: every PP rank must run the same execute_model steps, and
        # only the last one has a rejection sampler (spec_decode_pp_greedy_warmup).
        if model_runner.speculative_config is not None:
            # Greedy batches skip logits processing, so rejection kernels
            # also see model-dtype logits instead of FP32.
            _warmup_kernels(
'''

MARKER = "spec_decode_pp_greedy_warmup"
# present only in a vLLM that has the greedy warmup pass (#59260)
FEATURE = "greedy=True"


def find_target():
    """Locate vLLM without importing it (no GPU in a build container)."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        sys.exit("vllm is not installed; run this inside the container")
    root = os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))
    return os.path.join(root, TARGET)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report status without writing")
    args = ap.parse_args()

    path = find_target()
    if not os.path.exists(path):
        print(f"not needed: {path} does not exist (no MRv2 kernel warmup)")
        return 0
    with open(path) as fh:
        src = fh.read()

    if MARKER in src:
        print(f"already patched: {path}")
        return 0
    if FEATURE not in src:
        print(f"not needed: {path} has no greedy warmup pass")
        return 0
    if OLD not in src:
        print(f"ERROR: patch target not found in {path}", file=sys.stderr)
        print("vLLM changed warmup_kernels(); re-derive the patch", file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {path}")
        return 0

    with open(path, "w") as fh:
        fh.write(src.replace(OLD, NEW, 1))
    print(f"patched: {path}  (greedy spec-decode warmup on every PP rank)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
