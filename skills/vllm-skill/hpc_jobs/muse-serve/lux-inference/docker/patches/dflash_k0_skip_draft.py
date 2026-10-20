#!/usr/bin/env python3
"""
Skip the DFlash/DSpark draft forward on a step whose K is 0.

THE PROBLEM
A batch-size K schedule (speculative config
"num_speculative_tokens_per_batch_size": [[1, 96, 7], [97, 2048, 0]]) sets the
number of draft tokens per step from the number of scheduled requests, and K=0
is allowed. The runner passes that K to the speculator,

    vllm/v1/worker/gpu/model_runner.py
        draft_tokens = self.speculator.propose(..., num_speculative_tokens=K)
        if K < self.num_speculative_steps:
            draft_tokens[:, K:] = -1

but DFlashSpeculator.propose (vllm/v1/worker/gpu/spec_decode/dflash/
speculator.py), which DSpark and DFlash2 inherit, never reads the argument. At
K=0 every step still runs the full parallel draft pass and its sampling, and
the runner throws every draft away. On lux (Muse-Spark-Open-816B-A42B-Quark-
MXFP4, EP8, DSpark, vLLM 0e939af8, job 350029, 2026-10-18) that made the K=0
rungs (c>=128) 0.65x to 0.72x of a server with no draft at all.

THE FIX
When K is 0 for the step, run only the context half of propose: prepare the
draft inputs, combine and store the target's hidden states, publish the
context anchor, and write the draft's context K/V for every token the target
just processed. That keeps the draft's KV complete, so when the batch falls
back under the cutoff and K returns to 7 the draft sees the whole context.
Then return without the draft forward, its attention-metadata build or its
sampling. The context step runs eagerly (the FULL draft graph also contains
the draft forward). Only for data_parallel_size 1: with DP the draft forward is
a DP-synchronized step that every rank must run. Dummy and profile runs are
unchanged. Inert unless a step is scheduled with K=0.

Usage (inside the container):
    python3 dflash_k0_skip_draft.py [--check]
"""

import argparse
import importlib.util
import os
import sys

TARGET = "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"

OLD = '''                self.max_model_len,
                self.sample_from_anchor,
            )

        batch_sync, num_batch_tokens = (
'''

NEW = '''                self.max_model_len,
                self.sample_from_anchor,
            )

        if (
            num_speculative_tokens == 0
            and not dummy_run
            and not is_profile
            and self.dp_size == 1
        ):
            # zenith: K=0 this step, so the runner discards every draft. Store
            # the context K/V only, so the draft has the full context when K
            # rises again, and skip the draft forward (dflash_k0_skip_draft).
            logger.info_once(
                "%s: skipping the draft forward on K=0 steps", self._speculator_name
            )
            self._store_hidden_states(
                num_target_tokens, last_hidden_states, aux_hidden_states
            )
            self.prepare_context_anchor(
                num_reqs, input_batch.query_start_loc, num_rejected
            )
            self._precompute_context_kv(0, num_target_tokens)
            return self.draft_tokens[:num_reqs]

        batch_sync, num_batch_tokens = (
'''

MARKER = "dflash_k0_skip_draft"


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
        print(f"not needed: {path} does not exist (no DFlash speculator)")
        return 0
    with open(path) as fh:
        src = fh.read()

    if MARKER in src:
        print(f"already patched: {path}")
        return 0
    if "num_speculative_tokens: int | None = None" not in src:
        print(f"not needed: {path} has no per-step K (no K schedule)")
        return 0
    if src.count(OLD) != 1:
        print(f"ERROR: patch target not found once in {path}", file=sys.stderr)
        print("vLLM changed DFlashSpeculator.propose; re-derive the patch", file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {path}")
        return 0

    with open(path, "w") as fh:
        fh.write(src.replace(OLD, NEW, 1))
    print(f"patched: {path}  (DFlash/DSpark skip the draft forward on K=0 steps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
