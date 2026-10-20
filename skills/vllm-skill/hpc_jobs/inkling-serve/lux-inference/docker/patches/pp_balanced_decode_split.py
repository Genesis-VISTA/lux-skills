#!/usr/bin/env python3
"""
Split PP decode rows evenly between the in-flight micro-batches.

Why this patch exists
---------------------
With pipeline parallelism (PP=2) and async scheduling, each schedule() call
takes every running request that is not already in flight. An in-flight
request has num_new_tokens == 0 and is skipped, so the running set settles
into two alternating micro-batches -- but nothing makes them equal. The split
is set by whatever happened during the prefill wave and then sticks for the
whole decode tail:

    PP timeline f61wTnVBAobD-era runs, Inkling-Small TP8xPP2 on MI250X, c512:
      best_r1  260/252 reqs -> T 520/504  stage-0 fwd 48.4 / 45.3 ms
      best_r2  258/254 reqs -> T 516/508  stage-0 fwd 48.7 / 45.4 ms
      mtail2   256/256 reqs -> T 512/512  stage-0 fwd 45.0 / 45.0 ms

Four tokens past T=512 cost +3.5 ms (more than graph padding to 528 explains;
some kernel path changes above 512), and the PP cycle pays it every step.
Whether a run lands on 256/256 is luck: e2e c512 tails were 59.0 s
(r1/r2/r4, 10.36 tok/s/user) or 60.3-62.0 s (r3 and every variant, 10.0-10.2),
and c256 shows the same two modes at T=256 (tail 44.5 vs 47.5 s).

What it changes
---------------
When ZENITH_PP_DECODE_BALANCE_MIN is set, nothing is waiting, and
len(self.running) is at least that value, schedule() admits at most ceil(len(running) / pp_size) decode
requests per step. The rest are skipped (not preempted); they are not in
flight, so the next step -- the other micro-batch -- takes them. Prefill
chunks are not counted or capped. Unset (the default) or PP=1 leaves the
scheduler unchanged.

Usage: python3 pp_balanced_decode_split.py [--check | --revert] [--path FILE]
"""
import argparse
import importlib.util
import os
import sys

TARGET = "vllm/v1/core/sched/scheduler.py"
MARKER = "Zenith patch pp_balanced_decode_split.py"

OLD_1 = '''        # First, schedule the RUNNING requests.
        req_index = 0
'''
NEW_1 = '''        # First, schedule the RUNNING requests.
        req_index = 0
        # {marker}: cap decode rows per step at ceil(running / pp) so the two
        # PP micro-batches stay equal (a 260/252 split crosses a T>512 cliff).
        # Only once nothing is waiting: during the prefill wave the cap moved
        # decode rows next to prefill chunks and cost ~0.8 s at c512.
        pp_decode_cap = None
        pp_decode_count = 0
        if self.use_pp and not self.waiting and not self.skipped_waiting:
            _zmin = int(os.environ.get("ZENITH_PP_DECODE_BALANCE_MIN", "0") or 0)
            if _zmin > 0 and len(self.running) >= _zmin:
                _pp = self.parallel_config.pipeline_parallel_size
                pp_decode_cap = -(-len(self.running) // _pp)
'''.format(marker=MARKER)

OLD_2 = '''            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
'''
NEW_2 = '''            if pp_decode_cap is not None and not request.is_prefill_chunk:
                if pp_decode_count >= pp_decode_cap:
                    # left for the other micro-batch; not in flight, so the
                    # next step takes it
                    req_index += 1
                    continue
                pp_decode_count += 1

            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
'''

OLD_3 = '''import itertools
import time
'''
NEW_3 = '''import itertools
import os  # Zenith pp_balanced_decode_split.py
import time
'''

EDITS = [(OLD_1, NEW_1), (OLD_2, NEW_2), (OLD_3, NEW_3)]


def find_target():
    """Locate scheduler.py without importing vllm (no GPU in build containers)."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        sys.exit("vllm is not installed; run this inside the container")
    root = os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))
    path = os.path.join(root, TARGET)
    if not os.path.exists(path):
        sys.exit(f"expected {path} to exist; vLLM layout changed")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report status without writing")
    ap.add_argument("--revert", action="store_true",
                    help="restore the unpatched scheduler")
    ap.add_argument("--path", help="patch this file instead of the installed "
                    "vLLM's")
    args = ap.parse_args()

    path = args.path or find_target()
    with open(path) as fh:
        src = fh.read()
    edits = list(EDITS)

    if args.revert:
        if MARKER not in src:
            print(f"not patched: {path}")
            return 0
        if any(src.count(new) != 1 for _, new in edits):
            print(f"ERROR: {path} carries a different version of this patch; "
                  "restore scheduler.py from the image", file=sys.stderr)
            return 1
        for old, new in edits:
            src = src.replace(new, old, 1)
        with open(path, "w") as fh:
            fh.write(src)
        print(f"reverted: {path}")
        return 0

    if MARKER in src:
        print(f"already patched: {path}")
        return 0

    missing = [i for i, (old, _) in enumerate(edits) if src.count(old) != 1]
    if missing:
        print(f"ERROR: patch target(s) {missing} not found exactly once in {path}",
              file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {path}")
        return 0

    for old, new in edits:
        src = src.replace(old, new, 1)
    with open(path, "w") as fh:
        fh.write(src)
    print(f"patched: {path}  (PP decode rows capped at ceil(running/pp) when "
          "ZENITH_PP_DECODE_BALANCE_MIN is set)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
