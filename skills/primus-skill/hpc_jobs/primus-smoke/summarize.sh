#!/usr/bin/env bash
# Summarize a Primus log: last-rank iteration lines (loss, step time, TFLOP/s/GPU) + verdict.
#   scripts/summarize.sh runs/smoke-N2_1234.log
set -eu
L=${1:?log file}
T=$(sed 's/\x1b\[[0-9;]*m//g' "$L")
# Metrics are printed by the LAST rank (rank-<n-1>/<n>), not rank 0; each line appears twice (:425/:429).
LAST=$(echo "$T" | grep -oE "rank-[0-9]+/[0-9]+\]" | sort -t- -k2 -n | uniq | tail -1 | tr -d ']')
echo "$T" | grep -E "$LAST.*utils.py:429.*\] iteration +[0-9]+ */" \
 | sed -E 's/.*(iteration +[0-9]+ *\/ *[0-9]+).*per iteration \(ms\): ([0-9.\/]+).*(TFLOP\/s\/GPU\): [0-9.]+( \(avg [0-9.]+\))?).*(lm loss: [0-9.E+-]+).*/\1  step_ms \2  \3  \5/' | sed -n '1p;3p;$p'
echo "$T" | grep -E "=== exit" || echo "NO exit line (job still running or killed)"
echo "$T" | grep -qE "Training execution failed|OutOfMemoryError" && echo "FAILED: see 'Training execution failed' in log" || true
