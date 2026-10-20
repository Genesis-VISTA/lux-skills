#!/usr/bin/env bash
# Summarize a serve / bench job: startup milestones, fabric, KV pool, correctness, benchmark scores, failures.
#   scripts/summarize.sh <jobid | runs/inkling.serve.<jobid>.out>
# The job writes the same summary to runs/serve.<jobid>/summary.txt every minute (hpc_jobs/inkling-serve/summarize.sh).
set -eu
SK=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
R=${LUX_RUNS_DIR:-$SK/runs}
A=${1:?usage: summarize.sh <jobid or job log>}
if [ -f "$A" ]; then L=$A; J=$(basename "$A" | grep -oE '[0-9]+' | tail -1); else J=$A; L=$(ls "$R"/*."$J".out 2>/dev/null | head -1); fi
[ -f "${L:-}" ] || { echo "no job log for $A in $R"; exit 1; }
D=$R/serve.$J
echo "job log:    $L"; echo "server log: $D/server.log"
bash "$SK/hpc_jobs/inkling-serve/summarize.sh" "$D" "$L"
squeue -h -j "$J" -o 'state %T, elapsed %M of %l, nodes %N' 2>/dev/null || true
