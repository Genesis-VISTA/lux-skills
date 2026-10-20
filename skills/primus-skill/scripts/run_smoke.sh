#!/usr/bin/env bash
# Submit the Primus smoke test by hand.   scripts/run_smoke.sh <nodes> [KEY=VAL ...]
#   scripts/run_smoke.sh 1
#   scripts/run_smoke.sh 2 GLOBAL_BATCH_SIZE=512     # keep per-GPU batch fixed: gbs = 256 x nodes
# Submits the same job.lux.slurm VISTA does, with the resources of its cluster_defaults.json.
# Prints the job id and log path; outputs (train.log, summary.txt) land in <repo>/runs/<jobid>/.
set -eu
SK=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
source "$SK/env.lux.sh"
N=${1:?usage: run_smoke.sh <nodes> [KEY=VAL ...]}; shift || true
JOB=$SK/hpc_jobs/primus-smoke
export RUN_DIR_Lux=$JOB RUNS_DIR=$SK/runs
mkdir -p "$RUNS_DIR"
JID=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -N "$N" -t 00:30:00 -J "smoke-N$N" \
  --exclusive --gpus-per-node=8 --ntasks-per-node=1 -o "$RUNS_DIR/smoke-N${N}_%j.log" \
  "$JOB/job.lux.slurm" "$@" | tail -1)
echo "job $JID  log $RUNS_DIR/smoke-N${N}_${JID}.log  outputs $RUNS_DIR/$JID/"
