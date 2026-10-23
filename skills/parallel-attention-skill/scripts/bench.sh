#!/usr/bin/env bash
# Submit the RingX correctness tests and kernel benchmark.   scripts/bench.sh [KEY=VALUE ...]   (the job's
# script_args, see hpc_jobs/ringx-bench/README.md)
#   scripts/bench.sh                                          # tests + the default causal and bi-directional points
#   scripts/bench.sh MODE=bench CAUSAL_POINTS=1:65536,4:16384,16:4096 BIDIR_POINTS=
#   scripts/bench.sh MODE=correctness
#   TIME=02:00:00 scripts/bench.sh REPEATS=3 ...
#   NODES=4 scripts/bench.sh MODE=bench               # context parallel over 32 GPUs (4 nodes)
# Submits the same hpc_jobs/ringx-bench/job.lux.slurm VISTA does. Outputs land in runs/ringx.<jobid>/.
set -eu
SK=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
source "$SK/env.lux.sh"
export RUN_DIR_Lux=$JOB_DIR
mkdir -p "$RINGX_RUNS_DIR"
bash "$JOB_DIR/setup_lux.sh" >/dev/null || { bash "$JOB_DIR/setup_lux.sh"; exit 1; }
# Whole nodes, as in cluster_defaults.json; NODES (default 1) of them.
JID=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -N "${NODES:-1}" --ntasks-per-node=1 -J "${JOB_NAME:-ringx.bench}" \
      --exclusive --gpus-per-node=8 -t "${TIME:-01:00:00}" \
      -o "$RINGX_RUNS_DIR/%x.%j.out" -e "$RINGX_RUNS_DIR/%x.%j.out" ${SBATCH_ARGS:-} "$JOB_DIR/job.lux.slurm" "$@" | tail -1)
echo "job $JID  log $RINGX_RUNS_DIR/${JOB_NAME:-ringx.bench}.$JID.out  outputs $RINGX_RUNS_DIR/ringx.$JID/"
