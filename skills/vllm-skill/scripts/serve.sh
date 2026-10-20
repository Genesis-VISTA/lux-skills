#!/usr/bin/env bash
# Submit an Inkling-MXFP4 server.   scripts/serve.sh [nodes] [KEY=VAL ...]
#   scripts/serve.sh                      # 2 nodes, TP8 x PP2, 1,048,576-token context (the tutorial default)
#   scripts/serve.sh 2 LUX_MTP=1          # same, MTP speculative decoding: ~1.7x faster for 1-2 users
#   scripts/serve.sh 1                    # 1 node, TP8, 100k context, tuned for total throughput
#   SBATCH_ARGS="-t 02:00:00 --exclude=lux123" scripts/serve.sh
# Submits the same hpc_jobs/inkling-serve/job.lux.slurm VISTA does. KEY=VAL pairs are exported, so every LUX_* knob
# works here (VISTA accepts only the job's short script_args). The job is named inkling.serve, which is what
# tutorial 1 (run_icf_with_inkling.sh) looks for. Outputs land in runs/serve.<jobid>/.
set -eu
SK=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
source "$SK/env.lux.sh"
N=${1:-2}; shift || true
for kv in "$@"; do export "$kv"; done
export RUN_DIR_Lux=$JOB_DIR
mkdir -p "$LUX_RUNS_DIR"
bash "$JOB_DIR/setup_lux.sh" >/dev/null || { bash "$JOB_DIR/setup_lux.sh"; exit 1; }
JID=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -N "$N" -J "${JOB_NAME:-inkling.serve}" -t 08:00:00 \
      --exclusive --gpus-per-node=8 --ntasks-per-node=1 \
      -o "$LUX_RUNS_DIR/%x.%j.out" -e "$LUX_RUNS_DIR/%x.%j.out" ${SBATCH_ARGS:-} "$JOB_DIR/job.lux.slurm" | tail -1)
echo "job $JID  log $LUX_RUNS_DIR/${JOB_NAME:-inkling.serve}.$JID.out  outputs $LUX_RUNS_DIR/serve.$JID/"
