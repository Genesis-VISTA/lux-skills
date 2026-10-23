#!/usr/bin/env bash
# Submit an ICF 4-agent campaign.   scripts/campaign.sh [KEY=VALUE ...]   (the job's script_args, see its README)
#   scripts/campaign.sh                              # 3 MJ campaign against your running inkling-serve / inkling.serve job
#   scripts/campaign.sh SWEEP_MJ=2:4:1               # one campaign per laser energy, yield-vs-laser plot
#   scripts/campaign.sh SERVE_JOB=338068 ENERGY_MJ=2.5
#   scripts/campaign.sh BACKEND=self                 # serve gpt-oss-20b on this job's own node (exclusive, 8 GPUs)
#   SBATCH_ARGS="-t 02:00:00" scripts/campaign.sh ...
# Submits the same hpc_jobs/icf-campaign/job.lux.slurm VISTA does. Outputs land in runs/icf.<jobid>/.
set -eu
SK=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
source "$SK/env.lux.sh"
export RUN_DIR_Lux=$JOB_DIR
mkdir -p "$ICF_RUNS_DIR"
bash "$JOB_DIR/setup_lux.sh" >/dev/null || { bash "$JOB_DIR/setup_lux.sh"; exit 1; }
# The whole node, as in cluster_defaults.json. BACKEND=self runs vLLM on it, so it must be ours
# (start_vllm_lux.sh numbers GPUs as the node does), and gets more time.
case " $* " in
  *" BACKEND=self "*) T=${TIME:-02:00:00} ;;
  *)                  T=${TIME:-01:00:00} ;;
esac
RES="--exclusive --gpus-per-node=8 -t $T"
JID=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -N 1 --ntasks-per-node=1 -J "${JOB_NAME:-icf.campaign}" $RES \
      -o "$ICF_RUNS_DIR/%x.%j.out" -e "$ICF_RUNS_DIR/%x.%j.out" ${SBATCH_ARGS:-} "$JOB_DIR/job.lux.slurm" "$@" | tail -1)
echo "job $JID  log $ICF_RUNS_DIR/${JOB_NAME:-icf.campaign}.$JID.out  outputs $ICF_RUNS_DIR/icf.$JID/"
