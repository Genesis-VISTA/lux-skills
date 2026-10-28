#!/usr/bin/env bash
# Prepare the tutorial's work directory once on a Lux login node.
set -euo pipefail
SKILL_DIR=$(cd "$(dirname "$0")/.." && pwd)
source "$SKILL_DIR/env.lux.sh"
: "${ACCOUNT:?set ACCOUNT to your Lux Slurm account}"
export VISTA_JOB_DIR=$GRPO_ROOT
bash "$SKILL_DIR/hpc_jobs/grpo-train/setup_lux.sh"
