#!/usr/bin/env bash
# Settings for the Mistral-Large verl/GRPO tutorial on the Lux HPC side.
# Override values before sourcing; all large state stays on Orion Lustre.
export VERL_GRPO_SKILL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ACCOUNT=${ACCOUNT:-$(sacctmgr -nP show assoc user="$USER" format=Account 2>/dev/null | awk -F'|' 'NF && $1 != "" {print $1; exit}')}
export PKG=${PKG:-/lustre/orion/world-shared/stf218/junqi/Genesis-Hackathon}
export GRPO_ROOT=${GRPO_ROOT:-/lustre/orion/${ACCOUNT:-unknown}/scratch/$USER/verl-grpo-skill}
export WORKDIR=${WORKDIR:-$GRPO_ROOT/work}

[ -n "$ACCOUNT" ] || echo "env.lux.sh: set ACCOUNT to your Lux Slurm account" >&2
