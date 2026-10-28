#!/usr/bin/env bash
# Submit tutorial 4's smoke, control, or pubonly run from a Lux login node.
# RL_* settings and extra sbatch arguments pass through to its tested submit.sh.
set -euo pipefail
MODE=${1:?usage: scripts/train.sh smoke|control|pubonly [sbatch arguments]}
shift
case $MODE in smoke|control|pubonly) ;; *) echo "unknown mode: $MODE" >&2; exit 2 ;; esac
SKILL_DIR=$(cd "$(dirname "$0")/.." && pwd)
source "$SKILL_DIR/env.lux.sh"
bash "$SKILL_DIR/scripts/setup.sh"
source "$PKG/tutorials/04_mistral_rl/hackathon_env.sh"
cd "$WORKDIR"
exec bash ./submit.sh "$MODE" "$@"
