#!/usr/bin/env bash
# Preflight (seconds): account, the uploaded app's files, the BACKEND=self model and image. Changes nothing.
# The checks are the job folder's setup_lux.sh. The app's venv is built by the first job, on its compute node.
set -eu
SK=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
source "$SK/env.lux.sh"
ok=1; say() { printf '%-8s %s\n' "$1" "$2"; }
[ -n "$ACCOUNT" ] && say account "$ACCOUNT" || { say account "MISSING: export ACCOUNT=<proj>"; ok=0; }
RUN_DIR_Lux=$JOB_DIR bash "$JOB_DIR/setup_lux.sh" || ok=0
mkdir -p "$ICF_RUNS_DIR"
[ "$ok" = 1 ] && echo "ready: scripts/campaign.sh" || { echo "setup: fix the lines above"; exit 1; }
