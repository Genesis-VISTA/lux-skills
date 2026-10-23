#!/usr/bin/env bash
# Preflight and setup on the login node: account, then the job folder's setup_lux.sh (RingX clone at the pinned
# commit, ring-flash-attn, the image). Idempotent: the same script VISTA runs before every submission.
set -eu
SK=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
source "$SK/env.lux.sh"
[ -n "$ACCOUNT" ] && echo "account  $ACCOUNT" || { echo "account  MISSING: export ACCOUNT=<proj>"; exit 1; }
RUN_DIR_Lux=$JOB_DIR bash "$JOB_DIR/setup_lux.sh"
mkdir -p "$RINGX_RUNS_DIR"
echo "ready: scripts/bench.sh"
