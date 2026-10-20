#!/usr/bin/env bash
# Preflight (login node, seconds; --verify adds the 57 GB image checksum, ~5 min). Idempotent, changes nothing.
#   source env.lux.sh && scripts/setup.sh [--verify]
# The checks are the job folder's setup_lux.sh, which VISTA also runs before every submission; this adds the
# account and the login node's python, which the hand-run scripts need.
set -eu
: "${JOB_DIR:?source env.lux.sh first}"
ok=1; say() { printf '%-8s %s\n' "$1" "$2"; }
[ -n "$ACCOUNT" ] && say account "$ACCOUNT" || { say account "MISSING: export ACCOUNT=<proj>"; ok=0; }
PY=${PYTHON:-$(for p in python3.12 python3.11 python3.10 python3.9 python3; do command -v $p >/dev/null && { echo $p; break; }; done)}
$PY -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null && say python "$PY ($($PY -V 2>&1); the bench clients need >= 3.9)" \
  || { say python "MISSING python >= 3.9 for scripts/science.sh (set PYTHON=<interpreter>)"; ok=0; }
[ "${1:-}" = --verify ] && export VERIFY=1
RUN_DIR_Lux=$JOB_DIR bash "$JOB_DIR/setup_lux.sh" || ok=0
mkdir -p "$LUX_RUNS_DIR"
[ "$ok" = 1 ] && echo "ready: scripts/serve.sh to serve, scripts/bench.sh to score" || { echo "setup: fix the MISSING lines above"; exit 1; }
