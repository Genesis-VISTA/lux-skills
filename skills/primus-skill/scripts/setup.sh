#!/usr/bin/env bash
# One-time setup by hand (login node, a few minutes). Idempotent.   source env.lux.sh && scripts/setup.sh
# The work is in the job folder's setup_lux.sh, which VISTA also runs before every submission.
set -eu
: "${PRIMUS_ROOT:?source env.lux.sh first}"
SK=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
RUN_DIR_Lux=$SK/hpc_jobs/primus-smoke exec bash "$SK/hpc_jobs/primus-smoke/setup_lux.sh"
