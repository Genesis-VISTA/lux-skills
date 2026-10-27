#!/usr/bin/env bash
# Site layer for the RingX benchmark job on OLCF Lux.   source env.lux.sh
# Every value is overridable from the environment before sourcing.
export SKILL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
export JOB_DIR=${JOB_DIR:-$SKILL_DIR/hpc_jobs/ringx-bench}     # the VISTA job folder; the hand-run scripts submit it too

# Slurm account: first one the user has (Lux rejects jobs without -A).
export ACCOUNT=${ACCOUNT:-$(sacctmgr -nP show assoc user="$USER" format=Account 2>/dev/null | grep -v '^$' | head -1 | tr -d '|')}
export PARTITION=${PARTITION:-batch}

# The Primus image (torch, FlashAttention 2.8.3, RCCL), shared with primus-skill. Only read.
export SIF=${SIF:-/lustre/orion/world-shared/stf218/junqi/primus/primus-v26.5.sif}
export RINGX_ROOT=${RINGX_ROOT:-$SKILL_DIR/work/ringx-bench}   # RingX clone + ring-flash-attn, from setup_lux.sh
export RINGX_RUNS_DIR=${RINGX_RUNS_DIR:-$SKILL_DIR/runs}        # per-job outputs: runs/ringx.<jobid>/

# Lux has NO direct egress: setup_lux.sh clones from GitHub and installs from PyPI through the proxy.
export http_proxy=${http_proxy:-http://proxy.ccs.ornl.gov:3128} https_proxy=${https_proxy:-http://proxy.ccs.ornl.gov:3128}
export no_proxy=${no_proxy:-localhost,127.0.0.1,.ornl.gov,.ccs.ornl.gov}

[ -n "$ACCOUNT" ] || echo "env.lux.sh: could not detect a Slurm account; export ACCOUNT=<proj>" >&2
