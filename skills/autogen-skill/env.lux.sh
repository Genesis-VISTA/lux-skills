#!/usr/bin/env bash
# Site layer for the ICF 4-agent campaigns on OLCF Lux.   source env.lux.sh
# Every value is overridable from the environment before sourcing.
export SKILL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
export JOB_DIR=${JOB_DIR:-$SKILL_DIR/hpc_jobs/icf-campaign}     # the VISTA job folder; the hand-run scripts submit it too

# Slurm account: first one the user has (Lux rejects jobs without -A).
export ACCOUNT=${ACCOUNT:-$(sacctmgr -nP show assoc user="$USER" format=Account 2>/dev/null | grep -v '^$' | head -1 | tr -d '|')}
export PARTITION=${PARTITION:-batch}

# The read-only hackathon package: the gpt-oss-20b model and the vLLM image for BACKEND=self. Read in place.
export PKG=${PKG:-/lustre/orion/world-shared/stf218/junqi/Genesis-Hackathon}
export ICF_GPTOSS_MODEL=${ICF_GPTOSS_MODEL:-$PKG/models/gptoss-20b-hedp.03302026}
export ICF_VLLM_IMAGE_TAR=${ICF_VLLM_IMAGE_TAR:-$PKG/containers/inference_image_gfx950_v8mi250_20261011.tar}

export ICF_VENV_CACHE=${ICF_VENV_CACHE:-$SKILL_DIR/work}       # venv-<pins>.tar: the app's venv, built by the first job
export ICF_RUNS_DIR=${ICF_RUNS_DIR:-$SKILL_DIR/runs}            # per-job outputs: runs/icf.<jobid>/

[ -n "$ACCOUNT" ] || echo "env.lux.sh: could not detect a Slurm account; export ACCOUNT=<proj>" >&2
