#!/usr/bin/env bash
# Site layer for serving vLLM on OLCF Lux.   source env.lux.sh
# Every value is overridable from the environment before sourcing.
export SKILL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
export JOB_DIR=${JOB_DIR:-$SKILL_DIR/hpc_jobs/inkling-serve}     # the VISTA job folder; the hand-run scripts submit it too
export ENGINE_DIR=${ENGINE_DIR:-$JOB_DIR/lux-inference}       # the serving engine (vLLM launch scripts, patches, bench)
export MUSE_JOB_DIR=${MUSE_JOB_DIR:-$SKILL_DIR/hpc_jobs/muse-serve}  # the Muse job folder (zenith v0.1.9 engine); scripts/muse.sh

# Slurm account: first one the user has (Lux rejects jobs without -A).
export ACCOUNT=${ACCOUNT:-$(sacctmgr -nP show assoc user="$USER" format=Account 2>/dev/null | grep -v '^$' | head -1 | tr -d '|')}
export PARTITION=${PARTITION:-batch}

# The read-only hackathon package: models, the vLLM image, benchmark data. Read in place.
# World-readable copy of Inkling-MXFP4, the v8 image and the science benchmarks (same layout as the full package).
export PKG=${PKG:-/lustre/orion/world-shared/stf218/junqi/Genesis-Hackathon}

# Engine knobs. LUX_<KNOB> is the Lux name of the engine's ZENITH_<KNOB> (the image's baked-in
# patches read the ZENITH_ names); serve_lux.slurm.sh copies every LUX_* onto its ZENITH_* twin.
export LUX_MODELPATH=${LUX_MODELPATH:-$PKG/models}                      # dir holding Inkling-MXFP4/ etc.
export LUX_IMAGE_TAR=${LUX_IMAGE_TAR:-$PKG/containers/inference_image_gfx950_v8mi250_20261011.tar}
export LUX_DATASETSPATH=${LUX_DATASETSPATH:-$PKG/datasets/benchmarks}   # science/ holds the suite's 3 JSONL files
export LUX_RUNS_DIR=${LUX_RUNS_DIR:-$SKILL_DIR/runs}                    # per-job server.log, results
# Muse Spark Open (scripts/muse.sh): the package also holds the MXFP4 checkpoint and the musemain2 image.
export MUSE_MODELPATH=${MUSE_MODELPATH:-$PKG/models}                    # holds Muse-Spark-Open-816B-A42B-Quark-MXFP4/
export MUSE_IMAGE_TAR=${MUSE_IMAGE_TAR:-$PKG/containers/inference_image_gfx950_musemain2.tar}

[ -n "$ACCOUNT" ] || echo "env.lux.sh: could not detect a Slurm account; export ACCOUNT=<proj>" >&2
[ -d "$PKG" ] || echo "env.lux.sh: no hackathon package at $PKG; export PKG=<path> (or LUX_MODELPATH / LUX_IMAGE_TAR)" >&2
