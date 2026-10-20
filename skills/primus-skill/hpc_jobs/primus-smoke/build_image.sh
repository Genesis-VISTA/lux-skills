#!/usr/bin/env bash
# Build the Primus apptainer image at $SIF. COMPUTE NODE only (job.lux.slurm runs it with srun):
# it needs ~128 GB RAM and node-local tmp, and takes ~26 min. A login-node build with tmp on Lustre
# did not finish in an hour, and unpacking many small files on Lustre hurts everyone.
#   SIF           where the image goes (its directory must be writable)
#   PRIMUS_IMAGE  source, default rocm/primus:v26.5; set rocm/primus@sha256:<digest> to pin it
set -eu
: "${SIF:?}"
SRC=docker://${PRIMUS_IMAGE:-rocm/primus:v26.5}
TMP=/tmp/$USER/apptainer-build.${SLURM_JOB_ID:-$$}
mkdir -p "$TMP/tmp" "$TMP/cache"
trap 'rm -rf "$TMP"' EXIT
export APPTAINER_TMPDIR=$TMP/tmp APPTAINER_CACHEDIR=$TMP/cache
echo "building $SIF from $SRC on $(hostname) ($(date -u +%H:%M:%SZ))"
apptainer build "$TMP/primus.sif" "$SRC"
# Copy, then rename into place: a reader never sees a partial image, and if another job
# finished first, its image stays and ours is dropped.
cp "$TMP/primus.sif" "$SIF.partial.${SLURM_JOB_ID:-$$}"
mv -n "$SIF.partial.${SLURM_JOB_ID:-$$}" "$SIF"
rm -f "$SIF.partial.${SLURM_JOB_ID:-$$}"
echo "image ready: $SIF ($(du -h "$SIF" | cut -f1), $(date -u +%H:%M:%SZ))"
