#!/usr/bin/env bash
# Build Megatron's helpers_cpp in the image (it must match the container's Python 3.12 ABI).
# Skips if already built. Runs from setup_lux.sh (login node) or job.lux.slurm (after an image build).
#   PRIMUS_DIR, SIF; LUSTRE_ROOT default /lustre/orion
set -eu
: "${PRIMUS_DIR:?}" "${SIF:?}"
MEG=$PRIMUS_DIR/third_party/Megatron-LM
ls "$MEG"/megatron/core/datasets/helpers_cpp*.so >/dev/null 2>&1 && exit 0
mkdir -p /tmp/$USER/primus
apptainer exec --rocm --bind "${LUSTRE_ROOT:-/lustre/orion}:${LUSTRE_ROOT:-/lustre/orion}" --bind "$PRIMUS_DIR:/work/Primus" \
  --pwd /work/Primus/third_party/Megatron-LM/megatron/core/datasets "$SIF" make 2>&1 | tail -2
ls "$MEG"/megatron/core/datasets/helpers_cpp*.so >/dev/null
