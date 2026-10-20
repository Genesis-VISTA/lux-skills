#!/usr/bin/env bash
# Primus setup on the Lux LOGIN node (network via the proxy). Idempotent, so it runs on every submit.
#   VISTA: inlined by the Lux dispatcher before sbatch; a failure fails the submit call.
#   By hand: source env.lux.sh && scripts/setup.sh
# Clones Primus release/v26.5, pins Megatron-LM, builds helpers_cpp inside the image. If the image is
# missing, it only checks that it can be created; job.lux.slurm builds it (and helpers_cpp) on a compute node.
#
# Env (VISTA exports RUN_DIR_Lux, VISTA_JOB_DIR, the proxy, and cluster_defaults.json -> lux.iri.environment):
#   RUN_DIR_Lux  this folder's uploaded files (build_helpers.sh)
#   PRIMUS_ROOT  default $VISTA_JOB_DIR (shared by this job's runs); env.lux.sh sets it by hand
#   PRIMUS_DIR   default $PRIMUS_ROOT/Primus
#   SIF          the shared 8.1 GB image (cluster_defaults.json); USER_SIF the user's own fallback, see below
set -eu
PRIMUS_ROOT=${PRIMUS_ROOT:-${VISTA_JOB_DIR:?set PRIMUS_ROOT, or source env.lux.sh}}
PRIMUS_DIR=${PRIMUS_DIR:-$PRIMUS_ROOT/Primus}
SIF=${SIF:-}
LUSTRE_ROOT=${LUSTRE_ROOT:-/lustre/orion}
MEGATRON_PIN=${MEGATRON_PIN:-d3528a21301db2d12e92912b3ec025dc8a2ed4d6}   # what the image's Primus was validated with

: "${RUN_DIR_Lux:?RUN_DIR_Lux not set}"
# Image: the shared $SIF if readable, else this user's own copy (USER_SIF, built by the job if missing).
# USER_SIF defaults to the job's own working dir, beside the uploaded sources; the shared one is never written.
USER_SIF=${USER_SIF:-$(dirname "$RUN_DIR_Lux")/images/primus-v26.5.sif}
[ -s "$SIF" ] && [ -r "$SIF" ] || SIF=$USER_SIF
if [ ! -s "$SIF" ]; then
  mkdir -p "$(dirname "$SIF")" 2>/dev/null && [ -w "$(dirname "$SIF")" ] || {
    echo "ERROR: no readable shared image, and $(dirname "$SIF") is not writable for your own copy. Set USER_SIF to a writable path." >&2; exit 1; }
  echo "no readable shared image: the job will build your own at $SIF first (~30 min; submit with a duration of at least 1:00:00)"
fi
mkdir -p "$(dirname "$PRIMUS_DIR")"
# Two submits at once would race on the clone and the build.
exec 9>"$PRIMUS_ROOT/.setup.lock"; flock 9

if [ -e "$PRIMUS_DIR" ] && ! git -C "$PRIMUS_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  echo "ERROR: $PRIMUS_DIR exists but is not a usable git clone. Not touching it; set PRIMUS_ROOT/PRIMUS_DIR elsewhere." >&2; exit 1
fi
if [ ! -e "$PRIMUS_DIR" ]; then
  git clone --branch release/v26.5 https://github.com/AMD-AGI/Primus.git "$PRIMUS_DIR"
fi
MEG=$PRIMUS_DIR/third_party/Megatron-LM
if [ "$(git -C "$MEG" rev-parse HEAD 2>/dev/null)" != "$MEGATRON_PIN" ]; then
  # GitHub through the proxy resets connections now and then; retry the submodule.
  for i in 1 2 3; do git -C "$PRIMUS_DIR" submodule update --init --recursive third_party/Megatron-LM && break || sleep 5; done
  [ "$(git -C "$MEG" rev-parse HEAD)" = "$MEGATRON_PIN" ] || {
    git -C "$MEG" fetch --depth 50 origin "$MEGATRON_PIN" 2>/dev/null || git -C "$MEG" fetch origin
    git -C "$MEG" checkout --detach "$MEGATRON_PIN"; }
fi
[ -s "$SIF" ] && PRIMUS_DIR=$PRIMUS_DIR SIF=$SIF LUSTRE_ROOT=$LUSTRE_ROOT bash "$RUN_DIR_Lux/build_helpers.sh"
echo "setup OK: Primus $(git -C "$PRIMUS_DIR" rev-parse --short HEAD), Megatron ${MEGATRON_PIN:0:9}, image $SIF"
