#!/bin/bash
# Runs on the Lux LOGIN node before every sbatch (VISTA), or from scripts/setup_cpt.sh by hand.
# Makes the tutorial 3 work dir once (~2 min): the patched Primus tree, the tutorial's lux/ scripts and the
# prebuilt JIT caches, copied from the read-only package. Every run of the job shares it, and checkpoints
# (224 GB each) are written under it. Later submissions only check it.
set -euo pipefail
: "${PKG:?PKG not set (cluster_defaults.json or env.lux.sh)}"
CPT_WORKDIR=${CPT_WORKDIR:-${VISTA_JOB_DIR:?set CPT_WORKDIR or VISTA_JOB_DIR}/work}
CPT_SIF=${CPT_SIF:-$PKG/containers/primus_v26.5.sif}
T3=$PKG/tutorials/03_mistral_cpt
[ -x "$T3/setup_workdir.sh" ] || { echo "ERROR: no tutorial 3 scripts at $T3 (PKG=$PKG)"; exit 1; }
[ -r "$CPT_SIF" ] || { echo "ERROR: cannot read the image $CPT_SIF"; exit 1; }
[ -r "$PKG/models/mistral_large_megatron_tp1pp8/latest_checkpointed_iteration.txt" ] \
  || { echo "ERROR: cannot read the base checkpoint in $PKG/models/mistral_large_megatron_tp1pp8"; exit 1; }
mkdir -p "$(dirname "$CPT_WORKDIR")"
exec 9>"$CPT_WORKDIR.lock"; flock 9          # two submissions must not unpack into the same dir
if [ -d "$CPT_WORKDIR/Primus" ] && [ -f "$CPT_WORKDIR/lux/lux_env.sh" ]; then
  echo "work dir ready: $CPT_WORKDIR"
else
  [ -e "$CPT_WORKDIR" ] && { echo "ERROR: $CPT_WORKDIR exists but is not a tutorial 3 work dir; remove it and resubmit"; exit 1; }
  # Build beside it and rename into place, so a failed copy never leaves a half-made work dir behind.
  tmp=$CPT_WORKDIR.partial
  rm -rf "$tmp"
  if ! "$T3/setup_workdir.sh" "$tmp" > "$tmp.log" 2>&1; then
    tail -5 "$tmp.log"; rm -rf "$tmp"; echo "ERROR: copying the work dir failed (above); resubmit"; exit 1
  fi
  mv "$tmp" "$CPT_WORKDIR" && rm -f "$tmp.log"
  echo "work dir created: $CPT_WORKDIR"
fi
