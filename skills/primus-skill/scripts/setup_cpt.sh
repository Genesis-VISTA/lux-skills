#!/usr/bin/env bash
# One-time setup for Mistral-Large-123B continued pretraining (login node, ~2 min). Idempotent.
#   scripts/setup_cpt.sh [work dir]        (default $CPT_WORKDIR)
# Runs the shared package's tutorials/03_mistral_cpt/setup_workdir.sh: the patched Primus source (Primus.tar),
# the tutorial's lux/ and replicate/ scripts, mistral_common and prebuilt JIT caches go into the work dir,
# which every container mounts as /work. Models, data and the image stay in $PKG and are only read.
set -eu
SK=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
source "$SK/env.lux.sh"
W=${1:-$CPT_WORKDIR}
T3=$PKG/tutorials/03_mistral_cpt
[ -x "$T3/setup_workdir.sh" ] || { echo "ERROR: no tutorial 3 at $T3; set PKG" >&2; exit 1; }
[ -r "$CPT_SIF" ] || { echo "ERROR: cannot read the image $CPT_SIF; set CPT_SIF" >&2; exit 1; }
if [ -d "$W/Primus" ] && [ -f "$W/lux/lux_env.sh" ]; then
  echo "already set up: $W"
else
  "$T3/setup_workdir.sh" "$W"
fi
echo "setup OK: work dir $W, package $PKG, image $CPT_SIF"
[ "$W" = "$CPT_WORKDIR" ] || echo "note: export CPT_WORKDIR=$W before scripts/run_cpt.sh"
