#!/usr/bin/env bash
# VISTA runs this on the Lux login node before sbatch; scripts/setup.sh uses it by hand.
set -euo pipefail
: "${PKG:?set PKG to the read-only hackathon package}"
WORKDIR=${WORKDIR:-${VISTA_JOB_DIR:?set WORKDIR or VISTA_JOB_DIR}/work}
WORKDIR=${WORKDIR%/}
case $WORKDIR in /lustre/orion/*) ;; *) echo "ERROR: WORKDIR must be on Orion Lustre: $WORKDIR" >&2; exit 2 ;; esac
T4=$PKG/tutorials/04_mistral_rl
IMAGE=${VERL_SIF:-${VERL_V268_SIF:-${CONTAINERS:-$PKG/containers}/verl_v26.8A.sif}}
MODEL_ROOT=${RL_HFMODELS:-${HFMODELS:-$PKG/models}}
MODEL=${RL_MODEL:-$MODEL_ROOT/Mistral-Large-Instruct-2411}
DATA_ROOT=${RL_DATA:-${DATASETS:-$PKG/datasets}/sci_rl}
for f in "$T4/setup_workdir.sh" "$T4/hackathon_env.sh" \
  "$PKG/tutorials/02_zenith_inference/zenith_inference_v8_lux.tgz" \
  "$IMAGE" \
  "$MODEL/config.json" \
  "$DATA_ROOT/rl_control.parquet" \
  "$DATA_ROOT/rl_pubonly.parquet" \
  "$DATA_ROOT/sci_dev.parquet"; do
  [ -r "$f" ] || { echo "ERROR: required input is missing or unreadable: $f" >&2; exit 1; }
done
mkdir -p "$(dirname "$WORKDIR")"
exec 9>"$WORKDIR.lock"
flock 9
if [ -f "$WORKDIR/verl_rl/phase2.sh" ] && [ -f "$WORKDIR/lux/env.sh" ] && [ -f "$WORKDIR/submit.sh" ]; then
  echo "work dir ready: $WORKDIR"
  exit 0
fi
[ ! -e "$WORKDIR" ] || { echo "ERROR: $WORKDIR exists but is not a complete tutorial 4 work dir" >&2; exit 1; }
TMP=$WORKDIR.partial.$$
trap 'rm -rf "$TMP"' EXIT
bash "$T4/setup_workdir.sh" "$TMP"
mv -T "$TMP" "$WORKDIR"
trap - EXIT
echo "work dir created: $WORKDIR"
