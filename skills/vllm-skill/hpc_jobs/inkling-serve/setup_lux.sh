#!/bin/bash
# Runs on the Lux LOGIN node before every sbatch (VISTA), or from scripts/setup.sh by hand. Preflight only:
# checks that the shared package and the engine are readable, changes nothing, takes seconds. A failure here
# stops the submission with this output instead of a job that dies after the queue wait.
#   VERIFY=1   also checks the 57 GB image's sha256 (~5 min)
set -u
: "${RUN_DIR_Lux:?RUN_DIR_Lux not set}"
E=$RUN_DIR_Lux/lux-inference
ok=1; say() { printf '%-8s %s\n' "$1" "$2"; }
IMAGE_SHA=e03d16939f5e1b4b3cc7aa4e783246f666ca1350b3c5687faa3d6c9ce8f40713   # inference_image_gfx950_v8mi250_20261011.tar
[ -r "${LUX_IMAGE_TAR:-}" ] && [ -s "$LUX_IMAGE_TAR" ] && say image "$LUX_IMAGE_TAR" \
  || { say image "MISSING or unreadable: LUX_IMAGE_TAR=${LUX_IMAGE_TAR:-<unset>}"; ok=0; }
# The model is a script_arg, which setup does not see: check every model the package holds, require the default.
[ -r "${LUX_MODELPATH:-}/Inkling-MXFP4/config.json" ] && say model "$LUX_MODELPATH/Inkling-MXFP4" \
  || { say model "MISSING or unreadable: ${LUX_MODELPATH:-<LUX_MODELPATH unset>}/Inkling-MXFP4/config.json"; ok=0; }
for m in Inkling-bf16 Mistral-Large-Instruct-2411; do
  [ -r "${LUX_MODELPATH:-}/$m/config.json" ] && say model "$LUX_MODELPATH/$m (also available)"
done
D=${LUX_SCIENCE_DATA:-${LUX_DATASETSPATH:-}/science}
for f in gpqa_diamond mascqa mmlu_pro_sci; do
  [ -r "$D/$f.jsonl" ] || { say data "MISSING: $D/$f.jsonl (needed only for BENCH=science)"; }
done
[ -r "$D/gpqa_diamond.jsonl" ] && say data "$D (science suite)"
[ -s "$E/docker/patches/anp/librccl-anp.so" ] && [ -s "$E/docker/patches/anp/libionic.so.1.1.61" ] && say fabric "RCCL-ANP plugin present" \
  || { say fabric "MISSING: $E/docker/patches/anp/{librccl-anp.so,libionic.so.1.1.61} (cross-node RCCL would fall back to sockets)"; ok=0; }
if [ "${VERIFY:-0}" = 1 ] && [ -r "${LUX_IMAGE_TAR:-}" ]; then
  got=$(sha256sum "$LUX_IMAGE_TAR" | cut -d' ' -f1)
  [ "$got" = "$IMAGE_SHA" ] && say sha256 "image OK" || { say sha256 "MISMATCH: $got (want $IMAGE_SHA)"; ok=0; }
fi
[ "$ok" = 1 ] && echo "setup OK" || { echo "setup FAILED: fix the MISSING lines above"; exit 1; }
