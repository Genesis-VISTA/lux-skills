#!/bin/bash
# Runs on the Lux LOGIN node before every sbatch (VISTA), or from scripts/muse.sh by hand. Preflight only:
# checks that the Muse weights, the image and the engine are readable, changes nothing, takes seconds. A failure
# here stops the submission with this output instead of a job that dies after the queue wait. It does not see
# script_args, so a MODELPATH= or BUILD_IMAGE=1 submission is checked again by the job itself.
#   VERIFY=1   also prints the image's sha256 (~5 min)
set -u
: "${RUN_DIR_Lux:?RUN_DIR_Lux not set}"
E=$RUN_DIR_Lux/lux-inference
M=${LUX_MODELPATH:-}
ok=1; say() { printf '%-8s %s\n' "$1" "$2"; }
howto() {   # the walkthrough, also in ../../MUSE_SPARK.md section 3
  say ""    "To download a Muse checkpoint into your own directory (login node, once):"
  say ""    "  1. bf16 only: request access at https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B (Meta's"
  say ""    "     form; the Muse Model License), then make a read token at https://huggingface.co/settings/tokens"
  say ""    "  2. export https_proxy=http://proxy.ccs.ornl.gov:3128 http_proxy=http://proxy.ccs.ornl.gov:3128"
  say ""    "     python3 -m pip install --user 'huggingface_hub[cli]==1.33.0' && hf auth login   # bf16 only"
  say ""    "  3. scripts/muse.sh download /lustre/orion/<proj>/scratch/\$USER/hfmodels mxfp4|bf16   (checkpoint + DSpark draft)"
  say ""    "  4. submit with script_args=\"MODELPATH=/lustre/orion/<proj>/scratch/\$USER/hfmodels ...\""
}
found=0
for m in Muse-Spark-Open-816B-A42B-Quark-MXFP4 Muse-Spark-Open-816B-A42B-Quark-MXFP4-r63 Muse-Spark-Open-816B-A42B; do
  [ -r "$M/$m/config.json" ] && { say model "$M/$m"; found=1; }
done
[ "$found" = 1 ] || { say model "MISSING: no Muse checkpoint (<model>/config.json) in ${M:-<LUX_MODELPATH unset>}"; howto; ok=0; }
if [ -f "$M/Muse-Spark-Open-DSpark/.verified" ]; then say draft "$M/Muse-Spark-Open-DSpark (verified): SPEC=auto uses it"
else say draft "none in ${M:-<unset>}: SPEC=auto serves without speculative decoding (slower). The drafts are public;"
     say ""    "  the owner of $M adds them with: scripts/muse.sh download $M drafts"; fi
T=${LUX_IMAGE_TAR:-}
if [ -r "$T" ] && [ -s "$T" ]; then
  say image "$T"
  [ "${VERIFY:-0}" = 1 ] && say sha256 "$(sha256sum "$T" | cut -d' ' -f1)"
elif [ -s "${VISTA_JOB_DIR:-/nonexistent}/images/inference_image_gfx950_musemain2.tar" ]; then
  say image "$VISTA_JOB_DIR/images/inference_image_gfx950_musemain2.tar (built by an earlier BUILD_IMAGE=1 run)"
else
  say image "MISSING: ${T:-<LUX_IMAGE_TAR unset>}; a submission with BUILD_IMAGE=1 builds one (~1 h)"
fi
[ -s "$E/docker/vllm_src/vllm_musemain_0e939af8.bundle" ] && say engine "zenith v0.1.9 engine with the vLLM musemain bundle" \
  || { say engine "MISSING: $E/docker/vllm_src/vllm_musemain_0e939af8.bundle"; ok=0; }
D=${LUX_SCIENCE_DATA:-${LUX_DATASETSPATH:-}/science}
[ -r "$D/gpqa_diamond.jsonl" ] && say data "$D (science suite)" || say data "MISSING: $D (needed only for BENCH=science)"
[ "$ok" = 1 ] && echo "setup OK" || { echo "setup FAILED: fix the MISSING lines above"; exit 1; }
