#!/bin/bash
# Download the Inkling checkpoints from Hugging Face into $MODELS (default ~/hfmodels), retrying until complete.
# MXFP4 (~566 GB) first, then bf16 (~1.9 TB). The launcher serves /app/models/<ZENITH_MODEL>, so the directory
# names must stay Inkling-MXFP4 / Inkling-bf16 (or pass your own name as ZENITH_MODEL). small fetches
# Inkling-Small in bf16 (~532 GB) as Inkling-Small-bf16; it is not part of `all`. mistral fetches
# Mistral-Large-Instruct-2411 (~224 GB of Hugging Face shards; its duplicate consolidated.safetensors are skipped)
# for docs/LUX_MISTRAL_LARGE.md. That repo is gated: accept the Mistral Research Licence on its page first.
#
#   MODELS=/path/to/hfmodels bash download_models.sh [mxfp4|bf16|all|small|mistral]
#
# Needs the `hf` CLI (pip install -U "huggingface_hub[cli]" hf_xet). If a repo is gated for you, export your own
# Hugging Face token as HF_TOKEN in this shell first (`hf` reads it from the environment; nothing is stored here).
set -u
MODELS="${MODELS:-${HOME}/hfmodels}"
LOGDIR="${LOGDIR:-${MODELS}/logs}"
mkdir -p "${MODELS}" "${LOGDIR}"
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_XET_HIGH_PERFORMANCE=1

dl () {  # repo, dest, log, extra hf args
  local repo="$1" dest="$2" log="$3"; shift 3
  for attempt in $(seq 1 200); do
    echo "=== attempt ${attempt} $(date -Is) ===" >> "${log}"
    hf download "${repo}" --local-dir "${dest}" "$@" >> "${log}" 2>&1 && {
        echo "=== COMPLETE $(date -Is) ===" >> "${log}"; return 0; }
    echo "=== retry in 30s ===" >> "${log}"; sleep 30
  done
  echo "=== GAVE UP $(date -Is) ===" >> "${log}"; return 1
}

case "${1:-all}" in
  mxfp4) dl lightseekorg/Inkling-MXFP4 "${MODELS}/Inkling-MXFP4" "${LOGDIR}/dl_mxfp4.log" ;;
  bf16)  dl thinkingmachines/Inkling   "${MODELS}/Inkling-bf16"  "${LOGDIR}/dl_bf16.log" ;;
  small) dl thinkingmachines/Inkling-Small "${MODELS}/Inkling-Small-bf16" "${LOGDIR}/dl_small_bf16.log" ;;
  mistral) dl mistralai/Mistral-Large-Instruct-2411 "${MODELS}/Mistral-Large-Instruct-2411" \
             "${LOGDIR}/dl_mistral_large.log" --exclude "consolidated*" ;;
  all)   dl lightseekorg/Inkling-MXFP4 "${MODELS}/Inkling-MXFP4" "${LOGDIR}/dl_mxfp4.log"
         dl thinkingmachines/Inkling   "${MODELS}/Inkling-bf16"  "${LOGDIR}/dl_bf16.log" ;;
  *) echo "usage: download_models.sh [mxfp4|bf16|all|small|mistral]" >&2; exit 1 ;;
esac
