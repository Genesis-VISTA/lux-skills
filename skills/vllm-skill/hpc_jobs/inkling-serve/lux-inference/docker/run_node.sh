#!/bin/bash
# Per-node wrapper for running the launcher WITHOUT submit_lux.slurm.sh: it exports the environment the submit
# script would, then runs lux_container_launcher.sh in the foreground. This is how the 2026-10-04 Inkling
# re-validation was run (README.md, "Inkling on 2 x MI355X"), on two nodes that came from two separate Slurm holds.
#
# Run it on EVERY node of the run, with the same RUN_ID and the same ZENITH_HOSTS (head first):
#
#   ZENITH_HOSTS="nodeA nodeB" ZENITH_IMAGE_REF=localhost/inference_image_gfx950:latest \
#     ./run_node.sh ink-mxfp4-2n Inkling-MXFP4            # on nodeA and on nodeB
#
# One host in ZENITH_HOSTS = a 1-node run. Models are read from ${ZENITH_MODELPATH:-$HOME/hfmodels}/<MODEL>.
# Containers are named <ZENITH_CONTAINER_PREFIX>-<host suffix> (default prefix zenith-ink); stop them with
# `podman stop/rm` by that name when done. usage: run_node.sh <RUN_ID> <MODEL>
set -u
[ $# -eq 2 ] || { echo "usage: ZENITH_HOSTS=\"head [worker ...]\" $0 <RUN_ID> <MODEL>" >&2; exit 1; }
[ -n "${ZENITH_HOSTS:-}" ] || { echo "set ZENITH_HOSTS (space-separated, head node first)" >&2; exit 1; }
RUN_ID="$1"; MODEL="$2"
cd "$(dirname "$0")"
export ZENITH_RUN_ID="${RUN_ID}"
export ZENITH_MODEL="${MODEL}"
export ZENITH_HOSTS
export ZENITH_HEADNODE=$(echo ${ZENITH_HOSTS} | awk '{print $1}')
export ZENITH_NNODES=$(echo ${ZENITH_HOSTS} | wc -w)
export ZENITH_GPUS_PER_NODE="${ZENITH_GPUS_PER_NODE:-8}" ZENITH_CPUS_PER_NODE="${ZENITH_CPUS_PER_NODE:-$(nproc)}"
# tuned defaults (Inkling-MXFP4 on 1 and 2 nodes; see the file), before the generic ones below
source ./lux_tuned_defaults.sh "${ZENITH_NNODES}"
export ZENITH_MAX_MODEL_LEN="${ZENITH_MAX_MODEL_LEN:-100000}"
export ZENITH_TARGET_KV_GB="${ZENITH_TARGET_KV_GB:-64}"
export ZENITH_STARTUP_TRIES="${ZENITH_STARTUP_TRIES:-30}"      # 30 s polls for Ray on every node; 90 on a first run (image import)
export ZENITH_MTP="${ZENITH_MTP:-1}" ZENITH_MTP_PP="${ZENITH_MTP_PP:-0}" ZENITH_NO_SERVE="${ZENITH_NO_SERVE:-0}"
export ZENITH_TP="${ZENITH_TP:-}" ZENITH_PP="${ZENITH_PP:-}" ZENITH_DP="${ZENITH_DP:-}" ZENITH_USE_GPUS="${ZENITH_USE_GPUS:-}"
export ZENITH_MAX_NUM_BATCHED_TOKENS="${ZENITH_MAX_NUM_BATCHED_TOKENS:-}" ZENITH_MAX_NUM_SEQS="${ZENITH_MAX_NUM_SEQS:-}"
export ZENITH_MAX_CUDAGRAPH_CAPTURE_SIZE="${ZENITH_MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
export ZENITH_VLLM_LOG_LEVEL="${ZENITH_VLLM_LOG_LEVEL:-INFO}"
export ZENITH_VLLM_EXTRA_ARGS="${ZENITH_VLLM_EXTRA_ARGS:-}" ZENITH_VLLM_EXTRA_ENV="${ZENITH_VLLM_EXTRA_ENV:-}"
export ZENITH_CONTAINER_PREFIX="${ZENITH_CONTAINER_PREFIX:-zenith-ink}"
export ZENITH_IMAGE_REF="${ZENITH_IMAGE_REF:-localhost/inference_image_gfx950:${ZENITH_IMAGE_TAG:-latest}}"
# INFO-level RCCL logging is what proves the transport ("via NET/RCCL-ANP", no NET/Socket); WARN to silence
export ZENITH_NCCL_DEBUG="${ZENITH_NCCL_DEBUG:-INFO}"
export ZENITH_CONTAINER_EXTRA_ENV="${ZENITH_CONTAINER_EXTRA_ENV:-NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET}"
# (the launcher sources lux_podman_env.sh itself)
exec stdbuf -oL bash ./lux_container_launcher.sh "${ZENITH_MODELPATH:-${HOME}/hfmodels}" \
     "${ZENITH_DATASETSPATH:-${HOME}/datasets}" 2>&1 | grep --line-buffered -v PlacementGroupCleaner
