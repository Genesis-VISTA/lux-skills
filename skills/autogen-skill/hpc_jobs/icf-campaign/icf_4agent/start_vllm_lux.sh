#!/bin/bash
# Serve the ICF agents' shared model with vLLM on ONE MI355X of the lux compute node this runs on.
# Run it ON the compute node (the icf.vllm holding job does); the app then talks to localhost:$VLLM_PORT.
#
# Uses the gfx950 zenith inference image under rootless podman. lux has no subuid/subgid and
# does not create /run/user, so podman gets a private XDG_RUNTIME_DIR, a node-local graph root
# and fuse-overlayfs with ignore_chown_errors. The graph root is /scratch/$USER, or /tmp/$USER on
# the diskless nodes that have no /scratch -- RAM-backed and empty at every job start, so there
# the 61 GB image is loaded again each time (~2 min from Lustre).
#
# The weights and the image tar default to your own Lustre scratch (SCRATCH_ROOT below); point
# ICF_VLLM_MODEL_PATH / ICF_VLLM_IMAGE_TAR elsewhere if your copies live somewhere else.
#
# ICF_VLLM_PREPARE_ONLY=1 sets up podman and loads the image, then exits without serving.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
SCRATCH_ROOT="/lustre/orion/ven136/scratch/${USER}"
MODEL_PATH="${ICF_VLLM_MODEL_PATH:-${SCRATCH_ROOT}/hfmodels/gptoss-20b-hedp.03302026}"
MODEL_NAME="${ICF_VLLM_MODEL_NAME:-gptoss-20b-hedp}"
# No --chat-template by default. For a gpt_oss model this vLLM renders chat requests with its
# built-in harmony encoder and ignores --chat-template (chatml_simple.jinja gave the same
# prompt token counts), then splits the reply into content / reasoning_content. Set e.g.
# ICF_VLLM_CHAT_TEMPLATE=${SCRIPT_DIR}/chatml_simple.jinja to pass one anyway.
CHAT_TEMPLATE="${ICF_VLLM_CHAT_TEMPLATE:-}"
IMAGE_TAR="${ICF_VLLM_IMAGE_TAR:-${SCRATCH_ROOT}/zenith/images/inference_image_gfx950.tar}"
IMAGE_REF="${ICF_VLLM_IMAGE_REF:-localhost/inference_image_gfx950:latest}"
VLLM_PORT="${ICF_VLLM_PORT:-8000}"
VLLM_GPU="${ICF_VLLM_GPU:-0}"          # which MI355X of the node (0-7); vLLM gets only this one
MAX_MODEL_LEN="${ICF_VLLM_MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${ICF_VLLM_GPU_MEM_UTIL:-0.90}"
TIKTOKEN_DIR="${ICF_VLLM_TIKTOKEN_DIR:-${SCRIPT_DIR}/tiktoken}"
CNAME="icf-vllm-$(hostname -s)-${VLLM_PORT}"

# --- rootless podman environment ---
_uid="$(id -u)"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/podman-run-${_uid}}"
mkdir -p "${XDG_RUNTIME_DIR}/libpod/tmp" && chmod 700 "${XDG_RUNTIME_DIR}"
if [ -d /scratch ] && [ -w /scratch ]; then STORE_BASE="/scratch/$(whoami)"; else STORE_BASE="/tmp/$(whoami)"; fi
GRAPHROOT="${STORE_BASE}/containers/storage"
PTMPDIR="${STORE_BASE}/containers/tmp"
RUNROOT="${XDG_RUNTIME_DIR}/containers"
# Rootless podman's default network config dir is ~/.config/cni/net.d on NFS, and every
# podman command takes an NFS lock on cni.lock there; on lux that lock can hang forever
# (process stuck in nlmclnt_wait). The server uses --network=host, so keep it node-local.
NETCONF="${STORE_BASE}/containers/net.d"
mkdir -p "${GRAPHROOT}" "${PTMPDIR}" "${RUNROOT}" "${NETCONF}"
export TMPDIR="${PTMPDIR}"
PODMAN="podman --root ${GRAPHROOT} --runroot ${RUNROOT} --tmpdir ${PTMPDIR} \
 --network-config-dir ${NETCONF} \
 --cgroup-manager cgroupfs --storage-driver overlay \
 --storage-opt mount_program=/usr/bin/fuse-overlayfs --storage-opt ignore_chown_errors=true"

# Stale state from a killed job makes every podman command fail. The fix is
# `podman system migrate`, which stops ALL of this user's containers on the node, so never
# run it from here: report and stop.
if ! ${PODMAN} ps >/dev/null 2>&1; then
    echo "ERROR: podman is in an invalid internal state on $(hostname -s) (stale state from a" \
         "previous job). Not running 'podman system migrate' (it stops every container of yours" \
         "on the node); run it by hand once nothing else of yours is running there."
    ${PODMAN} ps 2>&1 | tail -3
    exit 1
fi

if ! ${PODMAN} image exists "${IMAGE_REF}"; then
    echo "Loading ${IMAGE_TAR} into ${GRAPHROOT} (first use on this node, several minutes) ..."
    ${PODMAN} load -i "${IMAGE_TAR}"
    ${PODMAN} image exists "${IMAGE_REF}" || {
        echo "ERROR: ${IMAGE_TAR} did not provide ${IMAGE_REF}; set ICF_VLLM_IMAGE_REF to the loaded name:"
        ${PODMAN} images
        exit 1
    }
fi
[ "${ICF_VLLM_PREPARE_ONLY:-0}" = 1 ] && { echo "Image ${IMAGE_REF} ready on $(hostname -s)"; exit 0; }

# --- GPU pinning: vLLM sees only GPU $VLLM_GPU ---
# HIP_VISIBLE_DEVICES (+ the CUDA_ alias vLLM insists matches it) and nothing else; the image
# sets none of the *_VISIBLE_DEVICES itself. Not ROCR_VISIBLE_DEVICES: the Ray that vLLM imports
# even for a single GPU refuses it ("Please use HIP_VISIBLE_DEVICES instead of
# ROCR_VISIBLE_DEVICES"), and stacking it under HIP_ would double-filter (ROCR re-numbers the
# survivors from 0, so HIP=N then names a device that no longer exists for N > 0). Handing the
# container only that GPU's /dev/dri nodes does not work either: HIP still counts 8 devices
# and torch fails with "No CUDA GPUs are available".
# HIP numbers GPUs in PCI bus order, the same order rocm-smi/amd-smi use (checked on lux014).
NUM_GPUS=0
for _p in /sys/class/kfd/kfd/topology/nodes/*/properties; do
    [ "$(awk '/^simd_count/{print $2}' "${_p}")" != 0 ] && NUM_GPUS=$((NUM_GPUS + 1))
done
[ "${VLLM_GPU}" -ge 0 ] 2>/dev/null && [ "${VLLM_GPU}" -lt "${NUM_GPUS}" ] || {
    echo "ERROR: no GPU ${VLLM_GPU} on $(hostname -s) (${NUM_GPUS} GPUs)"; exit 1; }

# The gpt-oss (harmony) prompt renderer needs the o200k_base tiktoken vocab, which it otherwise
# downloads at first use; lux compute nodes have no route to it. ./tiktoken holds a copy.
TIKTOKEN_ARGS=""
if [ -f "${TIKTOKEN_DIR}/o200k_base.tiktoken" ]; then
    TIKTOKEN_ARGS="-v ${TIKTOKEN_DIR}:/opt/tiktoken:ro --env TIKTOKEN_ENCODINGS_BASE=/opt/tiktoken"
fi

CHAT_TEMPLATE_MOUNT=""
CHAT_TEMPLATE_ARG=""
if [ -n "${CHAT_TEMPLATE}" ]; then
    CHAT_TEMPLATE_MOUNT="-v ${CHAT_TEMPLATE}:/app/chat_template.jinja:ro"
    CHAT_TEMPLATE_ARG="--chat-template /app/chat_template.jinja"
fi

echo "Starting vLLM server on $(hostname -s)"
echo "  Model:  ${MODEL_PATH}"
echo "  Name:   ${MODEL_NAME}"
echo "  GPU:    ${VLLM_GPU} (of ${NUM_GPUS} on this node)"
echo "  Port:   ${VLLM_PORT} (localhost only)"
echo "  Image:  ${IMAGE_REF}"
echo "  Chat template: ${CHAT_TEMPLATE:-model default}"
echo ""

${PODMAN} rm -f "${CNAME}" >/dev/null 2>&1 || true

# Library fix-ups the zenith launcher applies to this image before starting anything
# (run_container.sh SMI_FIXUP / KPACK_FIXUP / PATCH_FIXUP).
FIXUPS='
ROCMLIB="$(ls -d /opt/rocm*/lib 2>/dev/null | head -1)"
CORESMI="$(ls /opt/venv/lib/python*/site-packages/_rocm_sdk_core/lib/libamd_smi.so.* 2>/dev/null | head -1)"
if [ -n "$ROCMLIB" ] && [ -n "$CORESMI" ] && ! readlink -f "$ROCMLIB/libamd_smi.so" 2>/dev/null | grep -q _rocm_sdk_core; then
    ln -sf "$CORESMI" "$ROCMLIB/libamd_smi.so"
fi
for k in /opt/venv/lib/python*/site-packages/*/.kpack/*:*.kpack; do
    [ -e "$k" ] && [ -e "${k%%:*}.kpack" ] || continue
    mkdir -p /tmp/kpack_shadowed && mv "$k" /tmp/kpack_shadowed/
done
for _p in /opt/zenith_patches/*.py; do
    [ -f "$_p" ] || continue
    python "$_p" >/dev/null || echo "icf: WARNING -- patch $_p did not apply"
done
'

exec ${PODMAN} run --rm \
    --name "${CNAME}" \
    --log-level=error \
    --network=host \
    --ipc=host \
    --group-add=video --group-add=render \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd --device /dev/dri \
    --env HIP_VISIBLE_DEVICES="${VLLM_GPU}" \
    --env CUDA_VISIBLE_DEVICES="${VLLM_GPU}" \
    --env HSA_XNACK=0 \
    ${TIKTOKEN_ARGS} \
    -v "$(dirname "${MODEL_PATH}"):/models:ro" \
    ${CHAT_TEMPLATE_MOUNT} \
    "${IMAGE_REF}" \
    bash -lc "${FIXUPS} exec vllm serve /models/$(basename "${MODEL_PATH}") \
        --served-model-name ${MODEL_NAME} \
        --host 127.0.0.1 --port ${VLLM_PORT} \
        --tensor-parallel-size 1 \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization ${GPU_MEM_UTIL} \
        --trust-remote-code \
        ${CHAT_TEMPLATE_ARG} \
        ${ICF_VLLM_EXTRA_ARGS:-}"
