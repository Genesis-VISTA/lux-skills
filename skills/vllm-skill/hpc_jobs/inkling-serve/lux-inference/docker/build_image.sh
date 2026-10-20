#!/bin/bash
#
# build the zenith inference container image
#
# usage: [ZENITH_IMAGE_TAG=<tag>] ./build_image.sh [gfx950|gfx942|gfx90a]
#   gfx950 = MI355X (CDNA4, discrete, Pollara RoCE)   -- default
#   gfx942 = MI300A (CDNA3, APU,      HPE Slingshot)
#   gfx90a = MI250X (CDNA2, discrete dual-GCD, HPE Slingshot)
#
# The ROCm SDK wheels are published per GPU family, and the family name is not
# derivable from the arch string -- it has to be looked up in the index at
# https://rocm.nightlies.amd.com/v2/ . Two gfx9 families exist for CDNA2
# ("gfx90a" and "gfx90X-dcgpu") and only the bare "gfx90a" one carries builds
# as new as the ROCM_DEVEL_VERSION this Dockerfile pins; gfx90X-dcgpu stops at
# 7.12.0a. Check the index before changing these.
#
# ZENITH_IMAGE_TAG=<tag> (default: latest) builds localhost/inference_image_<arch>:<tag> and saves
# inference_image_<arch>_<tag>.tar, so a build on a host that already holds an image (e.g. zenith_v6 as
# inference_image_gfx950:latest + inference_image.tar) does not retag or overwrite it. Run the result
# with the same ZENITH_IMAGE_TAG (run_node.sh, run_container.sh).
#
# The tarball name carries the arch (inference_image_<arch>[_<tag>].tar): an image serves exactly one
# GPU arch, and a gfx942 image on an MI250X node only fails at the first kernel launch. Tarballs from
# before this were plain inference_image[_<tag>].tar; run_container.sh falls back to that name when
# the arch-named one is missing.
#
# ZENITH_SHS_VERSION=<x.y.z> (default 13.1.0) pins the image's Slingshot userspace (cassini headers,
# cxi uapi headers, libcxi) to that SHS release. It must not be newer than the host cxi driver of any
# site the image runs on; see the comment above ZENITH_SHS_VERSION in the Dockerfile.
set -euo pipefail
cd "$(dirname "$0")"
source ./lux_podman_env.sh

ARCH="${1:-gfx950}"
case "${ARCH}" in
  gfx950) ROCM_FAMILY="gfx950-dcgpu" ;;
  gfx942) ROCM_FAMILY="gfx94X-dcgpu" ;;
  gfx90a) ROCM_FAMILY="gfx90a" ;;
  *) echo "unknown arch ${ARCH} (expected gfx950, gfx942 or gfx90a)"; exit 1 ;;
esac

# VLLM_HASH=<sha> builds a candidate vLLM pin under its own image name and
# tarball, so the production inference_image_<arch>.tar is never touched by a trial.
VLLM_HASH="${VLLM_HASH:-}"
TAG_SUFFIX="${VLLM_HASH:+_vllm${VLLM_HASH:0:8}}"
IMAGE_TAG="${ZENITH_IMAGE_TAG:-latest}"
case "${IMAGE_TAG}" in
  *[!A-Za-z0-9_.-]*|""|[.-]*) echo "invalid ZENITH_IMAGE_TAG '${IMAGE_TAG}' (letters, digits, _ . -)"; exit 1 ;;
esac
IMAGE="inference_image_${ARCH}${TAG_SUFFIX}:${IMAGE_TAG}"
TAR_TAG=""; [ "${IMAGE_TAG}" != "latest" ] && TAR_TAG="_${IMAGE_TAG}"
TARBALL="inference_image_${ARCH}${TAG_SUFFIX}${TAR_TAG}.tar"
# ZENITH_IMAGE_TAR=<path> writes the tarball there instead (same variable run_container.sh reads),
# e.g. onto a parallel filesystem rather than next to this script on an NFS home.
TARBALL="${ZENITH_IMAGE_TAR:-${TARBALL}}"
mkdir -p "$(dirname "${TARBALL}")"
VLLM_ARGS=()
[ -n "${VLLM_HASH}" ] && VLLM_ARGS=(--build-arg VLLM_HASH="${VLLM_HASH}")

SHS_VERSION="${ZENITH_SHS_VERSION:-13.1.0}"

echo "building ${IMAGE} for ${ARCH} (rocm devel family ${ROCM_FAMILY}, SHS ${SHS_VERSION})"
echo "  tarball:   ${TARBALL}"
echo "  graphroot: ${ZENITH_GRAPHROOT}"
echo "  tmpdir:    ${ZENITH_TMPDIR}"

# keep-groups is what lets the rootless build see /dev/kfd, which is mode 0666
# but /dev/dri/renderD* are group `render`; without it the container user drops
# its supplementary groups and the devices are present but unusable.
GPU_DEVICES=(--device /dev/kfd --group-add keep-groups)
for d in /dev/dri/renderD*; do
    [ -e "${d}" ] && GPU_DEVICES+=(--device "${d}")
done
if [ ! -e /dev/kfd ]; then
    echo "ERROR: no /dev/kfd on $(hostname) -- the image's build-time GPU checks"
    echo "ERROR: cannot run here. Build on a GPU node."
    exit 1
fi

# ---------------------------------------------------------------------------
# stage AMD's RCCL-ANP net plugin into the build context
#
# The Pollara/ionic NICs need librccl-anp.so ("accelerated network path"), AMD's
# own RCCL net plugin. It ships with host ROCm 7.2.2, NOT with the ROCm 7.14
# container base. Without it RCCL falls back to its generic IB transport, which
# completes exactly one collective per communicator and then dies with
# `ibv_poll_cq() -> -5`. With it: all sizes pass and we get 1,048.6 Gbps.
#
# The plugin is linked against the vendor libionic, which needs rdma-core 50
# (IBVERBS_PRIVATE_34) -- the container has 61 -- so the matching libibverbs /
# librdmacm / libionic have to travel with it. They are loaded only by the
# plugin, via the LD_LIBRARY_PATH prefix run_container.sh sets; the rest of the
# image keeps using its own rdma-core 61.
#
# This is host-dependent, so it is done here rather than in the Dockerfile, and
# it is non-fatal: on a node without ROCm 7.2.2 the image simply builds without
# the plugin and run_container.sh falls back to generic IB.
ANP_SRC="/opt/rocm-7.2.2/lib/librccl-anp.so"
ANP_STAGE="patches/anp"
rm -rf "${ANP_STAGE}"
if [ -r "${ANP_SRC}" ]; then
    mkdir -p "${ANP_STAGE}/libibverbs.d"
    cp -f "${ANP_SRC}" "${ANP_STAGE}/"
    # note libionic-rdmav34.so lives under libibverbs/ on the host but is copied
    # in flat: rdma-core dlopens providers off LD_LIBRARY_PATH directly, not
    # from a libibverbs/ subdirectory.
    for f in /usr/lib/x86_64-linux-gnu/libionic.so.1.1.50 \
             /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so \
             /usr/lib/x86_64-linux-gnu/libibverbs.so.1 \
             /usr/lib/x86_64-linux-gnu/librdmacm.so.1; do
        [ -r "${f}" ] && cp -f "${f}" "${ANP_STAGE}/"
    done
    for f in librccl-anp.so libionic.so.1.1.50 libionic-rdmav34.so \
             libibverbs.so.1 librdmacm.so.1; do
        [ -r "${ANP_STAGE}/${f}" ] || { echo "ERROR: ANP staging missing ${f}"; exit 1; }
    done
    ( cd "${ANP_STAGE}" \
        && ln -sf libionic.so.1.1.50 libionic.so.1 \
        && ln -sf libibverbs.so.1 libibverbs.so )
    echo "driver ionic" > "${ANP_STAGE}/libibverbs.d/ionic.driver"
    echo "staged RCCL-ANP plugin from ${ANP_SRC}"
else
    echo "WARNING: ${ANP_SRC} not found -- image will have no RCCL-ANP plugin."
    echo "WARNING: multi-node RoCE will fall back to generic IB, which fails"
    echo "WARNING: after one collective. Build on a node with ROCm 7.2.2."
fi

# --format docker so that the ENV/CMD metadata survives the save/load round trip
mkdir -p logs
${PODMAN} build \
    --format docker \
    "${GPU_DEVICES[@]}" \
    --build-arg TRAIN_ARCHS="${ARCH}" \
    --build-arg ROCM_TARGET_FAMILY="${ROCM_FAMILY}" \
    --build-arg ZENITH_SHS_VERSION="${SHS_VERSION}" \
    "${VLLM_ARGS[@]}" \
    -f Dockerfile.lux.inference \
    -t "${IMAGE}" \
    . 2>&1 | tee "logs/build_${ARCH}${TAG_SUFFIX}${TAR_TAG}.log"

echo "saving ${IMAGE} -> ${TARBALL}"
# docker-archive refuses to write into a tarball that already holds an image
# ("docker-archive doesn't support modifying existing images"), so a rebuild
# fails at the very last step unless the old tarball is removed first.
rm -f "${TARBALL}"
${PODMAN} save --format docker-archive -o "${TARBALL}" "${IMAGE}"
ls -lh "${TARBALL}"
echo "build complete"
