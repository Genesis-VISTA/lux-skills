#!/bin/bash
#
# Stage the HOST's Slingshot stack (libfabric, libcxi and LLNL's aws-ofi-rccl
# build) into a directory run_container.sh can mount, for
# ZENITH_HOST_FABRIC_DIR=<dir>. Fallback for when the image's own source-built
# libcxi/libfabric/plugin does not talk to this system's cxi driver.
#
#   flux run -N1 -q <queue> ./stage_host_fabric.sh [dest]
#
#   dest        $1, else ZENITH_HOST_FABRIC_DIR (the variable run_container.sh
#               mounts it from), else ${ZENITH_SCRATCH:-/p/lustre5}/$USER/zenith_v7/hostfabric.
#               Replaced wholesale, so it has to be a dedicated directory.
#   ZENITH_HOST_OFI_DIR  the site's aws-ofi-rccl install lib/ (librccl-net.so);
#               the default is LLNL's TOSS 4 Cray build.
#
# Run it on a COMPUTE node of the system the containers will run on, not a
# login node and not another machine. What it copies is that node's own
# libfabric/libcxi, which have to match that node's cxi driver (the whole point
# of the exercise); a login node may carry a different SHS, or no libcxi at all,
# and the RHEL deps are resolved by that node's ldd. One staging per system: an
# RZVernal job needs a directory staged on RZVernal.
#
# Layout, and why it is split in two:
#   <dest>/       libfabric.so.1, libcxi.so.1, librccl-net.so -- PREPENDED to the
#                 container's LD_LIBRARY_PATH so they win over /opt/libfabric etc.
#   <dest>/deps/  everything else they link that is not glibc/libgcc/libstdc++ --
#                 APPENDED, so a soname the container already has (libz, libnl,
#                 libnuma, ...) keeps resolving to the container's copy and only
#                 RHEL-only sonames (libssl.so.1.1, libjson-c.so.4, libhwloc.so.15,
#                 libxpmem, ...) come from here. Prepending the lot would shadow the
#                 container's newer libz and friends for torch and vLLM too.
set -euo pipefail
DEST="${1:-${ZENITH_HOST_FABRIC_DIR:-${ZENITH_SCRATCH:-/p/lustre5}/$(whoami)/zenith_v7/hostfabric}}"
PLUGIN_DIR="${ZENITH_HOST_OFI_DIR:-/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-7.14/install/lib}"

[ -e /dev/cxi0 ] || { echo "ERROR: no /dev/cxi0 on $(hostname) -- run this on a compute node of the target system"; exit 1; }
for f in /usr/lib64/libfabric.so.1 /usr/lib64/libcxi.so.1 "${PLUGIN_DIR}/librccl-net.so"; do
    [ -r "${f}" ] || { echo "ERROR: ${f} not found on $(hostname) (set ZENITH_HOST_OFI_DIR for the plugin)"; exit 1; }
done
# rm -rf below: only on a directory that is empty or an earlier staging
if [ -d "${DEST}" ] && [ -n "$(ls -A "${DEST}")" ] && [ ! -e "${DEST}/libfabric.so.1" ]; then
    echo "ERROR: ${DEST} exists and is not a host-fabric staging directory; refusing to replace it"
    exit 1
fi

rm -rf "${DEST}"
mkdir -p "${DEST}/deps"
cp -L /usr/lib64/libfabric.so.1 /usr/lib64/libcxi.so.1 "${DEST}/"
cp -L "${PLUGIN_DIR}/librccl-net.so" "${DEST}/"

# A dependency ldd cannot resolve here would be missing in the container too,
# and the plugin would then fail to dlopen -- which RCCL reports only at INFO
# level before quietly falling back to TCP. ROCm's own libraries are the
# exception: the host may have no ROCm module loaded, and inside the container
# they come from the image anyway.
MISSING=$(for top in "${DEST}"/*.so*; do ldd "${top}"; done | awk '/=> not found/{print $1}' | sort -u \
    | grep -v -E '^(libamdhip64|libhsa-runtime64|libhsakmt|libhip|librccl|librocm|libroctx|libroctracer|librocprofiler|libamd_comgr|libamd_smi)' || true)
if [ -n "${MISSING}" ]; then
    echo "ERROR: unresolved non-ROCm dependencies of the staged libraries on $(hostname):"
    echo "${MISSING}" | sed 's/^/  /'
    exit 1
fi

for top in "${DEST}"/*.so*; do
    ldd "${top}" | awk '/=> \//{print $1, $3}'
done | sort -u | while read -r soname path; do
    case "${soname}" in
        libc.so.*|libm.so.*|libpthread.so.*|libdl.so.*|librt.so.*|ld-linux*|libgcc_s.so.*|libstdc++.so.*|libresolv.so.*|libcrypt.so.*) continue ;;
        libfabric.so.1|libcxi.so.1) continue ;;
    esac
    cp -L "${path}" "${DEST}/deps/${soname}"
done
# libfabric's ROCR hmem dlopens the unversioned libhsa-runtime64.so, which in
# the image is the _rocm_sdk_devel copy -- a different inode from the
# _rocm_sdk_core libhsa-runtime64.so.1 that HIP/torch already mapped. Two HSA
# runtimes in one process: libfabric's cannot see HIP allocations, treats GPU
# buffers as host memory, and cxil_map fails with EFAULT ("Couldn't register
# memory region ... Bad address"), so RCCL drops to TCP. Point the name at
# the core copy (dangling on the host, resolves inside the container).
ln -sfn /opt/venv/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1 \
    "${DEST}/libhsa-runtime64.so"
ls -l "${DEST}" "${DEST}/deps"
echo "staged host fabric -> ${DEST}"
echo "  libfabric $(/usr/bin/fi_info --version | awk '/^libfabric:/{print $2}'), libcxi $(rpm -q --qf '%{VERSION}' cray-libcxi 2>/dev/null), plugin ${PLUGIN_DIR}"
