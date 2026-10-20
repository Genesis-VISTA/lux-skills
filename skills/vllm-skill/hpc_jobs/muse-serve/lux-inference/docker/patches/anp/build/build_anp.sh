#!/bin/bash
# Build AMD's RCCL-ANP net plugin for lux, and inspect the image's rdma-core so the staging can match it.
# Two stages, because neither host has everything:
#   login1  (has /usr/include/libdrm, which RCCL's cmake needs via rocm_smi; has internet for FetchContent fmt):
#       bash build_anp.sh rccl
#   ANP: on a compute node (libionic-amd-devel 61, whose ionic_dv.h declares the ionic_dv_qp_set_gda ANP
#   calls) -- or on login1 once a compute node has copied /usr/include/libionic to $A/hostinc:
#       sbatch -A <account> -p batch -N1 --exclusive --gres=gpu:8 -t 00:30:00 -o build.%j.out --wrap "bash build_anp.sh anp"
# Inputs under $ANP_WORK (see ../README.md): rccl/ (git clone, tag rocm-7.2.4), anp-v1.4.0/ and
# anp-v1.5.0/ (amd-anp release tarballs), boost/ (boost 1.86 headers).
set -ux
A="${ANP_WORK:?set ANP_WORK to a scratch dir holding rccl/, anp-v1.4.0/, boost/}"
IMAGE_TAR="${ZENITH_IMAGE_TAR:?set ZENITH_IMAGE_TAR to the inference image tar}"
export ROCM_PATH=/opt/rocm-7.2.4
export PATH=$ROCM_PATH/bin:$PATH

if [ "${1:-anp}" = rccl ]; then
  # configure + hipify only: ANP compiles a handful of RCCL sources, never links librccl
  cd $A/rccl && rm -rf build
  cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx950 -DCMAKE_PREFIX_PATH=$ROCM_PATH \
        -DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/amdclang++ -DCMAKE_C_COMPILER=$ROCM_PATH/bin/amdclang \
        -DBUILD_TESTS=OFF > $A/rccl_cmake.log 2>&1 || { tail -30 $A/rccl_cmake.log; exit 1; }
  make -C build/release -j16 hipify_all > $A/rccl_hipify.log 2>&1 || { tail -30 $A/rccl_hipify.log; exit 1; }
  touch build/release/librccl.so   # only checked for existence by the ANP Makefile
  exit 0
fi

B=/mnt/bb/${SLURM_JOB_ID:-none}
# --- the image's rdma-core, in the background (podman unpacking the 57 GB tar takes a while)
(
  export XDG_RUNTIME_DIR=/tmp/zenith-xdg-${SLURM_JOB_ID} XDG_CONFIG_HOME=/tmp/zenith-xdg-${SLURM_JOB_ID}/config TMPDIR=$B/tmp
  mkdir -p $XDG_RUNTIME_DIR $B/storage $B/tmp && chmod 700 $XDG_RUNTIME_DIR
  # same storage flags as zenith_podman_env.sh: no subuid ranges here, so fuse-overlayfs + ignore_chown_errors
  podman --root $B/storage --runroot $XDG_RUNTIME_DIR/run --cgroup-manager cgroupfs --storage-driver overlay \
      --storage-opt mount_program=/usr/bin/fuse-overlayfs --storage-opt ignore_chown_errors=true run --rm --entrypoint bash \
      docker-archive:$IMAGE_TAR -c '
    echo "== image libibverbs"; ls -l /usr/lib/x86_64-linux-gnu/libibverbs.so* /usr/lib/x86_64-linux-gnu/librdmacm.so* /usr/lib/x86_64-linux-gnu/libionic* 2>&1
    dpkg -l | grep -E "rdma-core|ibverbs|ionic" | awk "{print \$2, \$3}"
    ls /usr/lib/x86_64-linux-gnu/libibverbs/ 2>&1 | tr "\n" " "; echo; ls /etc/libibverbs.d | tr "\n" " "; echo
    objdump -T /usr/lib/x86_64-linux-gnu/libibverbs.so.1 | grep -oE "IBVERBS_PRIVATE_[0-9]+" | sort -u
    cat /etc/os-release | head -2; ldd --version | head -1; echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    find / -xdev -name "librccl.so*" 2>/dev/null | head -3; ls /opt/zenith 2>&1' > $A/image_rdma.txt 2>&1
) &
IMGPID=$!

{
  echo "== host"; hostname
  for d in /sys/class/infiniband/*; do echo "$(basename $d) $(cat $d/ports/1/state)"; done
  rpm -qa | grep -iE "rdma-core|libibverbs|ionic" | sort
  objdump -T /usr/lib64/libibverbs.so.1 | grep -oE "IBVERBS_PRIVATE_[0-9]+" | sort -u
} > $A/host_rdma.txt 2>&1

# -I order: ROCm 7.2.4 before the Makefile's hardcoded /opt/rocm (7.14), AMD libionic's ionic_dv.h before
# the upstream rdma-core one, plus the fmt headers RCCL's proxy_trace.h now needs and the header-only
# boost (version, histogram, property_tree) ANP's telemetry needs; lux has no boost-devel, so login1
# fetched boost 1.86 into $A/boost
# libionic-amd 61 no longer defines the two udma pipeline masks ANP uses (one bit per pipeline), and
# RCCL rocm-7.2.4's ncclCudaCalloc grew a stream argument ahead of the flags
FMT=$A/rccl/build/release/_deps/fmt-src/include
IONIC_INC=/usr/include/libionic; [ -d $IONIC_INC ] || IONIC_INC=$A/hostinc/libionic
for t in v1.4.0 v1.5.0; do
  sed -i "s/ncclCudaCalloc(&rCommDev->gpuFlush.gpuFlushGpuMem, sizeof(int), hipDeviceMalloc/ncclCudaCalloc(\&rCommDev->gpuFlush.gpuFlushGpuMem, sizeof(int), nullptr, hipDeviceMalloc/" $A/anp-$t/src/net_ib.cc
  ( cd $A/anp-$t && make clean >/dev/null 2>&1; make -j32 RCCL_HOME=$A/rccl RCCL_BUILD=$A/rccl/build/release \
        ROCM_PATH=$ROCM_PATH CC=$ROCM_PATH/bin/hipcc \
        CPPFLAGS="-I$ROCM_PATH/include -I$IONIC_INC -I$FMT -I$A/boost -DIONIC_UDMA_MASK_LOW=0x1 -DIONIC_UDMA_MASK_HIGH=0x2" > $A/anp-$t.build.log 2>&1 ) \
    && echo "ANP $t built: $(ls -l $A/anp-$t/build/librccl-anp.so)" \
    || { echo "ANP $t FAILED"; grep -m10 -E "error" $A/anp-$t.build.log; }
done
for t in v1.4.0 v1.5.0; do
  f=$A/anp-$t/build/librccl-anp.so; [ -f $f ] || continue
  echo "== $t deps"; ldd $f; nm -D $f | grep -iE "ncclNet|NCCL_NET_PLUGIN|ncclCollNet"
done
wait $IMGPID
cat $A/host_rdma.txt $A/image_rdma.txt
