#!/bin/bash
# Stage the self-built RCCL-ANP plugin + the host's AMD libionic into zenith's patches dir, which
# run_container.sh bind-mounts at /opt/zenith_patches (so ZENITH_ANP_DIR=/opt/zenith_patches/anp).
# Run on a compute node (login1 has the upstream libionic, not libionic-amd):
#   srun -A <account> -p batch -N1 --exclusive --gres=gpu:8 -t 5 bash stage_anp.sh [v1.4.0]
# Unlike the ROCm 7.2.2 sites, the image's rdma-core 61 already speaks the host provider's ABI
# (IBVERBS_PRIVATE_59 on both), so only libionic is staged, not libibverbs/librdmacm.
set -eux
A="${ANP_WORK:?set ANP_WORK to the dir build_anp.sh used}"
D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # this package's docker/patches/anp
T=${1:-v1.4.0}
rm -rf $D/librccl-anp.so $D/libionic* $D/libibverbs.d && mkdir -p $D/libibverbs.d
cp $A/anp-$T/build/librccl-anp.so $D/
cp /usr/lib64/libionic/libionic.so.1.1.61 $D/
# one file serves as both the verbs provider and the ionic_dv library ANP links: symlinks to the same
# inode, so the dynamic loader maps it once and the dv calls see the provider's own QP structs
ln -s libionic.so.1.1.61 $D/libionic.so.1
ln -s libionic.so.1.1.61 $D/libionic-rdmav59.so
# an absolute driver path makes libibverbs load <path>-rdmav59.so instead of the image's
# upstream ionic provider from its provider dir
echo "driver /opt/zenith_patches/anp/libionic" > $D/libibverbs.d/ionic.driver
echo "$T $(md5sum < $A/anp-$T/build/librccl-anp.so | cut -c1-12) built against RCCL rocm-7.2.4, libionic-amd $(rpm -q libionic-amd)" > $D/VERSION
readelf -d $D/libionic.so.1.1.61 | grep NEEDED
ls -la $D $D/libibverbs.d; cat $D/VERSION
