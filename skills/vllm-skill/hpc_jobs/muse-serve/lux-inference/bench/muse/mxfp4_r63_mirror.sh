#!/bin/bash
# Build Muse-Spark-Open-816B-A42B-Quark-MXFP4-r63: a sibling of the MXFP4 checkpoint dir that symlinks every
# file except model-00063-of-00065.safetensors, whose only stripe sits on OST 530 (hung on 2026-10-17), and
# re-downloads that shard onto a healthy OST. Never stats the bad shard. Run detached on the login node:
#   setsid nohup bash mxfp4_r63_mirror.sh HFMODELS_DIR > LOG 2>&1 < /dev/null &
# Prints "VERIFIED" when the shard matches the Hugging Face LFS sha256.
set -u
H=${1:?usage: mxfp4_r63_mirror.sh HFMODELS_DIR}
SRC=$H/Muse-Spark-Open-816B-A42B-Quark-MXFP4
DST=$H/Muse-Spark-Open-816B-A42B-Quark-MXFP4-r63
REPO=amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4
REV=c7bc74fd64371315a091bc15f9723882eead0dc1
SHARD=model-00063-of-00065.safetensors
SHA=340af63b0b7fb4fd310e93dd642d0fecebe10e8ad85c31eaeacc36d1e7a534e9
SIZE=550288096
BAD_OST=530
OST=${R63_OST:-754}   # the RadixArk draft's OST, read fine on 2026-10-17

mkdir -p "$DST"
# names only (ls -f does not stat), so the bad shard is never touched. Relative links, so they resolve
# inside the container, where only the hfmodels dir is mounted (at /app/models).
ls -1f --color=never "$SRC" | while read -r f; do
    case "$f" in .|..|.cache|"$SHARD") continue ;; esac
    ln -sfn "../${SRC##*/}/$f" "$DST/$f"
done
echo "$(date) $(ls -1f "$DST" | grep -c '^model-000') shard links"

out=$DST/$SHARD
if [ ! -f "$out" ]; then
    lfs setstripe -c 1 -i "$OST" "$out" || exit 1
fi
ost=$(lfs getstripe -i "$out")
[ "$ost" = "$BAD_OST" ] && { echo "$(date) landed on OST $BAD_OST, abort"; exit 1; }
echo "$(date) $SHARD on OST $ost"
for try in $(seq 20); do
    [ "$(stat -c %s "$out")" = "$SIZE" ] && break
    curl -sSfL -C - -o "$out" "https://huggingface.co/$REPO/resolve/$REV/$SHARD" && break
    echo "$(date) try $try failed"; sleep 60
done
got=$(sha256sum "$out" | cut -d' ' -f1)
if [ "$got" = "$SHA" ]; then
    echo "$REPO $REV $SHARD" > "$DST/.r63_verified"; echo "$(date) VERIFIED"
else
    echo "$(date) SHA MISMATCH $got"; exit 1
fi
