#!/bin/bash
# Download the Muse Spark Open speculative-decoding drafts into the hfmodels dir and check them
# against the Hugging Face LFS sha256 (2026-10-17 revisions). Run detached on the login node:
#   setsid nohup bash spec_drafts_download.sh DEST > DEST/spec_drafts_download.log 2>&1 < /dev/null &
# Re-running resumes partial files and skips verified ones. Prints "ALL VERIFIED" when done.
set -u
DEST=${1:?usage: spec_drafts_download.sh HFMODELS_DIR}
# repo  revision  local dir  safetensors sha256
DRAFTS="
Inferact/Muse-Spark-Open-DSpark ffc3eb50c8ae7a5a391ddedc43085d2e5249bc35 Muse-Spark-Open-DSpark b09446d1ce58a93484aa4be6a4827b6f47bd01d3f70b5881db51edd43af9e8c1
RadixArk/Muse-Spark-Open-DFlash2 46c90f412549c472941a68382d91a99e5e6372a0 Muse-Spark-Open-DFlash2-RadixArk 06d5a6f7b9658d53eb484d1b692cb923975b29f49546e3af7805995f2a4e5feb
nvidia/Muse-Spark-Open-816B-A42B-DFlash2 fbec887a79866e9ecff6c57adfd9fd02aa3cadea Muse-Spark-Open-DFlash2-NVIDIA 91f365a6b96c08aedd2afa14149854ffe63cc99231dddf514169d4cf2d7fd483
"
fail=0
while read -r repo rev dir sha; do
    [ -n "$repo" ] || continue
    d=$DEST/$dir
    mkdir -p "$d"
    if [ -f "$d/.verified" ]; then echo "$(date) $dir already verified"; continue; fi
    echo "$(date) $repo@$rev -> $d"
    for f in config.json README.md; do
        curl -sSfL --retry 5 -o "$d/$f" "https://huggingface.co/$repo/resolve/$rev/$f" || echo "$(date) $f failed"
    done
    size=$(curl -sSfI -L "https://huggingface.co/$repo/resolve/$rev/model.safetensors" | tr -d '\r' |
        awk 'tolower($1)=="x-linked-size:"{s=$2} END{print s}')
    for try in $(seq 20); do
        [ -n "$size" ] && [ "$(stat -c %s "$d/model.safetensors" 2>/dev/null)" = "$size" ] && break
        curl -sSfL -C - -o "$d/model.safetensors" "https://huggingface.co/$repo/resolve/$rev/model.safetensors" && break
        echo "$(date) model.safetensors try $try failed"; sleep 60
    done
    got=$(sha256sum "$d/model.safetensors" | cut -d' ' -f1)
    if [ "$got" = "$sha" ]; then
        echo "$repo $rev" > "$d/.verified"; echo "$(date) $dir VERIFIED"
    else
        echo "$(date) $dir SHA MISMATCH $got"; fail=1
    fi
done <<< "$DRAFTS"
[ $fail = 0 ] && echo "$(date) ALL VERIFIED" || echo "$(date) FAILED"
