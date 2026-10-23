#!/bin/bash
# RingX setup on the Lux LOGIN node (network via the proxy). Idempotent, so it runs on every submit (seconds after the
# first one).
#   VISTA: run by the Lux dispatcher before sbatch; a failure fails the submit call.
#   By hand: scripts/bench.sh runs it (env.lux.sh sets RINGX_ROOT).
# Clones jqyin/ringX-attention at a pinned commit and installs ring-flash-attn (the baselines, pure Python) into a
# --target folder. Nothing goes into the image: the job puts both on PYTHONPATH.
#
# Env (VISTA exports RUN_DIR_Lux, VISTA_JOB_DIR, the proxy, and cluster_defaults.json -> lux.iri.environment):
#   RUN_DIR_Lux   this folder's uploaded files (bench_ringx.py, shims/, tests/, ...)
#   RINGX_ROOT    shared state, default $VISTA_JOB_DIR: ringX-attention/ and pydeps/
#   RINGX_COMMIT  the RingX commit (default d1e53d7, the one measured on Lux)
#   SIF           the Primus image (torch, FlashAttention 2.8.3, RCCL), only read
set -eu
RINGX_ROOT=${RINGX_ROOT:-${VISTA_JOB_DIR:?set RINGX_ROOT, or source env.lux.sh}}
RINGX_COMMIT=${RINGX_COMMIT:-d1e53d7394f36e12ec8f8717beabd862dd2a6cef}
RFA_VERSION=${RFA_VERSION:-0.1.8}
: "${RUN_DIR_Lux:?RUN_DIR_Lux not set}"
: "${SIF:?SIF not set (cluster_defaults.json or env.lux.sh)}"

for f in bench_ringx.py fa_ceiling.py summarize.py shims/fa_rng_fix.py shims/utils.py shims/ringX4_attn.py tests/test_ringX4_attn_func.py; do
  [ -r "$RUN_DIR_Lux/$f" ] || { echo "ERROR: $RUN_DIR_Lux/$f is missing (incomplete upload?)" >&2; exit 1; }
done
[ -s "$SIF" ] && [ -r "$SIF" ] || { echo "ERROR: the image $SIF is missing or unreadable. It is the primus-skill image; set SIF to a readable copy." >&2; exit 1; }

mkdir -p "$RINGX_ROOT"
# Two submits at once would race on the clone.
exec 9>"$RINGX_ROOT/.setup.lock"; flock 9

R=$RINGX_ROOT/ringX-attention
if [ -e "$R" ] && ! git -C "$R" rev-parse --git-dir >/dev/null 2>&1; then
  echo "ERROR: $R exists but is not a git clone. Not touching it; move it or set RINGX_ROOT elsewhere." >&2; exit 1
fi
# No submodules: they are the ViT/GPT apps, not needed for the kernels.
[ -e "$R" ] || for i in 1 2 3; do git clone -q https://github.com/jqyin/ringX-attention "$R" && break || { rm -rf "$R"; sleep 5; }; done
[ -d "$R/.git" ] || { echo "ERROR: could not clone jqyin/ringX-attention (GitHub via the proxy)" >&2; exit 1; }
if [ "$(git -C "$R" rev-parse HEAD)" != "$(git -C "$R" rev-parse -q --verify "$RINGX_COMMIT^{commit}" 2>/dev/null)" ]; then
  git -C "$R" rev-parse -q --verify "$RINGX_COMMIT^{commit}" >/dev/null || git -C "$R" fetch -q origin
  git -C "$R" checkout -q --detach "$RINGX_COMMIT" || { echo "ERROR: cannot check out RingX $RINGX_COMMIT" >&2; exit 1; }
fi
[ -z "$(git -C "$R" status --porcelain --untracked-files=no)" ] || echo "WARNING: $R has local changes; the job runs them as they are"

P=$RINGX_ROOT/pydeps
if [ ! -f "$P/ring_flash_attn-$RFA_VERSION.dist-info/METADATA" ]; then
  rm -rf "$P.new"
  python3 -m pip install -q --no-deps --target "$P.new" "ring-flash-attn==$RFA_VERSION" \
    || { rm -rf "$P.new"; echo "ERROR: pip could not install ring-flash-attn==$RFA_VERSION (PyPI via the proxy)" >&2; exit 1; }
  rm -rf "$P"; mv "$P.new" "$P"
fi
echo "setup OK: RingX $(git -C "$R" rev-parse --short HEAD), ring-flash-attn $RFA_VERSION, image $SIF"
