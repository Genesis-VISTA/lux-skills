#!/bin/bash -l
# Muse Spark Open on OLCF Lux: serve it as an OpenAI-compatible endpoint on one node, or serve it, benchmark it
# and exit; optionally build the musemain2 image first. Carries no #SBATCH lines: resources come from
# cluster_defaults.json under VISTA (which writes the header), and from scripts/muse.sh by hand.
# setup_lux.sh has already run on the login node.
#
# Env:
#   RUN_DIR_Lux     this folder's files (the lux-inference/ engine, summarize.sh); uploaded by VISTA, only read
#   VISTA_OUT       per-job output dir; VISTA creates it, by hand it defaults to $LUX_RUNS_DIR/muse.<jobid>
#   VISTA_JOB_DIR   state shared by every run of this job: a BUILD_IMAGE=1 image lands here
#   LUX_MODELPATH   dir holding the Muse checkpoint(s) and, optionally, Muse-Spark-Open-DSpark/ (cluster_defaults.json:
#                   the shared Genesis-Hackathon models dir, which holds the MXFP4 checkpoint)
#   LUX_IMAGE_TAR   the musemain2 image tar (cluster_defaults.json: the shared copy)
#   LUX_DATASETSPATH  benchmark data (science/ holds the suite's JSONL files)
# Args ("$@", VISTA's script_args): KEY=VALUE pairs, see README.md. Every LUX_* in the environment also reaches
# the engine (serve_lux.slurm.sh maps LUX_<KNOB> onto ZENITH_<KNOB>).
set -u
: "${RUN_DIR_Lux:?RUN_DIR_Lux not set}"
VISTA_OUT=${VISTA_OUT:-${LUX_RUNS_DIR:?set VISTA_OUT or LUX_RUNS_DIR}/muse.$SLURM_JOB_ID}
mkdir -p "$VISTA_OUT"

MXFP4=Muse-Spark-Open-816B-A42B-Quark-MXFP4 BF16=Muse-Spark-Open-816B-A42B DSPARK=Muse-Spark-Open-DSpark
for kv in "$@"; do
  case "$kv" in
    MODEL=*)         export LUX_MODEL="${kv#*=}" ;;
    SPEC=*)          SPEC="${kv#*=}" ;;
    BENCH=*)         export LUX_BENCH="${kv#*=}"; LUX_BENCH=${LUX_BENCH//,/ } ;;
    SCIENCE_N=*)     export LUX_SCIENCE_ARGS="--n ${kv#*=}" ;;
    GSM8K_N=*)       export LUX_GSM8K_N="${kv#*=}" ;;
    MAX_MODEL_LEN=*) export LUX_MAX_MODEL_LEN="${kv#*=}" ;;
    MODELPATH=*)     export LUX_MODELPATH="${kv#*=}"
                     case "$LUX_MODELPATH" in /lustre/orion/*) ;; *) echo "ERROR: MODELPATH must be a /lustre/orion/... directory" >&2; exit 2 ;; esac ;;
    BUILD_IMAGE=*)   BUILD_IMAGE="${kv#*=}" ;;
    *) echo "ERROR: unknown argument '$kv' (accepted: MODEL= SPEC= BENCH= SCIENCE_N= GSM8K_N= MAX_MODEL_LEN= MODELPATH= BUILD_IMAGE=)" >&2; exit 2 ;;
  esac
done
export LUX_MODEL=${LUX_MODEL:-$MXFP4}
case "$LUX_MODEL" in
  $MXFP4|$MXFP4-r63|$BF16) ;;
  *) echo "ERROR: MODEL must be $MXFP4 (or its -r63 mirror) or $BF16" >&2; exit 2 ;;
esac
case "${SPEC:=auto}" in
  auto|dspark|none) ;;
  *) echo "ERROR: SPEC must be auto, dspark or none (the DFlash2 drafts corrupt Muse's output; zenith results/muse/spec_decode_mxfp4_2026-10-17)" >&2; exit 2 ;;
esac
for s in ${LUX_BENCH:-}; do
  case "$s" in science|mmlu|gsm8k) ;; *) echo "ERROR: BENCH steps are science, mmlu, gsm8k (comma-separated)" >&2; exit 2 ;; esac
done
case "${BUILD_IMAGE:-0}" in 0|1) ;; *) echo "ERROR: BUILD_IMAGE must be 0 or 1" >&2; exit 2 ;; esac
[ -n "${LUX_SCIENCE_ARGS:-}" ] && [ -z "${LUX_BENCH:-}" ] && export LUX_BENCH=science
[ -n "${LUX_GSM8K_N:-}" ] && [ -z "${LUX_BENCH:-}" ] && export LUX_BENCH=gsm8k
[ "${SLURM_JOB_NUM_NODES:-1}" = 1 ] || { echo "ERROR: Muse serves one endpoint per node; submit with node_count=1 (run replicas for more users)" >&2; exit 2; }

: "${LUX_MODELPATH:?LUX_MODELPATH not set (cluster_defaults.json or env.lux.sh)}"
[ -r "$LUX_MODELPATH/$LUX_MODEL/config.json" ] || { echo "ERROR: no model $LUX_MODEL in $LUX_MODELPATH (bf16 is not shared: download it, MUSE_SPARK.md section 3, then MODELPATH=<dir>)" >&2; exit 2; }
# The draft must sit beside the model: the engine mounts LUX_MODELPATH as /app/models. auto = DSpark when it is there.
DRAFT_NOTE=
if [ -f "$LUX_MODELPATH/$DSPARK/.verified" ]; then
  [ "$SPEC" = none ] && export LUX_MUSE_DRAFT= || export LUX_MUSE_DRAFT=$DSPARK
elif [ "$SPEC" = dspark ]; then
  echo "ERROR: draft $DSPARK not downloaded/verified in $LUX_MODELPATH (scripts/muse.sh download), or SPEC=auto|none" >&2; exit 2
else
  export LUX_MUSE_DRAFT=
  [ "$SPEC" = auto ] && DRAFT_NOTE="no verified $DSPARK in $LUX_MODELPATH: serving without speculative decoding (slower; scripts/muse.sh download adds it)"
fi
IMAGES=${VISTA_JOB_DIR:-$VISTA_OUT/..}/images
OWN_TAR=$IMAGES/inference_image_gfx950_musemain2.tar
if [ ! -r "${LUX_IMAGE_TAR:-}" ] || [ ! -s "${LUX_IMAGE_TAR:-}" ]; then
  [ -s "$OWN_TAR" ] || [ "${BUILD_IMAGE:-0}" = 1 ] \
    || { echo "ERROR: no image at ${LUX_IMAGE_TAR:-<LUX_IMAGE_TAR unset>} or $OWN_TAR; resubmit with BUILD_IMAGE=1 (adds ~1 h)" >&2; exit 2; }
  export LUX_IMAGE_TAR=$OWN_TAR
fi
# Outputs and the engine's per-job state go under VISTA_OUT; the uploaded sources are never written.
export LUX_SERVE_OUT=$VISTA_OUT ZENITH_STATE_DIR=$VISTA_OUT/state

# Run the engine from a per-job copy on Lustre (the build writes its log beside the scripts).
E=$VISTA_OUT/engine
rm -rf "$E"; mkdir -p "$E"
cp -a "$RUN_DIR_Lux/lux-inference/docker" "$RUN_DIR_Lux/lux-inference/bench" "$E/" || { echo "ERROR: cannot stage the engine in $E" >&2; exit 3; }
rm -rf "$E"/docker/.zenith_state "$E"/docker/serve.*
MODE=serve; [ -n "${LUX_BENCH:-}" ] && MODE="bench ($LUX_BENCH)"
echo "job $SLURM_JOB_ID node=$SLURM_JOB_NODELIST model=$LUX_MODEL draft=${LUX_MUSE_DRAFT:-none} mode=$MODE out=$VISTA_OUT" \
  | tee "$VISTA_OUT/job.log"
[ -n "$DRAFT_NOTE" ] && echo "=== DRAFT none: $DRAFT_NOTE" | tee -a "$VISTA_OUT/job.log"

refresh() {
  bash "$RUN_DIR_Lux/summarize.sh" "$VISTA_OUT" > "$VISTA_OUT/summary.txt.tmp" 2>&1 \
    && mv -f "$VISTA_OUT/summary.txt.tmp" "$VISTA_OUT/summary.txt"
  url=$(grep -oE '^=== SERVING at [^ ]+' "$VISTA_OUT/job.log" | tail -1 | cut -d' ' -f4)
  [ -n "$url" ] && echo "$url" > "$VISTA_OUT/endpoint.txt"
  return 0
}
( while sleep 60; do refresh; done ) &
WATCH=$!
finish() { kill "$WATCH" 2>/dev/null; rm -rf "$E"; echo "=== exit $1" | tee -a "$VISTA_OUT/job.log"; refresh; cat "$VISTA_OUT/summary.txt"; exit "$1"; }

# The image: vLLM 0e939af8 (the musemain bundle in docker/vllm_src/) + AITER a3158e48, tag musemain2
# (zenith v0.1.9 README.md §1d.2). The build needs this node's GPUs and takes ~36 min (allow ~1 h).
if [ ! -s "$LUX_IMAGE_TAR" ]; then   # only reached with BUILD_IMAGE=1 (checked above)
  mkdir -p "$(dirname "$LUX_IMAGE_TAR")"
  (
    flock 9   # another run of this job may be building the same tar
    if [ -s "$LUX_IMAGE_TAR" ]; then echo "=== BUILD skipped: $LUX_IMAGE_TAR appeared while waiting"; exit 0; fi
    echo "=== BUILD start $(date) -> $LUX_IMAGE_TAR"
    export ZENITH_GRAPHROOT=/mnt/bb/$SLURM_JOB_ID/containers/storage ZENITH_TMPDIR=/mnt/bb/$SLURM_JOB_ID/containers/tmp
    export XDG_RUNTIME_DIR=/tmp/zenith-xdg-$SLURM_JOB_ID XDG_CONFIG_HOME=/tmp/zenith-xdg-$SLURM_JOB_ID/config
    mkdir -p "$ZENITH_GRAPHROOT" "$ZENITH_TMPDIR" "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"
    # The build pulls the base image and Python packages through the OLCF proxy (serve_lux.slurm.sh unsets it
    # again before serving, as zenith's sbatch does).
    export http_proxy=http://proxy.ccs.ornl.gov:3128 https_proxy=http://proxy.ccs.ornl.gov:3128
    export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'
    cd "$E/docker" && VLLM_HASH=0e939af8ad477f52230534fca91053d96e7da59a VLLM_BUNDLE=vllm_musemain_0e939af8.bundle \
      AITER_HASH=a3158e4800b6080e3adb25eb58bc48686bc1859c ZENITH_IMAGE_TAG=musemain2 ZENITH_BUILD_SKIP_PATCHES= \
      ZENITH_IMAGE_TAR="$LUX_IMAGE_TAR.partial" bash ./build_image.sh gfx950 > "$VISTA_OUT/build.log" 2>&1
    rc=$?
    cp -f "$E"/docker/logs/build_*.log "$VISTA_OUT/" 2>/dev/null
    podman unshare rm -rf "/mnt/bb/$SLURM_JOB_ID/containers" 2>/dev/null; rm -rf "$XDG_RUNTIME_DIR"
    # A build that does not print this line did not verify (zenith README.md, "Building the image").
    grep -q '=== zenith image build checks passed ===' "$VISTA_OUT/build.log" && [ "$rc" = 0 ] && [ -s "$LUX_IMAGE_TAR.partial" ] \
      || { echo "=== BUILD FAILED (rc $rc): see build.log"; rm -f "$LUX_IMAGE_TAR.partial"; exit 1; }
    mv -f "$LUX_IMAGE_TAR.partial" "$LUX_IMAGE_TAR" && echo "=== BUILD done $(date)"
  ) 9>"$LUX_IMAGE_TAR.lock" 2>&1 | tee -a "$VISTA_OUT/job.log"
  [ "${PIPESTATUS[0]}" = 0 ] || finish 3
fi

# The allocation asks for --gpus-per-node (VISTA's header on Lux), and srun inherits it from
# SLURM_GPUS_PER_NODE; the engine's srun asks for --gres=gpu:8, and Slurm refuses the two together.
unset SLURM_GPUS_PER_NODE
# serve_lux.slurm.sh finds its siblings from SLURM_SUBMIT_DIR; Muse's serving defaults are in lux_tuned_defaults.sh.
SLURM_SUBMIT_DIR=$E/docker bash "$E/docker/serve_lux.slurm.sh" 2>&1 | tee -a "$VISTA_OUT/job.log"
finish "${PIPESTATUS[0]}"
