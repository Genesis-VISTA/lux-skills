#!/usr/bin/env bash
# Submit Mistral-Large-Instruct-2411 (123B) continued pretraining: tutorial 3's lux/train.sbatch, FP8, TP1 x PP8
# (one node holds one copy of the model; every extra node adds data parallelism), 90% Sci-Base + 10% chat mix,
# starting from the Megatron TP1/PP8 checkpoint in $PKG.
#   scripts/run_cpt.sh [nodes] [KEY=VAL ...]                       default: 4 nodes, the tutorial's 4-step run (~10 min)
#   scripts/run_cpt.sh 4 TRAIN_ITERS=20 SAVE_INTERVAL=20 TIME=01:30:00
#   scripts/run_cpt.sh 256 TRAIN_GBS=32768 TRAIN_ITERS=283 TRAIN_WARMUP=50 TRAIN_EVAL_INTERVAL=32 \
#     SAVE_INTERVAL=71 LUX_GPU_CHECK=1 TIME=04:45:00 RUN_NAME=chatmix-1ep    # the reference epoch
# KEY=VAL pairs are exported into the job. The wrapper's own: TIME (-t, default 00:45:00), CHAT_SHARE (default 0.1;
# 0 = Sci-Base only), SAVE_INTERVAL (adds --save; checkpoints are 224 GB each), EXTRA_PRIMUS_ARGS (more
# "--key value" overrides). Everything else goes to lux/train.sbatch as is: TRAIN_ITERS, TRAIN_GBS, TRAIN_WARMUP,
# TRAIN_EVAL_ITERS, TRAIN_EVAL_INTERVAL, EPOCHS, RUN_NAME, LUX_GPU_CHECK, LUX_BW, GRAD_GUARD, LUX_EXCLUDE_EXTRA,
# DATA_PREFIX, DATA_VALID_PREFIX, CONFIG, NCCL_*/RCCL_*. Only TRAIN_*, DATA_*, LUX_*, PRIMUS_*, NCCL_*, RCCL_* (and a
# few more, lux/ctr.sh) reach the container.
set -eu
SK=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
source "$SK/env.lux.sh"
N=4
case "${1:-}" in ""|*=*) ;; *) N=$1; shift ;; esac
for kv in "$@"; do
  [[ $kv =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || { echo "ERROR: '$kv' is not KEY=VALUE" >&2; exit 2; }
  export "$kv"
done
[ -f "$CPT_WORKDIR/lux/lux_env.sh" ] || { echo "ERROR: $CPT_WORKDIR is not set up; run scripts/setup_cpt.sh" >&2; exit 1; }

# The tutorial's own env layers, in its order. SIF must be the tutorial's image, not the smoke test's.
export WORKDIR=$CPT_WORKDIR SIF=$CPT_SIF
source "$PKG/tutorials/03_mistral_cpt/hackathon_env.sh"
cd "$WORKDIR"
source lux/lux_env.sh

export CONFIG=${CONFIG:-/work/lux/configs/mistral_large_2411-lux-cpt-4node-fp8-tp1pp8.yaml}
export DATA_PREFIX=${DATA_PREFIX:-$DATASETS/scibase_sciall_train/sciall_train_text_document}
export DATA_VALID_PREFIX=${DATA_VALID_PREFIX:-$DATASETS/scibase_sciallval_s200-208/sciallval_s200-208_text_document}
if [ -z "${EPOCHS:-}" ]; then
  export TRAIN_ITERS=${TRAIN_ITERS:-4} TRAIN_WARMUP=${TRAIN_WARMUP:-1}
  export TRAIN_EVAL_ITERS=${TRAIN_EVAL_ITERS:-1} TRAIN_EVAL_INTERVAL=${TRAIN_EVAL_INTERVAL:-2}
fi
GBS=${TRAIN_GBS:-512}
# DP = nodes, micro-batch 1: the global batch must split evenly over the nodes.
(( GBS % N == 0 )) || { echo "ERROR: TRAIN_GBS=$GBS is not divisible by $N nodes" >&2; exit 2; }
[ "$N" -ge 4 ] || echo "warning: $N node(s) is below the tested minimum of 4 (memory per GPU grows as DP shrinks)" >&2

CHAT_SHARE=${CHAT_SHARE:-0.1}
ARGS="$TOKENIZER_ARGS"
if [ "$CHAT_SHARE" != 0 ]; then
  SCI_SHARE=$(awk -v c="$CHAT_SHARE" 'BEGIN { printf "%g", 1 - c }')
  ARGS+=" --train_data_path=['$SCI_SHARE','$DATA_PREFIX','$CHAT_SHARE','$DATASETS/chatmix_v7/chatmix_v7_text_document']"
fi
[ -n "${SAVE_INTERVAL:-}" ] && ARGS+=" --save /work/output --save_interval $SAVE_INTERVAL --disable_last_saving false"
export PRIMUS_ARGS="$ARGS ${EXTRA_PRIMUS_ARGS:-}"

NAME=${RUN_NAME:-cpt-N$N}
export RUN_NAME=$NAME
mkdir -p logs/train
JID=$(lux_sbatch --parsable -N "$N" -t "${TIME:-00:45:00}" -J "$NAME" lux/train.sbatch | tail -1)
echo "job $JID  ($N nodes, TRAIN_ITERS=${TRAIN_ITERS:-from EPOCHS}, gbs $GBS, chat share $CHAT_SHARE)"
echo "  job log:   $WORKDIR/logs/train/slurm-${NAME}_$JID.out"
echo "  ranks log: $WORKDIR/logs/train/${NAME}_n${N}_j${JID}_<date>.log"
[ -n "${SAVE_INTERVAL:-}" ] && echo "  checkpoints: $WORKDIR/output/amd/root/${NAME}_n${N}_j${JID}_<date>/checkpoints/"
exit 0
