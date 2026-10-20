#!/bin/bash -l
# Mistral-Large-Instruct-2411 (123B) continued pretraining on OLCF Lux: tutorial 3's tested lux/train.sbatch
# (FP8, TP1 x PP8, one model copy per node, data parallelism = nodes), 90% Sci-Base + 10% chat mix, from the
# Megatron TP1/PP8 checkpoint in the shared package. No #SBATCH lines: VISTA writes the header from
# cluster_defaults.json. setup_lux.sh has already made the work dir on the login node.
#
# Env:
#   RUN_DIR_Lux   this folder's files (summarize_cpt.sh); uploaded by VISTA, only read
#   VISTA_OUT     per-job output dir (train.log, summary.txt)
#   VISTA_JOB_DIR shared by every run: work/ holds Primus, caches, the ranks' logs and checkpoints
#   PKG           the shared hackathon package (models, data, image, tutorial scripts)
# Args ("$@", VISTA's script_args): KEY=VALUE pairs, see README.md.
set -u
: "${RUN_DIR_Lux:?RUN_DIR_Lux not set}" "${VISTA_OUT:?VISTA_OUT not set}" "${PKG:?PKG not set}"
CPT_WORKDIR=${CPT_WORKDIR:-${VISTA_JOB_DIR:?VISTA_JOB_DIR not set}/work}
mkdir -p "$VISTA_OUT"
N=$SLURM_JOB_NUM_NODES

TRAIN_ITERS= TRAIN_GBS=512 TRAIN_WARMUP= TRAIN_EVAL_INTERVAL= TRAIN_EVAL_ITERS=1 CHAT_SHARE=0.1 SAVE_INTERVAL= \
  LUX_GPU_CHECK=0 RUN_NAME=
for kv in "$@"; do
  case "$kv" in
    TRAIN_ITERS=*|TRAIN_GBS=*|TRAIN_WARMUP=*|TRAIN_EVAL_INTERVAL=*|TRAIN_EVAL_ITERS=*|CHAT_SHARE=*|SAVE_INTERVAL=*|\
    LUX_GPU_CHECK=*|RUN_NAME=*) declare "$kv" ;;
    *) echo "ERROR: unknown argument '$kv' (accepted: TRAIN_ITERS= TRAIN_GBS= TRAIN_WARMUP= TRAIN_EVAL_INTERVAL=" \
            "TRAIN_EVAL_ITERS= CHAT_SHARE= SAVE_INTERVAL= LUX_GPU_CHECK= RUN_NAME=)" >&2; exit 2 ;;
  esac
done
for v in TRAIN_ITERS TRAIN_GBS TRAIN_WARMUP TRAIN_EVAL_INTERVAL TRAIN_EVAL_ITERS SAVE_INTERVAL; do
  [[ -z ${!v} || ${!v} =~ ^[0-9]+$ ]] || { echo "ERROR: $v must be a whole number, not '${!v}'" >&2; exit 2; }
done
[[ $CHAT_SHARE =~ ^(0(\.[0-9]+)?|1)$ ]] || { echo "ERROR: CHAT_SHARE must be between 0 and 1, not '$CHAT_SHARE'" >&2; exit 2; }
[[ $RUN_NAME =~ ^[A-Za-z0-9._-]*$ ]] || { echo "ERROR: RUN_NAME may use only letters, digits, '.', '_' and '-'" >&2; exit 2; }
case "$LUX_GPU_CHECK" in 0|1) ;; *) echo "ERROR: LUX_GPU_CHECK must be 0 or 1" >&2; exit 2 ;; esac
# DP = nodes and micro-batch 1, so the global batch must split evenly over the nodes.
(( TRAIN_GBS % N == 0 )) || { echo "ERROR: TRAIN_GBS=$TRAIN_GBS is not divisible by $N nodes; use a multiple of $N" >&2; exit 2; }
[ -f "$CPT_WORKDIR/lux/lux_env.sh" ] || { echo "ERROR: no work dir at $CPT_WORKDIR (setup_lux.sh makes it)" >&2; exit 3; }

# The tutorial's defaults (4 steps, warm-up 1, evaluate 1 batch every 2 steps); a longer run evaluates
# at its last step unless TRAIN_EVAL_INTERVAL says otherwise.
if [ -z "$TRAIN_ITERS" ]; then TRAIN_ITERS=4; TRAIN_EVAL_INTERVAL=${TRAIN_EVAL_INTERVAL:-2}; fi
TRAIN_WARMUP=${TRAIN_WARMUP:-1} TRAIN_EVAL_INTERVAL=${TRAIN_EVAL_INTERVAL:-$TRAIN_ITERS}
RUN_NAME=${RUN_NAME:-cpt-N$N}

# The tutorial's own env layers, in its order (as scripts/run_cpt.sh does by hand).
export ACCOUNT=${SLURM_JOB_ACCOUNT:?} WORKDIR=$CPT_WORKDIR SIF=${CPT_SIF:-$PKG/containers/primus_v26.5.sif}
source "$PKG/tutorials/03_mistral_cpt/hackathon_env.sh"
cd "$WORKDIR"
source lux/lux_env.sh
export CONFIG=/work/lux/configs/mistral_large_2411-lux-cpt-4node-fp8-tp1pp8.yaml
export DATA_PREFIX=$DATASETS/scibase_sciall_train/sciall_train_text_document
export DATA_VALID_PREFIX=$DATASETS/scibase_sciallval_s200-208/sciallval_s200-208_text_document
export TRAIN_ITERS TRAIN_GBS TRAIN_WARMUP TRAIN_EVAL_INTERVAL TRAIN_EVAL_ITERS RUN_NAME LUX_GPU_CHECK
ARGS="$TOKENIZER_ARGS"
if [ "$CHAT_SHARE" != 0 ]; then
  SCI_SHARE=$(awk -v c="$CHAT_SHARE" 'BEGIN { printf "%g", 1 - c }')
  ARGS+=" --train_data_path=['$SCI_SHARE','$DATA_PREFIX','$CHAT_SHARE','$DATASETS/chatmix_v7/chatmix_v7_text_document']"
fi
[ -n "$SAVE_INTERVAL" ] && ARGS+=" --save /work/output --save_interval $SAVE_INTERVAL --disable_last_saving false"
export PRIMUS_ARGS=$ARGS

# Known bad nodes: VISTA cannot pass --exclude, so name any we were given; the rails preflight below stops
# the job if one is really broken.
BAD=$(comm -12 <(lux_exclude | tr , '\n' | sort -u) <(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -u) | paste -sd,)
{
  echo "job $SLURM_JOB_ID nodes=$N ($SLURM_JOB_NODELIST) run=$RUN_NAME iters=$TRAIN_ITERS gbs=$TRAIN_GBS" \
       "warmup=$TRAIN_WARMUP eval=${TRAIN_EVAL_ITERS}x every $TRAIN_EVAL_INTERVAL chat=$CHAT_SHARE" \
       "save=${SAVE_INTERVAL:-off} workdir=$WORKDIR"
  [ -n "$BAD" ] && echo "warning: allocated node(s) on the bad-node list: $BAD"
  [ "$N" -ge 4 ] || echo "warning: $N node(s) is below the tested minimum of 4 (memory per GPU grows as DP shrinks)"
} | tee "$VISTA_OUT/train.log"

# train.sbatch runs from the work dir: rails preflight, optional GPU check, then lux/launch.sh (which writes
# the ranks' log to logs/train/<run>.log, and prints it, so train.log here has it too).
SLURM_SUBMIT_DIR=$WORKDIR bash lux/train.sbatch 2>&1 | tee -a "$VISTA_OUT/train.log"
rc=${PIPESTATUS[0]}
bash "$RUN_DIR_Lux/summarize_cpt.sh" "$VISTA_OUT/train.log" "$rc" | tee "$VISTA_OUT/summary.txt"
exit "$rc"
