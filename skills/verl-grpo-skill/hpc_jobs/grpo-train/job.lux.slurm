#!/bin/bash -l
# Run tutorial 4's Ray + verl GRPO driver inside VISTA's allocation.
# VISTA supplies the #SBATCH header; this file intentionally has no #SBATCH directives.
set -uo pipefail
: "${RUN_DIR_Lux:?RUN_DIR_Lux not set}" "${VISTA_OUT:?VISTA_OUT not set}"
: "${VISTA_JOB_DIR:?VISTA_JOB_DIR not set}" "${PKG:?PKG not set}"
mkdir -p "$VISTA_OUT"
WORKDIR=${WORKDIR:-$VISTA_JOB_DIR/work}

MODE=smoke TARGET_STEP= RUN_NAME= GPU_CHECK=1 DIST_TIMEOUT_MIN=90
for kv in "$@"; do
  case $kv in
    MODE=*|TARGET_STEP=*|RUN_NAME=*|GPU_CHECK=*|DIST_TIMEOUT_MIN=*)
      key=${kv%%=*}; value=${kv#*=}; printf -v "$key" '%s' "$value" ;;
    *) echo "ERROR: unknown argument '$kv' (use MODE, TARGET_STEP, RUN_NAME, GPU_CHECK, DIST_TIMEOUT_MIN)" >&2; exit 2 ;;
  esac
done
case $MODE in smoke|control|pubonly) ;; *) echo "ERROR: MODE must be smoke, control, or pubonly" >&2; exit 2 ;; esac
case $GPU_CHECK in 0|1) ;; *) echo "ERROR: GPU_CHECK must be 0 or 1" >&2; exit 2 ;; esac
[[ $DIST_TIMEOUT_MIN =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: DIST_TIMEOUT_MIN must be a positive integer" >&2; exit 2; }
[[ -z $RUN_NAME || $RUN_NAME =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "ERROR: invalid RUN_NAME" >&2; exit 2; }
[[ -z $TARGET_STEP || $TARGET_STEP =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: TARGET_STEP must be a positive integer" >&2; exit 2; }
N=${SLURM_JOB_NUM_NODES:?SLURM_JOB_NUM_NODES not set}
case $N in 8|32) ;; *) echo "ERROR: this 123B recipe was validated at 8 and 32 nodes" >&2; exit 2 ;; esac
[ -f "$WORKDIR/verl_rl/phase2.sh" ] || { echo "ERROR: run setup_lux.sh first ($WORKDIR is missing)" >&2; exit 3; }

if [ "$MODE" = smoke ]; then
  TARGET_STEP=${TARGET_STEP:-3}
  [ "$TARGET_STEP" = 3 ] || [ "$TARGET_STEP" = 10 ] || { echo "ERROR: smoke uses 3 or 10 steps" >&2; exit 2; }
  RUN_NAME=${RUN_NAME:-smoke_$SLURM_JOB_ID}
  ARM=control
  export RL_SAVE_FREQ=-1 RL_TEST_FREQ=-1 RL_VAL_BEFORE=False
else
  TARGET_STEP=${TARGET_STEP:-25}
  (( TARGET_STEP <= 125 && TARGET_STEP % 25 == 0 )) || {
    echo "ERROR: TARGET_STEP must be 25, 50, 75, 100, or 125" >&2; exit 2;
  }
  RUN_NAME=${RUN_NAME:-rl_$MODE}
  ARM=$MODE
  # VISTA jobs are limited to 4 hours. A submission advances at most 25 steps.
  CKPT_DIR=$WORKDIR/work/ckpt/$RUN_NAME
  LATEST=0
  for d in "$CKPT_DIR"/global_step_*; do
    [ -d "$d" ] || continue
    s=${d##*global_step_}
    [[ $s =~ ^[0-9]+$ ]] && (( s > LATEST )) && LATEST=$s
  done
  (( TARGET_STEP > LATEST && TARGET_STEP - LATEST <= 25 )) || {
    echo "ERROR: latest checkpoint is step $LATEST; choose a higher TARGET_STEP no more than 25 steps ahead" >&2; exit 2;
  }
  echo "resuming $RUN_NAME from latest checkpoint step $LATEST toward $TARGET_STEP"
fi

# hackathon_env.sh must precede lux/env.sh: it selects the shared package inputs.
export ACCOUNT=${SLURM_JOB_ACCOUNT:?SLURM_JOB_ACCOUNT not set} WORKDIR
source "$PKG/tutorials/04_mistral_rl/hackathon_env.sh"
cd "$WORKDIR"
source lux/env.sh
export RL_ARM=$ARM RL_RUN=$RUN_NAME RL_STEPS=$TARGET_STEP RL_LR=2e-7
export RL_DIST_TIMEOUT_MIN=$DIST_TIMEOUT_MIN
export RL_GPU_CHECK=$GPU_CHECK RL_DRIVER=verl_rl/phase2.sh
echo "job $SLURM_JOB_ID: mode=$MODE run=$RUN_NAME target=$TARGET_STEP nodes=$N workdir=$WORKDIR" | tee "$VISTA_OUT/job.log"
SLURM_SUBMIT_DIR=$WORKDIR bash ray_slurm/run_verl.sbatch 2>&1 | tee "$VISTA_OUT/train.log"
TRAIN_RC=${PIPESTATUS[0]}
python3 "$RUN_DIR_Lux/summarize.py" "$VISTA_OUT/train.log" \
  --mode "$MODE" --target-step "$TARGET_STEP" --run-name "$RUN_NAME" \
  --exit-code "$TRAIN_RC" --workdir "$WORKDIR" | tee "$VISTA_OUT/summary.txt"
SUMMARY_RC=${PIPESTATUS[0]}
echo "train exit $TRAIN_RC; summary exit $SUMMARY_RC" >> "$VISTA_OUT/job.log"
[ "$TRAIN_RC" -eq 0 ] || exit "$TRAIN_RC"
exit "$SUMMARY_RC"
