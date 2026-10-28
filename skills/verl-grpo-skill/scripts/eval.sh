#!/usr/bin/env bash
# Make a vLLM-readable HF directory from a GRPO checkpoint and submit tutorial 2's science benchmark.
set -euo pipefail
RUN=${1:?usage: scripts/eval.sh <run name> <checkpoint step> [model name]}
STEP=${2:?usage: scripts/eval.sh <run name> <checkpoint step> [model name]}
[[ $RUN =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ && $STEP =~ ^[1-9][0-9]*$ ]] || { echo "invalid run name or step" >&2; exit 2; }
MODEL_NAME=${3:-${RUN}_s${STEP}}
[[ $MODEL_NAME =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "invalid model name" >&2; exit 2; }
SKILL_DIR=$(cd "$(dirname "$0")/.." && pwd)
source "$SKILL_DIR/env.lux.sh"
: "${ACCOUNT:?set ACCOUNT to your Lux Slurm account}"
source "$PKG/tutorials/04_mistral_rl/hackathon_env.sh"
STEP_DIR=$WORKDIR/work/ckpt/$RUN/global_step_$STEP
[ -d "$STEP_DIR" ] || { echo "checkpoint not found: $STEP_DIR" >&2; exit 1; }
mkdir -p "$WORKDIR/hf"
bash "$WORKDIR/verl_rl/hf_eval_dir.sh" "$STEP_DIR" "$HFMODELS/Mistral-Large-Instruct-2411" "$WORKDIR/hf/$MODEL_NAME"
cd "$ZENITH_DIR"
ZENITH_MODELPATH="$WORKDIR/hf" ML_MODEL="$MODEL_NAME" \
ZENITH_DATASETSPATH="$DATASETS/benchmarks" \
ZENITH_IMAGE_TAR="$PKG/containers/inference_image_gfx950_v8mi250_20261011.tar" \
ZENITH_MAX_MODEL_LEN=131072 ML_STEPS="correctness tool gsm8k science" \
  sbatch -A "$ACCOUNT" -p batch bench/mistral/mistral_bench.slurm.sh
