#!/usr/bin/env bash
# Export run_cpt.sh checkpoints to Hugging Face safetensors (tutorial 3's lux/export_hf.sbatch, 1 node, ~1 h),
# with its three checks: base-model round trip bitwise equal, shape/dtype comparison, held-out loss.
#   scripts/export_cpt.sh <checkpoints dir> "<iters>" [out dir]    e.g.
#   scripts/export_cpt.sh $CPT_WORKDIR/output/amd/root/cpt-N4_n4_j123_1013-1200/checkpoints "20" $CPT_WORKDIR/hf/my-cpt
# Each iteration lands in <out>/iter_NNNNNNN/, servable by vLLM (vllm-skill, bench/mistral). LOSS_CHECK=0 skips check 3.
set -eu
SK=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
source "$SK/env.lux.sh"
export EXPORT_CKPTS=${1:?usage: export_cpt.sh <checkpoints dir> "<iters>" [out dir]} EXPORT_ITERS=${2:?iterations, e.g. \"20\"}
export EXPORT_OUT=${3:-$CPT_WORKDIR/hf/$(basename "$(dirname "$EXPORT_CKPTS")")}
export WORKDIR=$CPT_WORKDIR SIF=$CPT_SIF
source "$PKG/tutorials/03_mistral_cpt/hackathon_env.sh"
cd "$WORKDIR"
source lux/lux_env.sh
mkdir -p logs/export
JID=$(lux_sbatch --parsable -t "${TIME:-04:00:00}" lux/export_hf.sbatch | tail -1)
echo "job $JID  log $WORKDIR/logs/export/slurm-export-hf_$JID.out  out $EXPORT_OUT/"
