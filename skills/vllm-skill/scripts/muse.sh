#!/usr/bin/env bash
# Muse Spark Open on one Lux node.   scripts/muse.sh [KEY=VAL ...]  |  scripts/muse.sh download <dir> <what>...
#   scripts/muse.sh                                    # serve the shared MXFP4 checkpoint (env.lux.sh: MUSE_MODELPATH)
#   scripts/muse.sh BENCH=gsm8k                        # serve, score gsm8k (250), exit
#   scripts/muse.sh MODEL=Muse-Spark-Open-816B-A42B MODELPATH=<dir>   # bf16, from your own download
#   scripts/muse.sh download <dir> bf16                # login node: bf16 checkpoint + DSpark draft into <dir>
#   scripts/muse.sh download <dir> mxfp4 | drafts      # a private MXFP4 copy + draft / the drafts only
# Submits the same hpc_jobs/muse-serve/job.lux.slurm VISTA does; KEY=VAL pairs are its script_args. Outputs land in
# runs/muse.<jobid>/. SBATCH_ARGS adds sbatch options (e.g. "-t 04:00:00 --exclude=lux123").
set -eu
SK=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
source "$SK/env.lux.sh"
if [ "${1:-}" = download ]; then
  # Hugging Face through the OLCF proxy. bf16 is gated: request access on its model page (Meta's form, Muse Model
  # License), then `hf auth login` once. The MXFP4 checkpoint and the drafts are public. zenith v0.1.9 README.md §1d.1.
  D=${2:?usage: scripts/muse.sh download <dir> mxfp4|bf16|drafts ...}; shift 2
  [ $# -gt 0 ] || { echo "say what to download: mxfp4, bf16 and/or drafts" >&2; exit 2; }
  export http_proxy=http://proxy.ccs.ornl.gov:3128 https_proxy=http://proxy.ccs.ornl.gov:3128
  mkdir -p "$D"
  for what in "$@"; do
    case "$what" in
      mxfp4|bf16)
        command -v hf >/dev/null || { echo "no hf CLI: python3 -m pip install --user 'huggingface_hub[cli]==1.33.0' (with the proxy above)" >&2; exit 2; }
        [ "$what" = mxfp4 ] && R=amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4 || R=meta-models/Muse-Spark-Open-816B-A42B
        hf download "$R" --local-dir "$D/${R#*/}"   # resumes; rerun after an interruption
        ;;
      drafts) ;;
      *) echo "unknown: $what (mxfp4, bf16, drafts)" >&2; exit 2 ;;
    esac
  done
  # The DSpark draft (and two DFlash2 drafts the job never uses), sha256-checked, resumable; writes <draft>/.verified.
  bash "$MUSE_JOB_DIR/lux-inference/bench/muse/spec_drafts_download.sh" "$D"
  echo "done. Serve from it: scripts/muse.sh MODELPATH=$D [MODEL=...]  (VISTA: script_args=\"MODELPATH=$D ...\")"; exit 0
fi
export RUN_DIR_Lux=$MUSE_JOB_DIR LUX_MODELPATH=$MUSE_MODELPATH LUX_IMAGE_TAR=$MUSE_IMAGE_TAR
mkdir -p "$LUX_RUNS_DIR"
bash "$MUSE_JOB_DIR/setup_lux.sh" >/dev/null || { bash "$MUSE_JOB_DIR/setup_lux.sh"; exit 1; }
T=02:00:00; case " $* " in *" BUILD_IMAGE=1 "*) T=04:00:00 ;; esac
JID=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -N 1 -J "${JOB_NAME:-muse.serve}" -t "$T" \
      --exclusive --gpus-per-node=8 --ntasks-per-node=1 \
      -o "$LUX_RUNS_DIR/%x.%j.out" -e "$LUX_RUNS_DIR/%x.%j.out" ${SBATCH_ARGS:-} "$MUSE_JOB_DIR/job.lux.slurm" "$@" | tail -1)
echo "job $JID  log $LUX_RUNS_DIR/${JOB_NAME:-muse.serve}.$JID.out  outputs $LUX_RUNS_DIR/muse.$JID/"
