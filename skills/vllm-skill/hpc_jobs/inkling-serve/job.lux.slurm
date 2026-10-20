#!/bin/bash -l
# vLLM on OLCF Lux: serve a model as an OpenAI-compatible endpoint, or serve it, benchmark it and exit.
# Carries no #SBATCH lines: resources come from cluster_defaults.json under VISTA (which writes the
# header), and from scripts/serve.sh / scripts/bench.sh by hand. setup_lux.sh has already run on the login node.
#
# Env:
#   RUN_DIR_Lux   this folder's files (the lux-inference/ engine, summarize.sh); uploaded by VISTA, only read
#   VISTA_OUT     per-job output dir; VISTA creates it, by hand it defaults to $LUX_RUNS_DIR/serve.<jobid>
#   LUX_MODELPATH, LUX_IMAGE_TAR, LUX_DATASETSPATH   the shared package (cluster_defaults.json / env.lux.sh)
# Args ("$@", VISTA's script_args): KEY=VALUE pairs, see README.md. By hand, scripts/serve.sh exports
# LUX_* knobs instead, and every LUX_* in the environment still reaches the engine.
set -u
: "${RUN_DIR_Lux:?RUN_DIR_Lux not set}"
VISTA_OUT=${VISTA_OUT:-${LUX_RUNS_DIR:?set VISTA_OUT or LUX_RUNS_DIR}/serve.$SLURM_JOB_ID}
mkdir -p "$VISTA_OUT"

for kv in "$@"; do
  case "$kv" in
    MODEL=*)         export LUX_MODEL="${kv#*=}" ;;
    BENCH=*)         export LUX_BENCH="${kv#*=}"; LUX_BENCH=${LUX_BENCH//,/ } ;;
    SCIENCE_N=*)     export LUX_SCIENCE_ARGS="--n ${kv#*=}" ;;
    MTP=*)           export LUX_MTP="${kv#*=}" ;;
    MAX_MODEL_LEN=*) export LUX_MAX_MODEL_LEN="${kv#*=}" ;;
    *) echo "ERROR: unknown argument '$kv' (accepted: MODEL= BENCH= SCIENCE_N= MTP= MAX_MODEL_LEN=)" >&2; exit 2 ;;
  esac
done
case "${LUX_MODEL:-Inkling-MXFP4}" in
  Inkling-MXFP4|Inkling-bf16|Mistral-Large-Instruct-2411) ;;
  *) echo "ERROR: MODEL must be Inkling-MXFP4, Inkling-bf16 or Mistral-Large-Instruct-2411" >&2; exit 2 ;;
esac
for s in ${LUX_BENCH:-}; do
  case "$s" in science|mmlu|needle) ;; *) echo "ERROR: BENCH steps are science, mmlu, needle (comma-separated)" >&2; exit 2 ;; esac
done
case "${LUX_MTP:-0}" in 0|1) ;; *) echo "ERROR: MTP must be 0 or 1" >&2; exit 2 ;; esac
[ -n "${LUX_SCIENCE_ARGS:-}" ] && [ -z "${LUX_BENCH:-}" ] && export LUX_BENCH=science

: "${LUX_MODELPATH:?LUX_MODELPATH not set (cluster_defaults.json or env.lux.sh)}"
: "${LUX_IMAGE_TAR:?LUX_IMAGE_TAR not set (cluster_defaults.json or env.lux.sh)}"
M=${LUX_MODEL:-Inkling-MXFP4}
[ -r "$LUX_MODELPATH/$M/config.json" ] || { echo "ERROR: no model $M in $LUX_MODELPATH" >&2; exit 2; }
case "$M:$SLURM_JOB_NUM_NODES" in
  Inkling-MXFP4:[12]|Inkling-bf16:2|Mistral-Large-Instruct-2411:1) ;;
  *) echo "ERROR: $M does not run on $SLURM_JOB_NUM_NODES nodes (Inkling-MXFP4: 1 or 2, Inkling-bf16: 2, Mistral-Large-Instruct-2411: 1)" >&2; exit 2 ;;
esac
# Outputs and the engine's per-job state go under VISTA_OUT; the uploaded sources are never written.
export LUX_SERVE_OUT=$VISTA_OUT ZENITH_STATE_DIR=$VISTA_OUT/state

# Run the engine from a per-job copy (15 MB) on Lustre, which every node sees. The RCCL-ANP plugin needs
# libionic.so.1 and libionic-rdmav59.so to be symlinks to ONE file (copies load it twice), and a skill
# import refuses symlinks, so the job folder ships the file once and the links are made here.
E=$VISTA_OUT/engine
rm -rf "$E"; mkdir -p "$E"
cp -a "$RUN_DIR_Lux/lux-inference/docker" "$RUN_DIR_Lux/lux-inference/bench" "$E/" || { echo "ERROR: cannot stage the engine in $E" >&2; exit 3; }
rm -rf "$E"/docker/.zenith_state "$E"/docker/serve.*
( cd "$E/docker/patches/anp" && ln -sfn libionic.so.1.1.61 libionic.so.1 && ln -sfn libionic.so.1.1.61 libionic-rdmav59.so ) \
  || { echo "ERROR: cannot link libionic in $E" >&2; exit 3; }
MODE=serve; [ -n "${LUX_BENCH:-}" ] && MODE="bench ($LUX_BENCH${LUX_SCIENCE_ARGS:+, science $LUX_SCIENCE_ARGS})"
echo "job $SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES ($SLURM_JOB_NODELIST) model=$M mode=$MODE out=$VISTA_OUT" \
  | tee "$VISTA_OUT/job.log"

# summary.txt (and endpoint.txt once serving) are refreshed every minute: a serve job ends at its time
# limit, so there is no "after" to write them in.
refresh() {
  bash "$RUN_DIR_Lux/summarize.sh" "$VISTA_OUT" > "$VISTA_OUT/summary.txt.tmp" 2>&1 \
    && mv -f "$VISTA_OUT/summary.txt.tmp" "$VISTA_OUT/summary.txt"
  url=$(grep -oE '^=== SERVING at [^ ]+' "$VISTA_OUT/job.log" | tail -1 | cut -d' ' -f4)
  [ -n "$url" ] && echo "$url" > "$VISTA_OUT/endpoint.txt"
  return 0
}
( while sleep 60; do refresh; done ) &
WATCH=$!

# The allocation asks for --gpus-per-node (VISTA's header on Lux), and srun inherits it from
# SLURM_GPUS_PER_NODE; the engine's srun asks for --gres=gpu:8, and Slurm refuses the two together.
unset SLURM_GPUS_PER_NODE
# serve_lux.slurm.sh finds its siblings from SLURM_SUBMIT_DIR.
SLURM_SUBMIT_DIR=$E/docker bash "$E/docker/serve_lux.slurm.sh" 2>&1 | tee -a "$VISTA_OUT/job.log"
rc=${PIPESTATUS[0]}
kill "$WATCH" 2>/dev/null
rm -rf "$E"   # the staged engine; a serve job killed at its time limit leaves it (15 MB)
echo "=== exit $rc" | tee -a "$VISTA_OUT/job.log"
refresh
cat "$VISTA_OUT/summary.txt"
exit "$rc"
