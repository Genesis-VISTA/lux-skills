#!/bin/bash -l
# ICF 4-agent capsule optimizer on OLCF Lux, headless: a campaign or a laser-energy sweep against an
# OpenAI-compatible endpoint, writing the result, the agents' transcript and the media to $VISTA_OUT.
# Carries no #SBATCH lines: resources come from cluster_defaults.json under VISTA (which writes the header),
# and from scripts/campaign.sh by hand. The job builds its own Python venv (below); nothing runs on a login node.
#
# Env:
#   RUN_DIR_Lux    this folder's files (icf_4agent/, run_campaign.py); uploaded by VISTA, only read
#   VISTA_OUT      per-job output dir; VISTA creates it, by hand it defaults to $ICF_RUNS_DIR/icf.<jobid>
#   VISTA_JOB_DIR  shared by every run: venv-<pins>.tar, the app's venv (built by the first job). By hand:
#                  ICF_VENV_CACHE. ICF_NVME=0 ignores the NVMe and uses /tmp, as a job without one does.
#   ICF_GPTOSS_MODEL, ICF_VLLM_IMAGE_TAR   BACKEND=self only (cluster_defaults.json / env.lux.sh)
# Args ("$@", VISTA's script_args): KEY=VALUE pairs, see README.md.
set -u
: "${RUN_DIR_Lux:?RUN_DIR_Lux not set}"
VISTA_OUT=${VISTA_OUT:-${ICF_RUNS_DIR:?set VISTA_OUT or ICF_RUNS_DIR}/icf.$SLURM_JOB_ID}
mkdir -p "$VISTA_OUT"
CACHE=${ICF_VENV_CACHE:-${VISTA_JOB_DIR:?set ICF_VENV_CACHE or VISTA_JOB_DIR}}
# Every failure leaves a summary.txt, so the reason is where VISTA (and the reader) looks first.
fail() { rc=$1; shift; echo "ERROR: $*" | tee -a "$VISTA_OUT/job.log" >&2; echo "=== CAMPAIGN FAILED: $*" > "$VISTA_OUT/summary.txt"; exit "$rc"; }
VPID= TMPV=
cleanup() { [ -n "$VPID" ] && { kill -TERM "$VPID" 2>/dev/null; wait "$VPID" 2>/dev/null; }; [ -n "$TMPV" ] && rm -rf "$TMPV"; }
trap cleanup EXIT

BACKEND=endpoint ENDPOINT= SERVE_JOB= ENERGY= SWEEP= MOVIE=1 DEBRIEF=1
for kv in "$@"; do
  case "$kv" in
    BACKEND=*)          BACKEND="${kv#*=}" ;;
    ENDPOINT=*)         ENDPOINT="${kv#*=}" ;;
    SERVE_JOB=*)        SERVE_JOB="${kv#*=}" ;;
    ENERGY_MJ=*)        ENERGY="${kv#*=}" ;;
    SWEEP_MJ=*)         SWEEP="${kv#*=}" ;;
    MOVIE=*)            MOVIE="${kv#*=}" ;;
    DEBRIEF=*)          DEBRIEF="${kv#*=}" ;;
    REASONING_EFFORT=*) export ICF_REASONING_EFFORT="${kv#*=}" ;;
    MAX_TOKENS=*)       export ICF_MAX_TOKENS="${kv#*=}" ;;
    TEMPERATURE=*)      export ICF_TEMPERATURE="${kv#*=}" ;;
    MAX_DELTA=*)        export ICF_MAX_DELTA="${kv#*=}" ;;
    MAX_ROUGHNESS=*)    export ICF_MAX_ROUGHNESS="${kv#*=}" ;;
    *) fail 2 "unknown argument '$kv' (accepted: BACKEND= ENDPOINT= SERVE_JOB= ENERGY_MJ= SWEEP_MJ= MOVIE= DEBRIEF= REASONING_EFFORT= MAX_TOKENS= TEMPERATURE= MAX_DELTA= MAX_ROUGHNESS=)" ;;
  esac
done
num='^[0-9]+([.][0-9]+)?$'
case "$BACKEND" in endpoint|self) ;; *) fail 2 "BACKEND must be endpoint or self" ;; esac
[ -n "$ENERGY" ] && [ -n "$SWEEP" ] && fail 2 "give ENERGY_MJ or SWEEP_MJ, not both"
[ -z "$ENERGY" ] || [[ "$ENERGY" =~ $num ]] || fail 2 "ENERGY_MJ must be a number of MJ, e.g. 3"
[ -z "$SWEEP" ] || [[ "$SWEEP" =~ ^[0-9.]+:[0-9.]+:[0-9.]+$ ]] || fail 2 "SWEEP_MJ is LO:HI:STEP in MJ, e.g. 2:4:1"
case "$MOVIE$DEBRIEF" in [01][01]) ;; *) fail 2 "MOVIE and DEBRIEF are 0 or 1" ;; esac
case "${ICF_REASONING_EFFORT:-medium}" in none|minimal|low|medium|high|max) ;; *) fail 2 "REASONING_EFFORT is none, minimal, low, medium, high or max" ;; esac
# The app's Anthropic backend would need an API key in the job's environment: not in this skill.
unset ICF_BACKEND ANTHROPIC_API_KEY ICF_ANTHROPIC_API_KEY

# Run from a per-job copy: the app writes media_output/ next to itself, and the uploaded files are only read.
A=$VISTA_OUT/app
rm -rf "$A"; cp -a "$RUN_DIR_Lux/icf_4agent" "$A" || fail 3 "cannot stage the app in $A"
echo "job $SLURM_JOB_ID on $(hostname -s) backend=$BACKEND args: $*" | tee "$VISTA_OUT/job.log"
# Everything runs on this node (no srun); the venv is on its local disk. Extra nodes would sit idle.
[ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ] && echo "WARNING: this job has $SLURM_JOB_NUM_NODES nodes but uses only $(hostname -s); submit with node_count 1" | tee -a "$VISTA_OUT/job.log"

# The app's venv, on node-local disk: from a compute node, Lustre takes ~35 ms or more per small file written
# (a venv is ~22,000 files), where the node's own disk takes none. The job uses the node's NVMe (/mnt/bb/<jobid>)
# when it has one, else /tmp: jobs submitted without the nvme constraint (VISTA's S3M submissions) have no
# /mnt/bb/<jobid>. The first job builds the venv (~20 s) and saves it as one tar in $CACHE, keyed by the pins;
# later jobs unpack the tar (~1 s). Lustre only ever holds that one file. The venv's path differs per job, so it is
# only run as $VENV/bin/python (its console scripts would not work).
key=$( (cat "$A/requirements.txt" "$A/constraints.txt"; echo "py3.11 imageio-ffmpeg") | sha256sum | cut -c1-16)
TAR=$CACHE/venv-$key.tar
NVME=/mnt/bb/$SLURM_JOB_ID
if [ "${ICF_NVME:-1}" != 0 ] && [ -d "$NVME" ] && [ -w "$NVME" ]; then
  LOCAL=$NVME where="NVMe ($NVME)"
else
  # /tmp may be memory-backed: the venv (~1.5 GB unpacked) is removed when the job exits.
  LOCAL=/tmp/icf-venv.$SLURM_JOB_ID where="/tmp, no NVMe on this job ($LOCAL)" TMPV=$LOCAL
fi
VENV=$LOCAL/venv
mkdir -p "$LOCAL" "$CACHE" || fail 3 "cannot create $LOCAL or $CACHE"
t0=$(date +%s)
if [ -s "$TAR" ]; then
  tar -xf "$TAR" -C "$LOCAL" || fail 3 "cannot unpack $TAR into $LOCAL; delete the tar to rebuild"
  echo "$(date +%T) venv: unpacked $TAR onto $where in $(( $(date +%s) - t0 )) s" | tee -a "$VISTA_OUT/job.log"
else
  echo "$(date +%T) venv: building on $where, first job only" | tee -a "$VISTA_OUT/job.log"
  # PyPI is reachable from compute nodes only through the OLCF proxy, set here for pip alone.
  PROXY=${ICF_PIP_PROXY:-http://proxy.ccs.ornl.gov:3128}
  pip_() { https_proxy=$PROXY http_proxy=$PROXY PIP_CACHE_DIR=$LOCAL/pip "$VENV/bin/python" -m pip "$@"; }
  { /usr/bin/python3.11 -m venv "$VENV" && pip_ install -q --upgrade pip \
    && pip_ install -q -r "$A/requirements.txt" -c "$A/constraints.txt" imageio-ffmpeg; } > "$VISTA_OUT/venv_build.log" 2>&1 \
    || { tail -20 "$VISTA_OUT/venv_build.log" >&2; fail 3 "venv build failed; see venv_build.log"; }
  rm -rf "$LOCAL/pip"
  echo "$(date +%T) venv built in $(( $(date +%s) - t0 )) s" | tee -a "$VISTA_OUT/job.log"
  # Written under a temporary name first, so no job ever unpacks a half-written tar.
  if tar -cf "$TAR.$SLURM_JOB_ID" -C "$LOCAL" venv && mv -f "$TAR.$SLURM_JOB_ID" "$TAR"; then
    echo "$(date +%T) venv saved to $TAR ($(du -h --apparent-size "$TAR" | cut -f1))" | tee -a "$VISTA_OUT/job.log"
  else
    rm -f "$TAR.$SLURM_JOB_ID"; echo "WARNING: could not save the venv to $CACHE; the next job rebuilds it" | tee -a "$VISTA_OUT/job.log"
  fi
fi
"$VENV/bin/python" -c 'import autogen, mcp, matplotlib, scipy, imageio_ffmpeg' 2>>"$VISTA_OUT/job.log" \
  || fail 3 "the venv does not import autogen/mcp/matplotlib/scipy/imageio_ffmpeg; delete $TAR to rebuild"
# The app's own offline checks (routing, RUN_SIMULATION parser, robust scoring): seconds, no LLM.
"$VENV/bin/python" "$A/tests/test_offline.py" > "$VISTA_OUT/offline_tests.log" 2>&1 \
  || fail 3 "the app's offline tests failed; see offline_tests.log"
echo "$(date +%T) offline tests: $(tail -1 "$VISTA_OUT/offline_tests.log")" | tee -a "$VISTA_OUT/job.log"

wait_up() {  # $1 base URL, $2 minutes
  for i in $(seq 1 $(( $2 * 2 ))); do
    curl -sf -m 10 -o /dev/null "$1/models" && return 0
    [ -n "${VPID:-}" ] && ! kill -0 "$VPID" 2>/dev/null && return 1
    sleep 30
  done
  return 1
}

if [ "$BACKEND" = self ]; then
  # vLLM on GPU 0 of this node. start_vllm_lux.sh numbers GPUs as the node does, so the node must be ours.
  [ "${SLURM_GPUS_ON_NODE:-0}" -ge 8 ] || fail 2 "BACKEND=self needs the whole node (exclusive_node_use true, gpus_per_node 8); this job has ${SLURM_GPUS_ON_NODE:-0} GPU(s)"
  : "${ICF_GPTOSS_MODEL:?ICF_GPTOSS_MODEL not set (cluster_defaults.json or env.lux.sh)}"
  : "${ICF_VLLM_IMAGE_TAR:?ICF_VLLM_IMAGE_TAR not set (cluster_defaults.json or env.lux.sh)}"
  export ICF_VLLM_MODEL_PATH=$ICF_GPTOSS_MODEL ICF_VLLM_MODEL_NAME=gptoss-20b-hedp
  export ICF_VLLM_IMAGE_REF=${ICF_VLLM_IMAGE_REF:-localhost/inference_image_gfx950:v8mi250_20261011}
  echo "$(date +%T) starting vLLM ($ICF_VLLM_MODEL_NAME) on GPU 0; log $VISTA_OUT/vllm.log" | tee -a "$VISTA_OUT/job.log"
  bash "$A/start_vllm_lux.sh" > "$VISTA_OUT/vllm.log" 2>&1 &
  VPID=$!
  URL=http://localhost:8000/v1
  wait_up "$URL" 30 || { tail -30 "$VISTA_OUT/vllm.log" >&2; fail 4 "vLLM did not come up on this node; see vllm.log"; }
else
  if [ -z "$ENDPOINT" ]; then
    if [ -n "$SERVE_JOB" ]; then
      nodes=$(squeue -h -j "$SERVE_JOB" -t RUNNING -o '%N' 2>/dev/null)
      [ -n "$nodes" ] || fail 2 "serve job $SERVE_JOB is not running"
    else
      # The newest running vllm-skill Inkling server of this user (VISTA names it after the job, inkling-serve; by
      # hand inkling.serve). vllm-serve is the job's name in the standalone vllm-on-lux skill.
      nodes=$(squeue -h -u "$USER" -t RUNNING -o '%i|%j|%N' | awk -F'|' '$2 ~ /(inkling|vllm)-serve$|^inkling\.serve$/' | sort -n | tail -1 | cut -d'|' -f3)
      [ -n "$nodes" ] || fail 2 "no ENDPOINT or SERVE_JOB given and no running inkling-serve job of yours"
    fi
    ENDPOINT=http://$(scontrol show hostnames "$nodes" | head -1):8000/v1
  fi
  URL=${ENDPOINT%/}
  echo "$(date +%T) waiting for $URL (an inkling-serve job is up 11-14 min after it starts)" | tee -a "$VISTA_OUT/job.log"
  wait_up "$URL" 40 || fail 4 "nothing answers at $URL/models after 40 min"
fi
echo "$(date +%T) endpoint up: $URL" | tee -a "$VISTA_OUT/job.log"

ARGS=(--out "$VISTA_OUT")
[ -n "$SWEEP" ] && ARGS+=(--sweep "$SWEEP") || ARGS+=(--energy "${ENERGY:-3}")
[ "$MOVIE" = 1 ] && ARGS+=(--movie)
[ "$DEBRIEF" = 1 ] && ARGS+=(--debrief)
export ICF_BASE_URL=$URL ICF_APP_DIR=$A PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR=$VISTA_OUT/.mpl
# ffmpeg for the movie: the static binary of the imageio-ffmpeg wheel in the venv.
FF=$("$VENV/bin/python" -c 'import imageio_ffmpeg as i; print(i.get_ffmpeg_exe())' 2>/dev/null) && mkdir -p "$A/bin" && ln -sf "$FF" "$A/bin/ffmpeg"
export PATH=$A/bin:$VENV/bin:$PATH
cd "$A"
"$VENV/bin/python" "$RUN_DIR_Lux/run_campaign.py" "${ARGS[@]}" 2>&1 | tee -a "$VISTA_OUT/job.log"
rc=${PIPESTATUS[0]}
echo "=== exit $rc" | tee -a "$VISTA_OUT/job.log"
exit "$rc"
