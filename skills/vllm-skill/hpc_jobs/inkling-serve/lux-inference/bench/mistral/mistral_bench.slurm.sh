#!/bin/bash
#SBATCH --job-name=mistral.bench
##SBATCH --partition=<your MI355X partition>
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=mistral.bench.%j.out
#
# Serve Mistral-Large-Instruct-2411 (or a fine-tune of it) on one node with the
# package's tuned defaults, run checks and benchmarks against it, and stop.
# This is the job that produced docs/LUX_MISTRAL_LARGE.md. Run it from the
# package root:
#
#   ZENITH_MODELPATH=<dir holding the model dir> \
#   ZENITH_IMAGE_TAR=<path to inference_image_gfx950_v8mi250_20261011.tar> \
#   ZENITH_DATASETSPATH=<datasets dir; science/ in it from bench/science/fetch_science_data.py> \
#   ML_STEPS="correctness tool gsm8k science" \
#     sbatch -A <account> bench/mistral/mistral_bench.slurm.sh
#
# Knobs (environment):
#   ML_STEPS     any of: correctness tool gsm8k needle science 128k 8k
#                (default "correctness tool gsm8k science")
#   ML_MODEL     model directory name (default Mistral-Large-Instruct-2411). Any
#                other name is served as a fine-tune of it (ZENITH_TUNED_AS).
#   ML_TAG       label for the run (default: the model name)
#   ML_ARGS      replaces the default vLLM flags (ZENITH_VLLM_EXTRA_ARGS)
#   ML_NEEDLE    long_context_check.py args (default: its 15-cell grid to 1,040k)
#   ML_SCIENCE   extra science_suite.py args, e.g. "--n 20" for a smoke test
#   ML_CONC      8k concurrency list (default 1 2 4 ... 1024)
#   ML_CONC128   128k concurrency list (default 1 2 4 8)
#   ML_OUT       results dir (default bench/mistral/runs/<tag>.<jobid>)
# Any ZENITH_* setting passes through to the server.
#
# science.json holds every GPQA question and reply: the run dir is created
# private (umask 077); keep it off the web (bench/science/README.md).
set -u
# LUX_<KNOB> is the Lux spelling of every ZENITH_<KNOB> and wins.
for _v in $(compgen -e | grep '^LUX_'); do export "ZENITH_${_v#LUX_}=${!_v}"; done
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f docker/submit_lux.slurm.sh ] || { echo "run from the package root"; exit 1; }
P=$(pwd)
B=$P/bench/inkling
M=$P/bench/mistral
: "${ZENITH_MODELPATH:?set ZENITH_MODELPATH to the directory holding the model}"
: "${ZENITH_IMAGE_TAR:?set ZENITH_IMAGE_TAR to inference_image_gfx950_v8mi250_20261011.tar}"
export ZENITH_MODELPATH ZENITH_IMAGE_TAR
export ZENITH_DATASETSPATH="${ZENITH_DATASETSPATH:-${HOME}/datasets}"
export ZENITH_MODEL="${ML_MODEL:-Mistral-Large-Instruct-2411}"
[ "${ZENITH_MODEL}" = "Mistral-Large-Instruct-2411" ] \
    || export ZENITH_TUNED_AS="${ZENITH_TUNED_AS:-Mistral-Large-Instruct-2411}"
[ -n "${ML_ARGS:-}" ] && export ZENITH_VLLM_EXTRA_ARGS="${ML_ARGS}"
TAG=${ML_TAG:-${ZENITH_MODEL}}
STEPS=" ${ML_STEPS:-correctness tool gsm8k science} "
OUT=${ML_OUT:-$M/runs/${TAG}.${SLURM_JOB_ID}}
( umask 077; mkdir -p "$OUT" )
# Lux: per-job podman storage on the node's burst buffer, off NFS
export ZENITH_GRAPHROOT="${ZENITH_GRAPHROOT:-/mnt/bb/${SLURM_JOB_ID}/containers/storage}"
export ZENITH_TMPDIR="${ZENITH_TMPDIR:-/mnt/bb/${SLURM_JOB_ID}/containers/tmp}"
export XDG_RUNTIME_DIR=/tmp/zenith-xdg-${SLURM_JOB_ID}
export XDG_CONFIG_HOME=/tmp/zenith-xdg-${SLURM_JOB_ID}/config
export ZENITH_STARTUP_TRIES="${ZENITH_STARTUP_TRIES:-120}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
BASE=http://$HEAD:8000
{ echo "job=$SLURM_JOB_ID tag=$TAG node=$SLURM_JOB_NODELIST steps=$STEPS $(date)"; env | grep '^ZENITH_' | sort; } \
    | tee "$OUT/config.txt"
srun --ntasks-per-node=1 --overlap bash -c \
    "mkdir -p $ZENITH_GRAPHROOT $ZENITH_TMPDIR $XDG_RUNTIME_DIR && chmod 700 $XDG_RUNTIME_DIR"
cd "$P/docker"
SLURM_SUBMIT_DIR="$P/docker" bash ./submit_lux.slurm.sh > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
    kill "$SERVER_PID" 2>/dev/null; sleep 30
    ( source ./lux_podman_env.sh >/dev/null 2>&1 \
        && srun --ntasks-per-node=1 --overlap bash -c "${PODMAN} rm -af -t 10" >/dev/null 2>&1 )
    srun --ntasks-per-node=1 --overlap bash -c \
        "podman unshare rm -rf /mnt/bb/${SLURM_JOB_ID}/containers; rm -rf /tmp/zenith-xdg-${SLURM_JOB_ID}" 2>/dev/null
}
trap cleanup EXIT
t0=$(date +%s)
until curl -sf -m 5 "$BASE/v1/models" >/dev/null; do
    kill -0 $SERVER_PID 2>/dev/null || { echo "=== server exited before serving"; tail -80 "$OUT/server.log"; exit 1; }
    sleep 20
done
echo "=== UP at $BASE/v1 after $(( ($(date +%s) - t0) / 60 )) min ($(date))"
grep -hE 'zenith: tuned defaults|GPU KV cache size|Using .* backend' "$OUT/server.log" 2>/dev/null \
    | cut -c1-400 | sort -u | tee "$OUT/startup.txt"

step() { echo "=== $1 start $(date +%T)"; }
alive() { curl -sf -m 10 "$BASE/v1/models" >/dev/null || { echo "=== server dead before $1"; return 1; }; }
has() { case "$STEPS" in *" $1 "*) return 0 ;; esac; return 1; }

has correctness && alive correctness && { step correctness
python3 "$B/correctness_probe.py" --host "$HEAD" --port 8000 --trials 1 --out "$OUT/correctness.json" \
    > "$OUT/correctness.txt" 2>&1
tail -2 "$OUT/correctness.txt"; }

has tool && alive tool && { step tool
python3 "$M/toolcheck.py" "$BASE" > "$OUT/tool.txt" 2>&1; cat "$OUT/tool.txt"; }

has gsm8k && alive gsm8k && { step gsm8k
python3 "$B/gsm8k_chat.py" --host "$HEAD" --port 8000 --n 250 --effort '' --out "$OUT/gsm8k.json" \
    > "$OUT/gsm8k.txt" 2>&1
tail -1 "$OUT/gsm8k.txt"; }

has needle && alive needle && { step needle
python3 "$B/long_context_check.py" --host "$HEAD" --port 8000 ${ML_NEEDLE:-} --out "$OUT/long_context.json" \
    > "$OUT/long_context.txt" 2>&1
cat "$OUT/long_context.txt"; }

has science && alive science && { step science
( umask 077; python3 "$P/bench/science/science_suite.py" --host "$HEAD" --port 8000 \
    --data-dir "$ZENITH_DATASETSPATH/science" --concurrency 128 ${ML_SCIENCE:-} --out "$OUT/science.json" \
    > "$OUT/science.txt" 2>&1 )
sed -n '/^suite score/,$p' "$OUT/science.txt"; }

has 128k && for c in ${ML_CONC128:-1 2 4 8}; do
    alive "128k c=$c" || break
    step "128k c=$c"
    r=$(( (4 + c - 1) / c )); [ $r -lt 2 ] && r=2
    python3 "$B/bench_throughput.py" --host "$HEAD" --port 8000 --concurrency $c --rounds $r \
        --prompt-tokens 128000 --max-tokens 1024 --temperature 0.7 --api chat \
        --label "${TAG}_128k_c$c" --json-out "$OUT/bench_128k.jsonl" > "$OUT/bench_128k_c$c.txt" 2>&1
done

has 8k && for c in ${ML_CONC:-1 2 4 8 16 32 64 128 256 512 1024}; do
    alive "8k c=$c" || break
    step "8k c=$c"
    r=$(( (16 + c - 1) / c ))
    python3 "$B/bench_throughput.py" --host "$HEAD" --port 8000 --concurrency $c --rounds $r \
        --prompt-tokens 8000 --max-tokens 1024 --temperature 0.7 --api chat \
        --label "${TAG}_8k_c$c" --json-out "$OUT/bench_8k.jsonl" > "$OUT/bench_8k_c$c.txt" 2>&1
done
has 8k && python3 "$M/mlscore.py" "$TAG=$OUT/bench_8k.jsonl"
has 128k && python3 "$M/mlscore.py" "$TAG=$OUT/bench_128k.jsonl"
echo "=== DONE $(date); results in $OUT"
