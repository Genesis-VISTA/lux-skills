#!/bin/bash
#SBATCH --job-name=muse.serve
##SBATCH --partition=<your MI355X partition>
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=muse.serve.%j.out
#SBATCH --error=muse.serve.%j.out
#
# The Lux serve wrapper of the inkling-serve engine, on the Zenith v0.1.9 engine
# (Muse Spark Open). It wraps submit_lux.slurm.sh, which holds every serving
# default (Muse's are in lux_tuned_defaults.sh, keyed on ZENITH_MODEL); this
# script only adds the Lux site setup, a check of the fabric vLLM actually got
# (multi-node only), and a correctness probe before announcing the endpoint.
#
#   ZENITH_MODEL=Muse-Spark-Open-816B-A42B-Quark-MXFP4 \
#   ZENITH_MODELPATH=<dir holding the model and Muse-Spark-Open-DSpark/> \
#   ZENITH_IMAGE_TAR=<path to inference_image_gfx950_musemain2.tar> \
#     sbatch -A <account> -N 1 docker/serve_lux.slurm.sh
#
# When the log says "=== SERVING at http://<head>:8000/v1", point any
# OpenAI-compatible client at that URL with the model's directory name as the
# model. Reasoning comes back in each message's `reasoning` field, tool calls in
# `tool_calls` (the launcher picks the muse parsers for Muse-Spark*). The
# server runs until the job's time limit or `scancel`.
#
# LUX_BENCH="science mmlu gsm8k" (any subset) turns the job into a benchmark
# run: once serving, it runs those steps against the endpoint, writes the
# results next to server.log and exits. LUX_SCIENCE_ARGS adds arguments to
# science_suite.py (e.g. "--n 20"), LUX_GSM8K_N sets the gsm8k question count
# (default 250, the zenith release check).
set -u
# Every engine knob is named ZENITH_<KNOB> (the image's baked-in patches read
# those names); LUX_<KNOB> is the Lux spelling of the same knob and wins.
for _v in $(compgen -e | grep '^LUX_'); do export "ZENITH_${_v#LUX_}=${!_v}"; done
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f ./submit_lux.slurm.sh ] || cd docker
[ -f ./submit_lux.slurm.sh ] || { echo "run from the package root or docker/"; exit 1; }
DOCKER=$(pwd)
BENCH=$(cd ../bench/inkling && pwd)
OUT="${ZENITH_SERVE_OUT:-${ZENITH_RUNS_DIR:-${DOCKER}}/serve.${SLURM_JOB_ID}}"
mkdir -p "${OUT}"

: "${ZENITH_MODELPATH:?set ZENITH_MODELPATH to the directory holding the model}"
: "${ZENITH_IMAGE_TAR:?set ZENITH_IMAGE_TAR to inference_image_gfx950_musemain2.tar}"
export ZENITH_MODELPATH ZENITH_IMAGE_TAR
# Lux: per-job podman storage on the node's burst buffer, off NFS (podman
# hangs on an NFS lock otherwise).
export ZENITH_GRAPHROOT="${ZENITH_GRAPHROOT:-/mnt/bb/${SLURM_JOB_ID}/containers/storage}"
export ZENITH_TMPDIR="${ZENITH_TMPDIR:-/mnt/bb/${SLURM_JOB_ID}/containers/tmp}"
export XDG_RUNTIME_DIR=/tmp/zenith-xdg-${SLURM_JOB_ID}
export XDG_CONFIG_HOME=/tmp/zenith-xdg-${SLURM_JOB_ID}/config
export ZENITH_STARTUP_TRIES="${ZENITH_STARTUP_TRIES:-120}"
# INFO-level RCCL logging in the health check and in vLLM itself, so the
# check below can see which transport every cross-node channel took.
export ZENITH_NCCL_DEBUG="${ZENITH_NCCL_DEBUG:-INFO}"
export ZENITH_VLLM_EXTRA_ENV="${ZENITH_VLLM_EXTRA_ENV:-NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

HEAD=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)
BASE=http://${HEAD}:8000
srun --ntasks-per-node=1 --overlap bash -c \
    "mkdir -p ${ZENITH_GRAPHROOT} ${ZENITH_TMPDIR} ${XDG_RUNTIME_DIR} && chmod 700 ${XDG_RUNTIME_DIR}"

SLURM_SUBMIT_DIR="${DOCKER}" bash ./submit_lux.slurm.sh > "${OUT}/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
    kill "${SERVER_PID}" 2>/dev/null; sleep 20
    ( source ./lux_podman_env.sh >/dev/null 2>&1 \
        && srun --ntasks-per-node=1 --overlap bash -c "${PODMAN} rm -af -t 10" >/dev/null 2>&1 )
    srun --ntasks-per-node=1 --overlap bash -c \
        "podman unshare rm -rf /mnt/bb/${SLURM_JOB_ID}/containers; rm -rf /tmp/zenith-xdg-${SLURM_JOB_ID}" 2>/dev/null
}
trap cleanup EXIT

echo "server log: ${OUT}/server.log"
t0=$(date +%s)
until curl -sf -m 5 "${BASE}/v1/models" >/dev/null; do
    kill -0 "${SERVER_PID}" 2>/dev/null || { echo "=== server exited before serving"; tail -60 "${OUT}/server.log"; exit 1; }
    sleep 20
done
echo "=== UP at ${BASE}/v1 after $(( ($(date +%s) - t0) / 60 )) min"

# Fabric check. run_container.sh already pins RCCL to the RCCL-ANP plugin and
# the health check fails the launch below a bandwidth floor; this reads what
# vLLM's own communicators logged. Every cross-node channel has to be
# "via NET/RCCL-ANP/<n>/GDRDMA": NET/Socket, NET/IB or a missing GDRDMA
# stops the service.
if [ "${SLURM_JOB_NUM_NODES}" -gt 1 ]; then
    LOGS=("${OUT}/server.log" "${ZENITH_STATE_DIR:-${DOCKER}/.zenith_state}"/"${SLURM_JOB_ID}"/*.log)
    NETRE='via NET/[A-Za-z_-]+/[0-9]+(/[A-Za-z]+)?'
    grep -hoE "${NETRE}" "${LOGS[@]}" 2>/dev/null | sort | uniq -c | tee "${OUT}/net_transports.txt"
    grep -hoE 'TP all-reduce \(.*|stage [0-9]+ -> [0-9]+: .*' "${LOGS[@]}" 2>/dev/null | sort -u \
        | tee -a "${OUT}/net_transports.txt"
    NET_TOTAL=$(grep -hoE "${NETRE}" "${LOGS[@]}" 2>/dev/null | wc -l)
    NET_BAD=$(grep -hoE "${NETRE}" "${LOGS[@]}" 2>/dev/null | grep -vcE '^via NET/RCCL-ANP/[0-9]+/GDRDMA$')
    if [ "${NET_TOTAL}" -eq 0 ] || [ "${NET_BAD}" -ne 0 ]; then
        echo "=== FABRIC CHECK FAILED: ${NET_TOTAL} cross-node channels, ${NET_BAD} not RCCL-ANP/GDRDMA; stopping"
        exit 1
    fi
    echo "=== FABRIC OK: ${NET_TOTAL} cross-node channels, all RCCL-ANP with GPUDirect RDMA"
fi
grep -hE 'GPU KV cache size|Maximum concurrency' "${OUT}/server.log" 2>/dev/null | sort -u

python3 "${BENCH}/correctness_probe.py" --host "${HEAD}" --port 8000 --trials 1 \
    --out "${OUT}/correctness.json" > "${OUT}/correctness.txt" 2>&1
tail -2 "${OUT}/correctness.txt"
echo "=== SERVING at ${BASE}/v1 (model ${ZENITH_MODEL:-Inkling-MXFP4}; $(date))"
[ -z "${LUX_BENCH:-}" ] && { wait "${SERVER_PID}"; exit; }

# Benchmark mode: run the steps, then exit (the trap stops the server).
# science.json holds every GPQA question and reply: keep it private.
SCI=$(cd ../bench/science && pwd)
SCI_DATA="${LUX_SCIENCE_DATA:-${ZENITH_DATASETSPATH:-${HOME}/datasets}/science}"
for step in ${LUX_BENCH}; do
    t1=$(date +%s)
    echo "=== BENCH ${step} start $(date)"
    case "${step}" in
        science) ( umask 077; python3 "${SCI}/science_suite.py" --host "${HEAD}" --port 8000 \
                     --data-dir "${SCI_DATA}" --concurrency 256 --max-tokens 16384 ${LUX_SCIENCE_ARGS:-} \
                     --out "${OUT}/science.json" > "${OUT}/science.txt" 2>&1 )
                 sed -n '/^suite score/,$p' "${OUT}/science.txt" ;;
        mmlu)    python3 "${BENCH}/mmlu_pro_quick.py" --host "${HEAD}" --port 8000 \
                     --out "${OUT}/mmlu_pro.json" > "${OUT}/mmlu_pro.txt" 2>&1
                 grep -E '^MMLU-Pro subset' "${OUT}/mmlu_pro.txt" ;;
        gsm8k)   # zenith v0.1.9's release check (bench/muse/muse_ladder_spec.sbatch): 250 questions, effort low
                 python3 "${BENCH}/gsm8k_chat.py" --host "${HEAD}" --port 8000 --n "${LUX_GSM8K_N:-250}" \
                     --effort low --max-tokens 4096 --out "${OUT}/gsm8k.json" > "${OUT}/gsm8k.txt" 2>&1
                 tail -1 "${OUT}/gsm8k.txt" ;;
        *)       echo "unknown LUX_BENCH step '${step}' (science, mmlu, gsm8k)" ;;
    esac
    echo "=== BENCH ${step} done in $(( ($(date +%s) - t1) / 60 )) min"
done
echo "=== BENCH finished $(date)"
