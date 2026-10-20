#!/bin/bash
#SBATCH --job-name=lux.inference
# set --partition (and --gres type below) for your site
##SBATCH --partition=<your MI355X partition>
#SBATCH --nodes=2
#SBATCH --exclusive
# Exclude any node that enumerates fewer than 8 GPUs: the launcher derives
# TP from the GPUs it can see, so landing on one silently gives TP=7 on that
# node and a mismatched PP group. (Our cluster had one such node.)
##SBATCH --exclude=<node with a missing GPU>
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --gres=gpu:amd_instinct_mi355_oam:8
#SBATCH --time=04:00:00
#SBATCH --output=lux.%j.out
#SBATCH --error=lux.%j.out
#
# Slurm submission for OLCF Lux (8 x MI355X per node, Pollara RoCE fabric).
# Derived from the Zenith v8 inference package (zenith_inference_v8_lux, 2026-10-12);
# the knobs keep their ZENITH_* names because the image's baked-in patches read
# them. The skill's env.lux.sh maps LUX_<KNOB> onto ZENITH_<KNOB>.
#
# The defaults here are the Lux hackathon tutorial configuration:
# **Inkling-MXFP4 on 2 nodes, TP8 x PP2, with the model's full 1,048,576-token
# context**. PP2 halves the weights per GPU, leaving most of HBM for KV. The
# 2-node serving defaults are in lux_tuned_defaults.sh (KV 200 GB/GPU, MTP
# off -- ZENITH_MTP=1 turns it on with the settings it needs -- 32k-token
# batches). Cross-node RCCL runs over the RCCL-ANP plugin in
# patches/anp with GPUDirect RDMA, and a socket fallback fails the launch; see
# patches/anp/README.md. Reasoning and tool-call parsers are on for the chat
# API. See docs/LUX_HACKATHON.md.
#
# --nodes=1 gives the throughput-tuned single-node configuration instead (TP8
# over all 8 GPUs, MTP-3, KV 128 GB, CUDA graphs to 2048, kernels in
# patches/hl_overlay; 100k context). That is the highest total token rate one
# node gives on this model. Everything else is a deviation from these two:
#
#   sbatch submit_lux.slurm.sh                          # Inkling-MXFP4, 2 nodes, 1M context
#   sbatch --nodes=1 submit_lux.slurm.sh                # Inkling-MXFP4, 1 node, tuned
#   ZENITH_MODEL=Inkling-bf16 sbatch --nodes=2 submit_lux.slurm.sh
#   ZENITH_MODEL=gptoss-20b-hedp.03302026 sbatch --nodes=1 submit_lux.slurm.sh
#
# Inkling bf16 on 2 nodes, TP8 PP2, was the reference configuration before:
# the full-precision model at its arithmetic minimum node count (1 node is
# infeasible at utilization 1.057), measured at 364.6 decode tok/s at
# concurrency 32 and 3585-class throughput at saturation; see RESULTS.md.
#
# Inkling-bf16 on --nodes=1 will NOT run: kv_cache_sizing.py reports
# INFEASIBLE and the launcher exits before starting vLLM rather than failing as
# an OOM 40 minutes into the load.
#
set -u

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# path containing all of the LLMs here on the host; container location is
# always /app/models. On Lux the models live in the user's NFS home.
MODELPATH="${ZENITH_MODELPATH:-${HOME}/hfmodels}"
DATASETSPATH="${ZENITH_DATASETSPATH:-${HOME}/datasets}"
mkdir -p "${DATASETSPATH}"

# model + serving configuration, consumed by lux_container_launcher.sh
export ZENITH_MODEL="${ZENITH_MODEL:-Inkling-MXFP4}"
# Defaults tuned for one model per node count (Inkling-MXFP4 on 1 and 2
# nodes); they only fill in what is unset, so they go before the generic ones
# below. ZENITH_TUNED_DEFAULTS=0 skips them.
source ./lux_tuned_defaults.sh "${SLURM_JOB_NUM_NODES:-1}"
export ZENITH_MAX_MODEL_LEN="${ZENITH_MAX_MODEL_LEN:-100000}"

# bf16 is a 1905 GB checkpoint and takes ~43 min to load cold, so the launcher's
# default polling window (30 x 30 s = 15 min) gives up while vLLM is still
# healthy and still reading shards. 90 tries = 45 min. The #SBATCH --time above
# then has to cover that load plus the benchmarking, and **a running job's time
# limit cannot be extended by an ordinary user on Lux** -- a too-short --time
# means resubmitting and re-paying the 43 minutes. Override downward for MXFP4
# (a quarter of the checkpoint) if you want the shorter window back.
export ZENITH_STARTUP_TRIES="${ZENITH_STARTUP_TRIES:-90}"
# 64 GB of KV per GPU is ~22% of an MI355X's 288 GiB and is the value all the
# Lux numbers in the upstream RESULTS.md were taken at.
export ZENITH_TARGET_KV_GB="${ZENITH_TARGET_KV_GB:-64}"
# Scheduler batch limits. Empty = keep vLLM's defaults (max_num_seqs 1024,
# max_num_batched_tokens 16384), so every pre-existing result stays
# reproducible. Exported unconditionally-but-empty for the same
# srun --export=ALL reason as ZENITH_TP/PP below: set-but-not-exported here
# means the launcher never sees it and the job silently runs at the default
# while the operator believes otherwise.
export ZENITH_MAX_NUM_BATCHED_TOKENS="${ZENITH_MAX_NUM_BATCHED_TOKENS:-}"
export ZENITH_MAX_NUM_SEQS="${ZENITH_MAX_NUM_SEQS:-}"
# Largest CUDA-graph capture size; empty = vLLM's 512. Same export rule.
export ZENITH_MAX_CUDAGRAPH_CAPTURE_SIZE="${ZENITH_MAX_CUDAGRAPH_CAPTURE_SIZE:-}"

# Parallelism overrides, for sweeping layouts other than the default. Unset, the launcher derives TP=GPUSPERNODE, PP=NNODES, which is the right
# default and what every RESULTS.md number was taken at. These are exported
# unconditionally-but-empty so that a value given on the sbatch command line
# reaches the launcher through srun --export=ALL; the launcher treats empty
# as absent.
export ZENITH_TP="${ZENITH_TP:-}"
export ZENITH_PP="${ZENITH_PP:-}"
# ZENITH_USE_GPUS runs on fewer GPUs than the node has, for sub-node scaling
# (e.g. ZENITH_USE_GPUS=4 ZENITH_TP=4 on one node). The node is still taken
# exclusively -- the question being asked is how performance scales with GPU
# count, and sharing the node would fold another job's bandwidth and power draw
# into the answer. Note TP x PP x DP must equal ZENITH_USE_GPUS x nodes, not
# the physical GPU count, or the launcher refuses to serve.
export ZENITH_USE_GPUS="${ZENITH_USE_GPUS:-}"
# ZENITH_DP widens expert parallelism. EP = DP x TP (PP is not a factor -- see
# the launcher), so EP16 on two nodes is ZENITH_PP=1 ZENITH_DP=2 with TP left
# at 8. Only meaningful together with
# ZENITH_VLLM_EXTRA_ARGS="--enable-expert-parallel".
export ZENITH_DP="${ZENITH_DP:-}"
# ZENITH_MTP=0 turns speculative decoding off. Exported for the same reason as
# TP/PP above -- without it, `ZENITH_MTP=0 sbatch ...` sets the variable in the
# submitting shell only, srun --export=ALL never carries it to the launcher, and
# the job runs *with* MTP while the operator believes it is the baseline. That
# failure is silent and produces a wrong number rather than an error, which is
# the worst kind. Note the launcher's own default is unquoted-1 (ZENITH_MTP:-1),
# so exporting empty here would read as "not 1" and disable MTP everywhere;
# hence the :-1 default rather than the :- used for TP/PP.
export ZENITH_MTP="${ZENITH_MTP:-1}"
# ZENITH_MTP_PP=1 lets MTP run at PP>1, which the launcher otherwise forbids
# because it asserted (async on) or deadlocked (async off) on this vLLM build.
# The assert path is fixed by patches/spec_decode_pp_broadcast_width.py; the
# deadlock path is not, so the launcher refuses this combination when
# --no-async-scheduling is also passed. Same export-or-it-does-not-exist rule
# as ZENITH_MTP above -- and here the silent-failure mode is the benign
# direction (MTP stays suppressed, the cell is just an MTP-off repeat), which
# is exactly why it needs stating: the run would look like a successful
# MTP+PP test while measuring nothing of the kind. Check the launcher banner
# for "MTP+PP ENABLED" before trusting a row from this path.
export ZENITH_MTP_PP="${ZENITH_MTP_PP:-0}"
# ZENITH_NO_SERVE=1 prepares the cluster (containers, Ray, RCCL health check,
# KV sizing) and then holds without starting a server, so serve_disagg.sh can
# bring up a prefill instance and a decode instance with different roles. Same
# export-or-it-does-not-exist rule as the knobs above.
export ZENITH_NO_SERVE="${ZENITH_NO_SERVE:-0}"
# DEBUG surfaces vLLM's logger.debug_once diagnostics (e.g. why an optional
# fast path did not engage), which are invisible at INFO. Costs a noisier log,
# nothing else.
export ZENITH_VLLM_LOG_LEVEL="${ZENITH_VLLM_LOG_LEVEL:-INFO}"
# Chat-API reasoning / tool-call parsers: "auto" (inkling for Inkling models),
# "none", or a vLLM parser name. See the launcher.
export ZENITH_REASONING_PARSER="${ZENITH_REASONING_PARSER:-auto}"
export ZENITH_TOOL_CALL_PARSER="${ZENITH_TOOL_CALL_PARSER:-auto}"
export ZENITH_VLLM_EXTRA_ARGS="${ZENITH_VLLM_EXTRA_ARGS:-}"
# Extra NAME=VALUE environment for the served vLLM process, space-separated.
# Same export-or-it-does-not-exist rule as every knob above: without the
# export, `ZENITH_VLLM_EXTRA_ENV=... sbatch ...` sets it in the submitting
# shell only and srun --export=ALL never carries it to the launcher, so the job
# runs without it while the operator believes otherwise.
#
# Current use: ZENITH_VLLM_EXTRA_ENV="VLLM_USE_V2_MODEL_RUNNER=1" to get MTP
# working at PP>1. The V1 model runner does not implement async scheduling with
# PP (config/vllm.py:588 caps concurrent batches at pp_size and says so), so
# drafts never reach the non-last PP rank and acceptance pins at exactly 0.
export ZENITH_VLLM_EXTRA_ENV="${ZENITH_VLLM_EXTRA_ENV:-}"
# Arms VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1, which vLLM defaults to
# False -- see the launcher comment. Off by default because the flag is global
# across every MoE model here and the measured configurations run with it off.
export ZENITH_AITER_FUSE_SHARED_EXPERTS="${ZENITH_AITER_FUSE_SHARED_EXPERTS:-}"

# Image: run_container.sh runs docker-archive:<scriptdir>/inference_image.tar
# directly out of the tarball that build_image.sh writes, unless
# ZENITH_IMAGE_TAR names another tarball (a candidate build from
# `VLLM_HASH=<sha> ./build_image.sh`) or ZENITH_IMAGE_REF names any podman
# image reference (e.g. localhost/inference_image_gfx950:latest, once loaded).
# ZENITH_IMAGE_TAG=<tag> (the tag given to build_image.sh) selects
# inference_image_<tag>.tar instead; ZENITH_IMAGE_TAR / _REF still win.
export ZENITH_IMAGE_TAG="${ZENITH_IMAGE_TAG:-}"
export ZENITH_IMAGE_TAR="${ZENITH_IMAGE_TAR:-}"
export ZENITH_IMAGE_REF="${ZENITH_IMAGE_REF:-}"

# there is no module system on this cluster -- ROCm is installed in the image.
# (on Tuolumne the flux script does: module load amd/rocm/PrgEnv + enable-podman)

# rootless podman needs XDG_RUNTIME_DIR and a writable graph root; this script
# sets them up identically on every node in the allocation
source ./lux_podman_env.sh

# resource info from slurm rather than `flux resource info`
NNODES="${SLURM_JOB_NUM_NODES}"
CPUSPERNODE="${SLURM_CPUS_ON_NODE:-$(nproc)}"
GPUSPERNODE=$(echo "${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-0,1,2,3,4,5,6,7}}" | tr ',' '\n' | grep -c .)
export ZENITH_NNODES="${NNODES}"
export ZENITH_GPUS_PER_NODE="${GPUSPERNODE}"
export ZENITH_CPUS_PER_NODE="${CPUSPERNODE}"

# ordered hostlist; the first entry is the ray head
ZENITH_HOSTS=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | tr '\n' ' ')
export ZENITH_HOSTS
export ZENITH_HEADNODE=$(echo "${ZENITH_HOSTS}" | awk '{print $1}')

echo "Booting up Ray containers on ${NNODES} nodes with ${CPUSPERNODE} cores"
echo "  and ${GPUSPERNODE} gpus per node"
echo "  hosts:  ${ZENITH_HOSTS}"
echo "  head:   ${ZENITH_HEADNODE}"
echo "  model:  ${ZENITH_MODEL}"

# when the workload gets heavy, ray spews this annoying message - filter it out
WARNING_STRING="PlacementGroupCleaner"

# one task per node, each running the launcher. --exclusive/--overlap semantics
# differ from flux: we want exactly one launcher per node with all local GPUs
# visible, hence --ntasks=NNODES --ntasks-per-node=1 and no per-task gpu binding
# (the container itself sets HIP_VISIBLE_DEVICES).
stdbuf -oL srun \
    --nodes="${NNODES}" \
    --ntasks="${NNODES}" \
    --ntasks-per-node=1 \
    --cpus-per-task="${CPUSPERNODE}" \
    --gres=gpu:"${GPUSPERNODE}" \
    --export=ALL \
    --kill-on-bad-exit=1 \
    bash ./lux_container_launcher.sh "${MODELPATH}" "${DATASETSPATH}" \
    2>&1 | grep --line-buffered -v "${WARNING_STRING}"
