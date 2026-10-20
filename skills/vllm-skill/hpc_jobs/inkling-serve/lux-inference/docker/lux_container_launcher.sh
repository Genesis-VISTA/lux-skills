#!/bin/bash
#
# automates the launch of the container + ray cluster + vllm inference server
# zenith uses a split-fabric approach where the control plane (Ray/GCS/ethernet)
# is separate from the data plane (RCCL over slingshot or RoCE)
#
# scheduler-agnostic: resource/hostlist discovery comes from flux on the
# MI300A/Tuolumne system and from slurm on the MI355X/Lux system.
#

# arg checking
if [ $# -lt 2 ]; then
    echo "Usage: $0 <directory holding the models> <directory holding datasets>"
    exit 1
fi
MODELPATH="$1"          # directory holding our LLMs
DATASETSPATH="$2"	# directory holding our datasets

# arg checking on model path
if [ ! -d "${MODELPATH}" ]; then
    echo "Error: Model path directory ${MODELPATH} does not exist"
    exit 1
fi

SCRIPTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPTDIR}"
source ./lux_podman_env.sh

MYNODE=$(/usr/bin/hostname)
# per-node container name: "node", or <ZENITH_CONTAINER_PREFIX>-<hostname suffix> when the
# prefix is set (lets two launchers share a node; must match run_container.sh)
CNAME="${ZENITH_CONTAINER_PREFIX:+${ZENITH_CONTAINER_PREFIX}-${MYNODE##*-}}"; CNAME="${CNAME:-node}"

# ---------------------------------------------------------------------------
# how many AMD GPUs does this node have?
#   4 on MI300A, 8 on MI355X, 8 on a 4x MI250X node
# ---------------------------------------------------------------------------
# An MI250X is two GCDs in one package and every layer of the stack -- KFD,
# rocm-smi, HIP, torch, vLLM -- enumerates them as two separate devices, so a
# node advertised as "4x MI250X" counts as 8 GPUs here. That is the number we
# want: TP is set from it, and TP=8 across 8 GCDs is exactly right.
#
# NB: do NOT count /sys/class/drm/card*/device/drm -- that glob matches every
# DRM node including the render/connector children and the BMC's ASPEED VGA,
# which returns 65 on a Lux node. rocm-smi and the KFD topology (filtered on
# simd_count, which is 0 for CPU-only KFD nodes) are the reliable sources.
count_local_gpus () {
    local n=0
    if command -v rocm-smi >/dev/null 2>&1; then
        n=$(rocm-smi --showid --csv 2>/dev/null | grep -c '^card')
    fi
    if [ -z "${n}" ] || [ "${n}" -eq 0 ]; then
        n=$(for d in /sys/class/kfd/kfd/topology/nodes/*/; do
                [ "$(awk '/^simd_count/{print $2}' "${d}/properties" 2>/dev/null)" != "0" ] \
                    && echo x
            done | wc -l)
    fi
    [ -z "${n}" ] && n=0
    echo "${n}"
}

# ---------------------------------------------------------------------------
# how much VRAM does one GPU on this node have?
#   128 GiB MI300A, 288 GiB MI355X, 64 GiB per MI250X GCD
# ---------------------------------------------------------------------------
# This has to be detected, not assumed. kv_cache_sizing.py solves
#
#     util = (weights + overhead + target_kv) / vram_per_gpu
#
# so a wrong vram_per_gpu scales the utilization vLLM is handed by exactly the
# ratio of the two cards. Running the MI355X constant (288 GiB) on an MI300A
# asks for 0.40 instead of 0.90 -- a *budget* of 51 GiB on a 128 GiB APU, which
# is less than the weights alone, so vLLM OOMs during model load. That looks
# nothing like a sizing bug from the log, which is why it is detected here and
# asserted below rather than left to a default.
#
# KFD reports it exactly: each GPU topology node has mem_banks/N/properties with
# heap_type 1 (VRAM, or "coarse-grained system memory" on the MI300A APU, which
# is the same physical HBM) and size_in_bytes. rocm-smi's --showmeminfo vram is
# the fallback; note it reports *usable* VRAM, which is a hair under the
# physical size, so prefer KFD.
detect_vram_bytes () {
    local best=0 sz d b ht
    for d in /sys/class/kfd/kfd/topology/nodes/*/; do
        [ "$(awk '/^simd_count/{print $2}' "${d}/properties" 2>/dev/null)" = "0" ] \
            && continue
        for b in "${d}"mem_banks/*/; do
            ht=$(awk '/^heap_type/{print $2}' "${b}/properties" 2>/dev/null)
            # heap_type 1 = HEAP_TYPE_FB_PUBLIC (device VRAM)
            [ "${ht}" = "1" ] || continue
            sz=$(awk '/^size_in_bytes/{print $2}' "${b}/properties" 2>/dev/null)
            [ -n "${sz}" ] && [ "${sz}" -gt "${best}" ] 2>/dev/null && best=${sz}
        done
    done
    if [ "${best}" -eq 0 ] && command -v rocm-smi >/dev/null 2>&1; then
        best=$(rocm-smi --showmeminfo vram --csv 2>/dev/null \
               | awk -F, 'NR>1 && $2 ~ /^[0-9]+$/ {print $2; exit}')
    fi
    [ -z "${best}" ] && best=0
    echo "${best}"
}

# KFD gfx_target_version of the first GPU node: 90010 gfx90a (MI250X), 90402
# gfx942 (MI300A), 90500 gfx950 (MI355X); 0 if it cannot be read
detect_gfx_target_version () {
    local d v
    for d in /sys/class/kfd/kfd/topology/nodes/*/; do
        v=$(awk '/^gfx_target_version/{print $2}' "${d}/properties" 2>/dev/null)
        [ -n "${v}" ] && [ "${v}" != "0" ] && { echo "${v}"; return; }
    done
    echo 0
}

# ---------------------------------------------------------------------------
# resource discovery -- flux (Tuolumne) or slurm (Lux)
# ---------------------------------------------------------------------------
if [ -n "${ZENITH_HOSTS:-}" ]; then
    # supplied by submit_lux.slurm.sh
    SCHEDULER="slurm"
    MYJOBHOSTS="${ZENITH_HOSTS}"
    NNODES="${ZENITH_NNODES}"
    GPUSPERNODE="${ZENITH_GPUS_PER_NODE}"
    CPUSPERNODE="${ZENITH_CPUS_PER_NODE}"
    HEADNODE="${ZENITH_HEADNODE}"
elif [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    SCHEDULER="slurm"
    MYJOBHOSTS=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | tr '\n' ' ')
    NNODES="${SLURM_JOB_NUM_NODES}"
    CPUSPERNODE="${SLURM_CPUS_ON_NODE:-$(nproc)}"
    if [ -n "${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-}}" ]; then
        GPUSPERNODE=$(echo "${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS}}" | tr ',' '\n' | grep -c .)
    else
        GPUSPERNODE=$(count_local_gpus)
    fi
    HEADNODE=$(echo "${MYJOBHOSTS}" | awk '{print $1}')
elif command -v flux >/dev/null 2>&1 && [ -n "${FLUX_JOB_ID:-}" ]; then
    SCHEDULER="flux"
    RESOURCEINFO=$(flux resource info)
    NNODES=$(echo ${RESOURCEINFO} | awk '{print $1}')
    NCPUS=$(echo ${RESOURCEINFO} | awk '{print $3}')
    NGPUS=$(echo ${RESOURCEINFO} | awk '{print $5}')
    GPUSPERNODE=$((NGPUS / NNODES))
    CPUSPERNODE=$((NCPUS / NNODES))
    MYJOBHOSTS=$(flux job hostpids ${FLUX_JOB_ID} | sed 's/:[^:]*,/ /g' | sed 's/:[^:]*/ /g')
    HEADNODE=$(echo ${MYJOBHOSTS} | awk '{print $1}')
else
    # bare metal / interactive single node
    SCHEDULER="none"
    MYJOBHOSTS="${MYNODE}"
    NNODES=1
    CPUSPERNODE=$(nproc)
    GPUSPERNODE=$(count_local_gpus)
    [ "${GPUSPERNODE}" -eq 0 ] && GPUSPERNODE=8
    HEADNODE="${MYNODE}"
fi

# cross-check the scheduler's view against what the hardware actually reports.
# A mismatch here means the allocation and the node disagree, and every
# downstream sizing decision (TP, KV cache, health-check world size) would be
# wrong -- so fail loudly rather than requesting 65 GPUs from ray.
LOCAL_GPUS=$(count_local_gpus)
if [ "${LOCAL_GPUS}" -gt 0 ] && [ "${GPUSPERNODE}" != "${LOCAL_GPUS}" ]; then
    echo "WARNING: scheduler reports ${GPUSPERNODE} GPUs/node but this node has ${LOCAL_GPUS};"
    echo "WARNING: using the hardware count (${LOCAL_GPUS})."
    GPUSPERNODE=${LOCAL_GPUS}
fi
# ZENITH_USE_GPUS deliberately uses fewer GPUs than the node has, for sub-node
# scaling experiments. It has to be applied *after* the hardware cross-check
# above, which would otherwise see the mismatch it creates and "correct" it
# straight back to the physical count -- the run would then serve on 8 GPUs
# while the operator believed it was measuring 4, which is a wrong number
# rather than an error. run_container.sh applies the matching restriction to
# HIP_VISIBLE_DEVICES; both must agree or vLLM sizes for GPUs it cannot see.
if [ -n "${ZENITH_USE_GPUS:-}" ] && [ "${ZENITH_USE_GPUS}" -ge 1 ] 2>/dev/null \
   && [ "${ZENITH_USE_GPUS}" -lt "${GPUSPERNODE}" ]; then
    echo "NOTE: ZENITH_USE_GPUS=${ZENITH_USE_GPUS} -- using ${ZENITH_USE_GPUS} of"
    echo "NOTE: ${GPUSPERNODE} GPUs per node (sub-node run; node still allocated exclusively)."
    GPUSPERNODE=${ZENITH_USE_GPUS}
fi
# 8 is the ceiling on every machine here: 8 MI355X, 4 MI300A, or 8 MI250X GCDs.
if [ "${GPUSPERNODE}" -lt 1 ] || [ "${GPUSPERNODE}" -gt 8 ]; then
    echo "Error: implausible GPUs-per-node (${GPUSPERNODE}); refusing to continue"
    exit 1
fi

# VRAM per GPU, detected here and passed to kv_cache_sizing.py below. Refuse to
# guess: sizing against the wrong card is silent and lands as an OOM deep in
# model loading (see detect_vram_bytes above).
VRAM_BYTES="${ZENITH_VRAM_BYTES:-$(detect_vram_bytes)}"
if [ "${VRAM_BYTES}" -lt 1000000000 ] 2>/dev/null; then
    echo "Error: could not detect VRAM per GPU on $(hostname) (got '${VRAM_BYTES}')."
    echo "Error: set ZENITH_VRAM_BYTES explicitly:"
    echo "Error:   68719476736   MI250X (64 GiB per GCD)"
    echo "Error:   137438953472  MI300A (128 GiB)"
    echo "Error:   309220868096  MI355X (288 GiB)"
    echo "Error: Refusing to guess: a wrong value here surfaces as an"
    echo "Error: out-of-memory failure during model load."
    exit 1
fi

NUMWORKERS=$(( $(echo ${MYJOBHOSTS} | wc -w) - 1 ))

if [ "${MYNODE}" == "${HEADNODE}" ]; then
    NODE_TYPE="head"
else
    NODE_TYPE="worker"
fi

# echo to stdout if we are the head node
function headecho() {
  if [ ${MYNODE} == ${HEADNODE} ]; then
    stdbuf -oL echo "$@"
  fi
}

# echo to stdout if we are a worker node
function workerecho() {
  if [ ${MYNODE} != ${HEADNODE} ]; then
    stdbuf -oL echo "$@"
  fi
}

# ---------------------------------------------------------------------------
# per-job state directory
# ---------------------------------------------------------------------------
# The head node and the workers rendezvous through files in the working
# directory: the head publishes its addresses, each node writes its container
# log, and the head polls those logs for "Ray runtime started" to decide the
# cluster is up. Keeping those files at fixed paths means consecutive jobs share
# them, and that is a race with two distinct failure modes -- both of which were
# observed:
#
#   * a worker sees the *previous* job's container.headnode_eth_address, skips
#     its wait loop, and starts its container against a stale (or simply
#     already-dead) head address;
#   * the head's cleanup then unlinks that early worker's already-open log. The
#     worker keeps writing happily to a deleted inode, ray comes up fine, and
#     the head polls a path that no longer exists -- so it counts "1 out of 2
#     ray instances are running" until the startup window expires. Nothing in
#     either log says anything is wrong, because nothing is: the cluster is up,
#     the head just cannot see it.
#
# Fixing the cleanup ordering only narrows the window. Namespacing the state by
# job id removes it: a fresh directory per job cannot contain a stale anything,
# and two zenith jobs sharing this tree no longer interfere at all.
ZENITH_JOBID="${ZENITH_RUN_ID:-${SLURM_JOB_ID:-${FLUX_JOB_ID:-$$}}}"  # ZENITH_RUN_ID: one id for nodes from separate allocations
# ZENITH_STATE_DIR: where the per-job state dirs live (default beside this script). Lux jobs set it to
# their output dir, so a read-only source tree works too. It must be on a filesystem every node sees.
STATEROOT="${ZENITH_STATE_DIR:-${SCRIPTDIR}/.zenith_state}"
STATEDIR="${STATEROOT}/${ZENITH_JOBID}"
mkdir -p "${STATEDIR}"
HEAD_ETH_FILE="${STATEDIR}/headnode_eth_address"
HEAD_HSI_FILE="${STATEDIR}/headnode_hsi_address"
MY_LOG="${STATEDIR}/container.${MYNODE}.log"

# prune state from jobs that finished long ago. By age, not by "everything that
# is not me" -- another zenith job may be running in this same tree right now.
if [ "${NODE_TYPE}" == "head" ]; then
  find "${STATEROOT}" -maxdepth 1 -mindepth 1 -type d -mtime +7 \
       -exec rm -rf {} + 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# publish the head node's addresses
# ---------------------------------------------------------------------------
# the ray control plane runs over ethernet; the data plane over the high speed
# fabric. Interface names are NOT hardcoded (they differ per site).
pick_eth () {
    local best="" bestspeed=-1
    for ifc in /sys/class/net/*; do
        n=$(basename "${ifc}")
        case "${n}" in lo|docker*|veth*|br-*|virbr*|hsi*|hsn*) continue ;; esac  # hsi*/hsn*: Slingshot
        [ "$(cat ${ifc}/operstate 2>/dev/null)" = "up" ] || continue
        addr=$(ip -o -4 addr show "${n}" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)
        [ -n "${addr}" ] || continue
        sp=$(cat ${ifc}/speed 2>/dev/null); [ -n "${sp}" ] || sp=0
        case "${addr}" in 10.*|192.168.*|172.1[6-9].*|172.2[0-9].*|172.3[01].*) sp=$((sp + 1000000)) ;; esac
        if [ "${sp}" -gt "${bestspeed}" ]; then bestspeed=${sp}; best=${n}; fi
    done
    echo "${best}"
}
ETH_IFACE="${ZENITH_ETH_IFACE:-$(pick_eth)}"
MY_ETH_ADDRESS=$(ip -o -4 addr show "${ETH_IFACE}" | awk '{print $4}' | cut -d/ -f1 | head -1)
if ip link show hsi0 >/dev/null 2>&1; then
    MY_HSI_ADDRESS=$(ip -o -4 addr show hsi0 | awk '{print $4}' | cut -d/ -f1 | head -1)
else
    MY_HSI_ADDRESS="${MY_ETH_ADDRESS}"
fi

# Publish atomically: a worker polling for the file must never be able to read
# it while it is half-written (or, worse, empty because the shell truncated it
# before echo ran). Write to a temp name in the same directory, then rename.
if [ "${NODE_TYPE}" == "head" ]; then
    echo "${MY_HSI_ADDRESS}" > "${HEAD_HSI_FILE}.tmp" && mv -f "${HEAD_HSI_FILE}.tmp" "${HEAD_HSI_FILE}"
    echo "${MY_ETH_ADDRESS}" > "${HEAD_ETH_FILE}.tmp" && mv -f "${HEAD_ETH_FILE}.tmp" "${HEAD_ETH_FILE}"
fi

# current rev
ZENITH_VERSION=`cat VERSION`
headecho "Launching Zenith version ${ZENITH_VERSION}"
# output ASCII art logo
headecho ''
headecho '      ___           ___           ___                                     ___     '
headecho '     /\__\         /\__\         /\  \                                   /\  \    '
headecho '    /::|  |       /:/ _/_        \:\  \       ___           ___          \:\  \   '
headecho '   /:/:|  |      /:/ /\__\        \:\  \     /\__\         /\__\          \:\  \  '
headecho '  /:/|:|  |__   /:/ /:/ _/_   _____\:\  \   /:/__/        /:/  /      ___ /::\  \ '
headecho ' /:/ |:| /\__\ /:/_/:/ /\__\ /::::::::\__\ /::\  \       /:/__/      /\  /:/\:\__\'
headecho ' \/__|:|/:/  / \:\/:/ /:/  / \:\~~\~~\/__/ \/\:\  \__   /::\  \      \:\/:/  \/__/'
headecho '     |:/:/  /   \::/_/:/  /   \:\  \        ~~\:\/\__\ /:/\:\  \      \::/__/     '
headecho '     |::/  /     \:\/:/  /     \:\  \          \::/  / \/__\:\  \      \:\  \     '
headecho '     |:/  /       \::/  /       \:\__\         /:/  /       \:\__\      \:\__\    '
headecho '     |/__/         \/__/         \/__/         \/__/         \/__/       \/__/    '
headecho ''

# output the job resource info
headecho "scheduler=${SCHEDULER}  ${NNODES} total nodes, headnode is ${HEADNODE}, mynode is ${MYNODE} which is the head node"
workerecho "scheduler=${SCHEDULER}  ${NNODES} total nodes, headnode is ${HEADNODE}, mynode is ${MYNODE} which is 1 of ${NUMWORKERS} workers"

# ensure a race condition doesn't exist whereby the head node hasn't yet written out it's interface IP
if [ ${NODE_TYPE} == "worker" ]; then
  MAX_RETRIES=60
  RETRIES=0
  while [ ! -s "${HEAD_ETH_FILE}" ] && [ $RETRIES -lt $MAX_RETRIES ]; do
    sleep 2
    RETRIES=$((RETRIES + 1))
  done
  if [ ! -s "${HEAD_ETH_FILE}" ]; then
    echo "Error: Timed out waiting for ${HEAD_ETH_FILE}"
    exit 1
  fi
fi

HEADNODE_HSI_ADDRESS=$(cat "${HEAD_HSI_FILE}")
HEADNODE_ETH_ADDRESS=$(cat "${HEAD_ETH_FILE}")
if [ -z "${HEADNODE_ETH_ADDRESS}" ]; then
  echo "Error: head node address file ${HEAD_ETH_FILE} is empty"
  exit 1
fi

echo "starting up container of type ${NODE_TYPE} on ${MYNODE} in the background"

# launch the container on each node (head + workers)
if [ ${NNODES} == 1 ]; then
  state_token="single"
else
  state_token="multi"
fi
# ray containers communicate over ethernet, RCCL over the high speed fabric
/bin/bash run_container.sh ${HEADNODE_ETH_ADDRESS} --${NODE_TYPE} ${MODELPATH} ${DATASETSPATH} ${state_token} > "${MY_LOG}" 2>&1 &
CONTAINER_PID=$!

# ---------------------------------------------------------------------------
# head node: wait for the cluster, run health checks, then serve
# ---------------------------------------------------------------------------
if [ ${MYNODE} == ${HEADNODE} ]; then

  echo "*** Starting up all containers to form the ray virtual cluster spanning the job allocation nodes ***"
  sleep 20

  # wait for ray to start on all nodes
  # On a node that has never run zenith before, podman has to import the 61 GB
  # image out of the docker-archive tarball into node-local storage before ray
  # can start, and under fuse-overlayfs that is far slower than the 15 minutes
  # the old fixed limit allowed. Keep 15 min as the default (a node with the
  # image already in storage starts in well under a minute) but let a first-run
  # multinode job raise it.
  ALL_STARTED=0
  COUNTER=1 ; MAXCOUNTER="${ZENITH_STARTUP_TRIES:-30}"
  while [ ${ALL_STARTED} -eq 0 ]; do

    if [ ${COUNTER} == ${MAXCOUNTER} ]; then
      echo "*** container startup failed, exiting after ${MAXCOUNTER} attempts ***"
      echo "*** tail of local container log: ***"
      tail -40 "${MY_LOG}"
      # a node whose log never appeared at all is the interesting case: it
      # means the launcher there died before run_container.sh, not that the
      # container is slow. Name them so the failure is diagnosable from the
      # head's stdout alone.
      for h in ${MYJOBHOSTS}; do
        [ -e "${STATEDIR}/container.${h}.log" ] || echo "*** no log ever written by ${h} ***"
      done
      exit 1
    fi

    STARTED_SUM=0
    for i in `seq 1 ${NNODES}`
    do
      HOST_ELEMENT=`echo $MYJOBHOSTS | cut -d " " -f $i`
      HOST_FILENAME="${STATEDIR}/container.${HOST_ELEMENT}.log"
      if [ -e ${HOST_FILENAME} ]; then
        if grep -q "Ray runtime started" ${HOST_FILENAME}; then
          HOST_IS_FINISHED=1
        else
          HOST_IS_FINISHED=0
        fi
      else
        HOST_IS_FINISHED=0
      fi
      STARTED_SUM=$((STARTED_SUM + ${HOST_IS_FINISHED}))
    done

    if [ ${STARTED_SUM} -eq ${NNODES} ]; then
      ALL_STARTED=1
      echo "*** All workers have successfully started ray ***"
      sleep 5
    else
      NUMSECS=30
      echo "Runtime check counter ${COUNTER} of ${MAXCOUNTER}: ${STARTED_SUM} out of ${NNODES} ray instances are running"
      sleep $NUMSECS
    fi
    COUNTER=$((COUNTER + 1))
  done

  # confirm ray itself agrees on the node count before we trust it
  headecho "*** ray cluster view: ***"
  ${PODMAN} exec ${CNAME} ray status 2>&1 | head -25

  # ---- parallel layout ----
  # Derived before the health checks because ray_health_rccl.py builds and
  # tests exactly the communicators vLLM will build for this layout.
  #
  # tensor parallel within a node, pipeline parallel across nodes. This keeps
  # every all-reduce on the intra-node XGMI/Infinity Fabric and reduces the
  # inter-node traffic to point-to-point activation handoffs, which is the
  # right split for both Slingshot and RoCE.
  #
  # The derived values are the right default and stay the default. The
  # overrides exist for parallelism sweeps, where the question is precisely
  # whether that default split is the fastest one, so the sweep has to be able
  # to hold TP fixed while varying everything else. Do not set these for
  # ordinary runs.
  TP=${ZENITH_TP:-${GPUSPERNODE}}
  PP=${ZENITH_PP:-${NNODES}}
  # Data parallel. Default 1: this is a wide-EP knob, not a general one.
  #
  # It exists because EP degree is NOT what "2 nodes with --enable-expert-parallel"
  # suggests. vLLM builds the EP group in parallel_state.py as
  #
  #     all_ranks.transpose(1, 2).reshape(
  #         -1, data_parallel_size * prefill_context_model_parallel_size
  #            * tensor_model_parallel_size).unbind(0)
  #
  # i.e. EP = DP x PCP x TP. **PP is absent from that product.** So TP8/PP2 on
  # two nodes is EP8, exactly the same expert sharding as one node -- the second
  # node adds pipeline depth, not expert width, and every expert is still
  # replicated across both stages. Adding nodes via PP can therefore never test
  # whether EP16 halves per-GPU weight reads. The only way to widen EP while
  # holding TP at 8 is DP.
  #
  # TP x PP x DP must equal the allocation, so on 2 nodes EP16 is TP8/PP1/DP2.
  DP=${ZENITH_DP:-1}

  # ---- health check stage ----
  headecho "*** Running series of health checks before launching inference ***"

  # shared helper imported by the ray_health_* checks; must land in /app (their
  # cwd) before any of them run
  ${PODMAN} cp lux_ray_util.py ${CNAME}:/app/lux_ray_util.py

  headecho "*** Fabric devices visible inside the container ***"
  ${PODMAN} exec ${CNAME} bash -lc '/opt/libfabric/bin/fi_info -l; \
      echo "--- providers actually usable ---"; \
      /opt/libfabric/bin/fi_info -p "${FI_PROVIDER}" 2>&1 | grep -E "provider|fabric|domain" | head -12; \
      echo "--- rdma devices ---"; (ibv_devinfo -l 2>/dev/null || echo none)'

  # flash attention 2 ROCm check
  headecho "*** Checking that container flash attention is using ROCm FA2 ***"
  HEALTH_FA2_KERNEL=ray_health_fa2.py
  ${PODMAN} cp ${HEALTH_FA2_KERNEL} ${CNAME}:/app/${HEALTH_FA2_KERNEL}
  ${PODMAN} exec ${CNAME} python ${HEALTH_FA2_KERNEL}

  # RCCL comms test on the communicators vLLM will build: TP all-reduce, the PP
  # chain handoff, and DP/EP when DP > 1. Never a world-wide comm -- see the
  # docstring of ray_health_rccl.py for why. A hang in any phase fails the check
  # after ZENITH_HEALTH_TIMEOUT seconds (default 180) and names the phase.
  HEALTH_RCCL_KERNEL=ray_health_rccl.py
  ${PODMAN} cp ${HEALTH_RCCL_KERNEL} ${CNAME}:/app/${HEALTH_RCCL_KERNEL}
  headecho "*** Testing vLLM's RCCL communicators for TP=${TP} x PP=${PP} x DP=${DP} on ${NNODES} nodes x ${GPUSPERNODE} GPUs ***"
  # NCCL_DEBUG defaults to WARN so a healthy run stays readable. Raise it to
  # INFO to find out *which transport* RCCL chose -- that is the only way to
  # tell a working fabric from a silent fall back to TCP sockets, because the
  # plugin-load failure that causes the fallback is logged at INFO level and
  # nothing at WARN mentions it. What to look for in the output:
  #
  #   NET/OFI ... provider: cxi        healthy on Slingshot
  #   via NET/RCCL-ANP                 healthy on Pollara RoCE (~936-942 Gbps at 16 ranks)
  #   NET/Socket                       fell back to TCP; ~1 Gbps on 1 GbE, ~70 Gbps on RoCE
  #
  # NCCL_DEBUG_SUBSYS=INIT,NET keeps INFO output to the interesting subsystems;
  # a bare INFO across 20 ranks is thousands of lines.
  NCCL_DEBUG_LEVEL="${ZENITH_NCCL_DEBUG:-WARN}"
  NCCL_DEBUG_ARGS=(--env NCCL_DEBUG="${NCCL_DEBUG_LEVEL}")
  if [ "${NCCL_DEBUG_LEVEL}" != "WARN" ]; then
      NCCL_DEBUG_ARGS+=(--env NCCL_DEBUG_SUBSYS="${ZENITH_NCCL_DEBUG_SUBSYS:-INIT,NET}")
      echo "*** NCCL_DEBUG=${NCCL_DEBUG_LEVEL}: grep this output for 'NET/' --"
      echo "***   'via NET/RCCL-ANP' (RoCE) or 'NET/OFI ... provider: cxi' is healthy,"
      echo "***   'NET/Socket' means RCCL fell back to TCP."
  fi
  if ! ${PODMAN} exec "${NCCL_DEBUG_ARGS[@]}" \
        --env ZENITH_HEALTH_TIMEOUT="${ZENITH_HEALTH_TIMEOUT:-180}" ${CNAME} python ${HEALTH_RCCL_KERNEL} \
        ${NNODES} ${CPUSPERNODE} ${GPUSPERNODE} ${TP} ${PP} ${DP}; then
      echo "*** RCCL health check FAILED -- not starting inference ***"
      exit 1
  fi

  # overall ray health
  HEALTH_OVERALL_KERNEL=ray_health_overall.py
  ${PODMAN} cp ${HEALTH_OVERALL_KERNEL} ${CNAME}:/app/${HEALTH_OVERALL_KERNEL}
  headecho "*** Launching tiny test kernel to check overall Ray cluster health ***"
  if ! ${PODMAN} exec ${CNAME} python ${HEALTH_OVERALL_KERNEL} \
        ${NNODES} ${CPUSPERNODE} ${GPUSPERNODE}; then
      echo "*** Ray cluster health check FAILED -- not starting inference ***"
      exit 1
  fi
  headecho "*** All health checks passed, moving to inference now ***"

  # ---- inference ----
  MODEL="${ZENITH_MODEL:-gpt-oss-120b}"
  MAX_MODEL_LEN="${ZENITH_MAX_MODEL_LEN:-100000}"
  TARGET_KV_GB="${ZENITH_TARGET_KV_GB:-64}"

  # ZENITH_NO_SERVE=1 is exempt from the world-size guard, and this is not a
  # loosening of it -- the guard is about the single cluster-wide server this
  # script starts, and under NO_SERVE it never starts one. serve_disagg.sh then
  # launches two independent vLLM instances, each with its own world size, and
  # the invariant that actually has to hold is per instance:
  #
  #     sum over instances of (TP x PP x DP)  ==  allocated GPUs
  #
  # Single-node disagg is the case that exposes the difference: two TP=4
  # instances, 4 + 4 = 8 GPUs, entirely valid, but TP*PP*DP = 4 != 8 and the
  # guard refused it (job 47024) after the containers and the RCCL health check
  # had already come up -- rejecting a topology it was never going to serve.
  #
  # The guard still runs in the serving path, where it is load-bearing: vLLM
  # requires world_size_across_dp == TP*PP*DP and hangs in Ray placement-group
  # creation rather than reporting the mismatch.
  if [ "${ZENITH_NO_SERVE:-0}" != "1" ] \
     && [ $(( TP * PP * DP )) -ne $(( NNODES * GPUSPERNODE )) ]; then
      echo "*** Refusing to serve: TP=${TP} x PP=${PP} x DP=${DP} = $(( TP * PP * DP )) does not"
      echo "***   match the ${NNODES} x ${GPUSPERNODE} = $(( NNODES * GPUSPERNODE )) GPUs allocated."
      echo "***   vLLM requires world_size_across_dp == TP*PP*DP; it would hang in"
      echo "***   Ray placement-group creation rather than say so."
      exit 1
  fi

  # TP may not exceed one node's GPU count. vLLM's Ray executor wants all
  # tensor-parallel ranks in one placement-group bundle set per node; when TP
  # is larger it emits "tensor_parallel_size=N is bigger than a reserved
  # number of M GPUs in a node" as a *warning* (ray_utils.py:325), builds the
  # group anyway, and then never starts the engine core. Job 46612 (TP=32 on
  # 4x8) sat at "Waiting for 1 local, 0 remote core engine proc(s) to start"
  # for 29 minutes with every GPU at 0% and the placement group already torn
  # down. The arithmetic guard above does not catch it -- 32x1 == 4x8 is fine
  # -- so it needs its own check. Costs a whole job's wall clock otherwise,
  # and on Lux a running job's TimeLimit cannot be extended.
  if [ "${TP}" -gt "${GPUSPERNODE}" ]; then
      echo "*** Refusing to serve: TP=${TP} exceeds the ${GPUSPERNODE} GPUs on a node."
      echo "***   vLLM would warn and then hang in engine-core startup rather"
      echo "***   than fail. Keep TP <= ${GPUSPERNODE} and scale with PP instead."
      exit 1
  fi

  # solve for the gpu-memory-utilization that yields TARGET_KV_GB of KV cache
  # per GPU. The default of 64 GB comes from the original MI355X requirement
  # (half of an MI300A's 128 GB HBM) and is still what Lux runs. It is NOT
  # right everywhere:
  #   * MI300A: use 32. Its HBM *is* the node's system memory, so 64 GB/GPU
  #     works out to ~85% of node RAM and the batch system's memory cgroup
  #     kills the job. More nodes do not help -- this target is per-GPU.
  #   * MI250X: 64 GiB per GCD makes 64 GB of KV impossible; see
  #     MI250X_PORTABILITY.md for the node-count/KV table.
  # The submit scripts set ZENITH_TARGET_KV_GB per site.
  #
  # --vram-bytes is passed explicitly rather than left to the script's own
  # detection. Both read the same KFD topology and agree, but the launcher has
  # already detected and validated the value above, and passing it keeps the
  # number that gets logged and the number that gets used identical. An early
  # version of the script defaulted to the MI355X figure when not given one,
  # which sized every MI300A run against a card 2.25x too large and produced
  # util 0.40 where 0.90 was needed; it now refuses to guess, but there is no
  # reason to make it detect a second time.
  SIZING_ARGS=(--model "${MODELPATH}/${MODEL}"
               --tp ${TP} --pp ${PP} --target-kv-gb ${TARGET_KV_GB}
               --max-model-len ${MAX_MODEL_LEN}
               --vram-bytes ${VRAM_BYTES})
  SIZING=$(python3 kv_cache_sizing.py "${SIZING_ARGS[@]}" 2>/dev/null)
  GPU_MEM_UTIL=$(echo "${SIZING}" | awk -F= '/^ZENITH_GPU_MEM_UTIL=/{print $2}')
  # 1 unless this model's expert intermediate dim, once sharded by TP, would
  # trip AITER's 32-element alignment assert during the fused-MoE weight shuffle
  AITER_MOE=$(echo "${SIZING}" | awk -F= '/^ZENITH_AITER_MOE=/{print $2}')
  : "${AITER_MOE:=1}"
  if [ -z "${GPU_MEM_UTIL}" ]; then
      echo "*** kv_cache_sizing.py could not size the KV cache for ${MODEL}"
      echo "*** (model may not fit on ${NNODES} node(s)); re-running for detail:"
      python3 kv_cache_sizing.py "${SIZING_ARGS[@]}"
      exit 1
  fi

  # Sanity-check the number before handing it to vLLM. A utilization that is
  # implausibly low is the signature of sizing against the wrong card, and it
  # fails as an OOM during model load rather than as anything that names the
  # cause -- so catch it here, where the numbers are still in hand.
  if awk "BEGIN{exit !(${GPU_MEM_UTIL} < 0.20)}"; then
      echo "*** Refusing to serve: computed gpu_memory_utilization"
      echo "***   ${GPU_MEM_UTIL} is implausibly low for ${MODEL} on ${NNODES}x${GPUSPERNODE} GPUs."
      echo "***   VRAM detected: ${VRAM_BYTES} bytes/GPU"
      echo "***   ($(awk "BEGIN{printf \"%.1f\", ${VRAM_BYTES}/1073741824}") GiB)."
      echo "***   If that GiB figure is wrong for this hardware, VRAM detection"
      echo "***   failed -- set ZENITH_VRAM_BYTES. If it is right, the model is"
      echo "***   small enough that this check is spurious; raise the threshold."
      exit 1
  fi

  headecho "*** Serving ${MODEL} on ${NNODES} nodes: TP=${TP} PP=${PP} DP=${DP} (EP=$(( DP * TP )) if --enable-expert-parallel)"
  headecho "***   max_model_len=${MAX_MODEL_LEN}  target KV=${TARGET_KV_GB} GB/GPU"
  headecho "***   VRAM=$(awk "BEGIN{printf \"%.0f\", ${VRAM_BYTES}/1073741824}") GiB/GPU (detected)"
  headecho "***   gpu_memory_utilization=${GPU_MEM_UTIL}"

  # environment variables needed for AMD-specific vllm invocation.
  #
  # INFO is right for normal runs. DEBUG exists because some of vLLM's
  # optional fast paths report their own disablement through
  # logger.debug_once -- one level below INFO. If such a path's preconditions
  # fail, the server still starts, still serves, and is simply slower, with
  # nothing at INFO to say so. Set ZENITH_VLLM_LOG_LEVEL=DEBUG to see why.
  VLLM_ENV_CMDLINE="--env VLLM_LOGGING_LEVEL=${ZENITH_VLLM_LOG_LEVEL:-INFO}"
  VLLM_ENV_CMDLINE+=" --env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
  VLLM_ENV_CMDLINE+=" --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
  VLLM_ENV_CMDLINE+=" --env SAFETENSORS_FAST_GPU=1"
  # NB: the variable is VLLM_ROCM_USE_AITER, not VLLM_USE_AITER. The latter was
  # set here too and does nothing -- vLLM says so itself on every startup:
  # "Unknown vLLM environment variable detected: VLLM_USE_AITER". Removed rather
  # than kept "just in case", because an unknown var that looks like the real
  # one is how you end up believing AITER is on when it is not.
  VLLM_ENV_CMDLINE+=" --env VLLM_ROCM_USE_AITER=1"
  # An A/B run (jobs 21778 vs 21773, RESULTS.md "Why the c=32 numbers move")
  # confirmed that re-adding VLLM_USE_AITER changes nothing measurable, so there
  # is no reason to carry it. Do not "restore" it if a c=32 number looks low.
  #
  # On MI250X (gfx90a / CDNA2) AITER is unsupported: vLLM gates its AITER
  # entry points on is_aiter_found_and_supported(), which requires
  # get_cdna_version() > 2, so this variable is mostly inert there. NOT
  # entirely: when VLLM_ROCM_USE_AITER is explicitly set, the unquantized-MoE
  # oracle (fused_moe/oracle/unquantized.py) skips its priority list and
  # *forces* the AITER backend, which then raises "Unquantized MoE backend ROCm
  # AITER does not support ... current device rocm". Inkling-MXFP4 has
  # unquantized MoE layers and died on exactly that (tioga job f618XyXKjXs5,
  # 2026-10-06), so CDNA2 also gets VLLM_ROCM_USE_AITER_MOE=0, which sends the
  # oracle back to its list (TRITON). Do not read "AITER=1" in an MI250X log
  # as evidence that AITER kernels are running.
  GFX_TARGET_VERSION=$(detect_gfx_target_version)
  # AITER's fused-MoE weight shuffle asserts the TP-sharded expert intermediate
  # dim is a multiple of 32. gpt-oss-20b (2880/8 = 360) is not; Inkling
  # (3072/8 = 384) is. Turn off just that kernel when it would fail -- the rest
  # of AITER (attention, rmsnorm, custom all-reduce) stays on.
  if [ "${GFX_TARGET_VERSION}" -ge 90000 ] 2>/dev/null && [ "${GFX_TARGET_VERSION}" -lt 90400 ]; then
      headecho "***   CDNA2 (gfx_target_version ${GFX_TARGET_VERSION}): AITER MoE off, oracle falls back to Triton"
      VLLM_ENV_CMDLINE+=" --env VLLM_ROCM_USE_AITER_MOE=0"
      VLLM_ENV_CMDLINE+=" --env VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0"
  elif [ "${AITER_MOE}" = "0" ]; then
      headecho "***   AITER fused-MoE disabled for this model/TP shape"
      VLLM_ENV_CMDLINE+=" --env VLLM_ROCM_USE_AITER_MOE=0"
      VLLM_ENV_CMDLINE+=" --env VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0"
  elif [ "${ZENITH_AITER_FUSE_SHARED_EXPERTS:-0}" = "1" ]; then
      # Shared-expert fusion is OPT-IN in vLLM: envs.py defaults
      # VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS to False, and a model that
      # could use it silently takes the unfused path, with a log that looks
      # identical to one that qualified. Turning it on requires this knob.
      #
      # Left off by default deliberately. The flag is global: it arms fusion
      # in the generic FusedMoEConfig path that every MoE model on this
      # cluster goes through, including Inkling on MI355X and MI300A. Those
      # configs are measured in RESULTS.md with the flag off, and vLLM asserts
      # on it outright when Mori kernels are in play. Per-layer fallback
      # exists (fused_moe/utils.py:87 logs a warning and skips) but "probably
      # degrades gracefully" is not a reason to change the default under
      # every model here.
      headecho "***   AITER shared-expert fusion requested"
      VLLM_ENV_CMDLINE+=" --env VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1"
  fi
  VLLM_ENV_CMDLINE+=" --env VLLM_WORKER_MULTIPROC_METHOD=spawn"
  VLLM_ENV_CMDLINE+=" --env VLLM_USE_MODELSCOPE=False"
  VLLM_ENV_CMDLINE+=" --env VLLM_NO_USAGE_STATS=1"
  VLLM_ENV_CMDLINE+=" --env DO_NOT_TRACK=1"
  # the model dir is a local path, so keep HF fully offline
  VLLM_ENV_CMDLINE+=" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1"

  # Arbitrary extra environment for the served vLLM process, as a
  # space-separated list of NAME=VALUE pairs:
  #
  #   ZENITH_VLLM_EXTRA_ENV="VLLM_USE_V2_MODEL_RUNNER=1"
  #
  # Every other env knob above is a named, documented flag because each one
  # encodes a decision worth explaining. This is the escape hatch for one-off
  # experiments that do not (yet) deserve a knob -- currently the V2 model
  # runner, which is the only way to get MTP working at PP>1 (the V1 runner
  # does not implement async scheduling with PP; see config/vllm.py:588).
  #
  # Values may not contain spaces: VLLM_EXEC_STRING is deliberately unquoted at
  # the podman exec below so flags word-split into argv, and a space here would
  # split one --env across two arguments. Checked rather than assumed, because
  # the failure is a confusing podman error a long way from this line.
  if [ -n "${ZENITH_VLLM_EXTRA_ENV:-}" ]; then
      for _kv in ${ZENITH_VLLM_EXTRA_ENV}; do
          case "${_kv}" in
              *=*) ;;
              *) headecho "***   ZENITH_VLLM_EXTRA_ENV: ignoring '${_kv}' (not NAME=VALUE)"
                 continue ;;
          esac
          headecho "***   extra env: ${_kv}"
          VLLM_ENV_CMDLINE+=" --env ${_kv}"
      done
  fi

  VLLM_COMMON_CMDLINE="--log-level=error ${CNAME} vllm serve /app/models/${MODEL}"
  VLLM_COMMON_CMDLINE+=" --served-model-name ${MODEL}"
  VLLM_COMMON_CMDLINE+=" --host 0.0.0.0 --port 8000"
  VLLM_COMMON_CMDLINE+=" --tensor-parallel-size ${TP}"
  VLLM_COMMON_CMDLINE+=" --pipeline-parallel-size ${PP}"
  # Only emitted when asked for. DP>1 changes the engine topology (one engine
  # core per DP rank rather than one overall), so passing --data-parallel-size 1
  # unconditionally would be a no-op in principle but is left off to keep the
  # default command line byte-identical to every run in RESULTS.md.
  if [ "${DP}" -gt 1 ]; then
      VLLM_COMMON_CMDLINE+=" --data-parallel-size ${DP}"
      # --data-parallel-backend ray is not optional here, and leaving it off is
      # how job 46902 died:
      #
      #   Exception: Error computing device indices for CUDA_VISIBLE_DEVICES:
      #     local range: [8, 16) base value: "0,1,2,3,4,5,6,7"
      #
      # The default DP backend is "mp", which assumes every DP rank is a local
      # process on this node and hands rank 1 the GPU indices [8,16). There are
      # only 8 GPUs per node, so it fails before the model loads. The mp path
      # expects the multi-node deployment to be launched per-node, with
      # --data-parallel-size-local, --data-parallel-address/-rpc-port, a
      # --data-parallel-start-rank per node and --headless on the non-head
      # ones -- a launch topology this script does not have and does not need.
      #
      # We already build a Ray cluster spanning the whole allocation and exec
      # vLLM once on the head (see the ray start above), which is exactly the
      # shape the ray DP backend wants: upstream documents it as "a single
      # launch command (on any node) is needed to start all local and remote DP
      # ranks", with address and rpc-port unnecessary. parallel.py's backend
      # selection takes it as an explicit override, so it wins over the
      # device-count heuristic that produced the [8,16) range.
      VLLM_COMMON_CMDLINE+=" --data-parallel-backend ray"
      # ...and --data-parallel-address is not optional either, which is how the
      # *next* attempt (job 46950) died, one layer further in:
      #
      #   AssertionError: The DP master node (ip: 104.207.140.221) is missing
      #                   or dead
      #     v1/engine/utils.py:569 create_dp_placement_groups
      #
      # Nothing is dead. 104.207.140.221 is this node's *public WAN* address.
      # With --data-parallel-address unset, arg_utils.py:2302 takes the ray
      # branch and calls get_ip(), which resolves the local address by opening
      # a UDP socket to 8.8.8.8 and reading back getsockname() -- i.e. it
      # returns whichever interface holds the default route. On Lux that is the
      # public interface, not the 10.2.96.0/24 cluster network. Meanwhile
      # run_container.sh starts every raylet with
      # --node-ip-address=${MY_ETH_ADDRESS} on that cluster network, so Ray's
      # resource table is keyed "node:10.2.96.14" and the lookup for
      # "node:104.207.140.221" finds nothing. The assert reports it as a dead
      # node, which sends you looking at the fabric instead of at the hostname.
      #
      # HEADNODE_ETH_ADDRESS is the same value the head wrote to HEAD_ETH_FILE
      # and the same one it passed to `ray start --head --node-ip-address`, so
      # by construction it is a key Ray knows. Using it rather than
      # MY_ETH_ADDRESS is deliberate even though this block only runs on the
      # head: it states *which* node is meant instead of relying on where the
      # code happens to execute.
      VLLM_COMMON_CMDLINE+=" --data-parallel-address ${HEADNODE_ETH_ADDRESS}"
      # Third and last layer of the same onion (job 46951):
      #
      #   ValueError: Not enough resources to allocate 2 DP ranks on DP master
      #               node 10.2.96.14, possible to fit 1 DP ranks.
      #     v1/engine/utils.py:680 create_dp_placement_groups
      #
      # With --data-parallel-size-local unset and pack strategy "strict" (the
      # default), arg_utils.py:2295 sets data_parallel_size_local =
      # data_parallel_size -- i.e. "put every DP rank on this node". The
      # placement loop then computes dp_size_available = 8 // world_size = 8//8
      # = 1 for the master and refuses. That default is written for the
      # single-node "DP over local GPUs" case; it is simply wrong for a DP group
      # that spans nodes.
      #
      # DP_LOCAL is the number of DP ranks *per node*. Each DP rank is one full
      # world_size = TP x PP group, so a node holds GPUSPERNODE / (TP*PP) of
      # them. Computing it rather than hardcoding 1 keeps TP4/DP4-on-one-node
      # working if anyone tries it.
      #
      # Note "span" is not the escape hatch it sounds like: it asserts
      # world_size > max_device_per_node, and ours is exactly equal (8 == 8), so
      # it fails a different assert. strict + correct DP_LOCAL is the path.
      DP_LOCAL=$(( GPUSPERNODE / (TP * PP) ))
      [ "${DP_LOCAL}" -lt 1 ] && DP_LOCAL=1
      VLLM_COMMON_CMDLINE+=" --data-parallel-size-local ${DP_LOCAL}"
  fi
  VLLM_COMMON_CMDLINE+=" --max-model-len ${MAX_MODEL_LEN}"
  VLLM_COMMON_CMDLINE+=" --gpu-memory-utilization ${GPU_MEM_UTIL}"
  VLLM_COMMON_CMDLINE+=" --trust-remote-code"

  # Reasoning and tool-call parsers, for agents on the chat API. Inkling
  # reasons by default (its chat template's reasoning_effort defaults to 0.9).
  # Without a reasoning parser the thinking comes back inside `content`, and
  # without a tool parser tool calls come back as raw text. With the inkling
  # parsers they land in `reasoning` and `tool_calls`. /v1/completions (the
  # benchmarks) is unaffected. "auto" picks the parser from the model name
  # (Mistral* gets the mistral tool parser and no reasoning parser), "none"
  # turns it off, and anything else is passed to vLLM as the parser name.
  _auto_parser() {
      case "$1" in
          auto) case "${MODEL}" in
                    Inkling*) echo inkling ;;
                    Mistral*) if [ "$2" = tool ]; then echo mistral; fi ;;
                esac ;;
          none|"") ;;
          *) echo "$1" ;;
      esac
  }
  REASONING_PARSER=$(_auto_parser "${ZENITH_REASONING_PARSER:-auto}" reasoning)
  TOOL_CALL_PARSER=$(_auto_parser "${ZENITH_TOOL_CALL_PARSER:-auto}" tool)
  if [ -n "${REASONING_PARSER}" ]; then
      VLLM_COMMON_CMDLINE+=" --reasoning-parser ${REASONING_PARSER}"
      headecho "***   reasoning parser: ${REASONING_PARSER} (ZENITH_REASONING_PARSER=none to disable)"
  fi
  if [ -n "${TOOL_CALL_PARSER}" ]; then
      VLLM_COMMON_CMDLINE+=" --enable-auto-tool-choice --tool-call-parser ${TOOL_CALL_PARSER}"
      headecho "***   tool-call parser: ${TOOL_CALL_PARSER} (ZENITH_TOOL_CALL_PARSER=none to disable)"
  fi

  # ---- scheduler batch limits ----
  #
  # Both unset by default, which keeps vLLM's own defaults (max_num_seqs 1024,
  # max_num_batched_tokens 16384 in the OPENAI_API_SERVER context) and so
  # leaves every existing result reproducible. They are overridable because at
  # the 8:1 shape they are the binding constraint on the high-concurrency end
  # of the curve, and that end is where the published B200 figure earns its
  # steep limb:
  #
  #   * 88.7% of every token counted at 8k/1k is prompt, and prefill is what
  #     batches well. max_num_batched_tokens=16384 admits only ~2 concurrent
  #     8k prefills per step, which caps the prefill throughput the curve is
  #     mostly made of.
  #   * max_num_seqs bounds the decode batch independently of whether the KV
  #     pool could hold more. Raising TARGET_KV_GB without raising this just
  #     moves the bottleneck one step along.
  #
  # Raising either costs activation memory, which is drawn from the same
  # budget kv_cache_sizing.py hands to the KV pool -- so if you raise these,
  # re-check that the run still fits rather than assuming the headroom is free.
  if [ -n "${ZENITH_MAX_NUM_BATCHED_TOKENS:-}" ]; then
      VLLM_COMMON_CMDLINE+=" --max-num-batched-tokens ${ZENITH_MAX_NUM_BATCHED_TOKENS}"
  fi
  if [ -n "${ZENITH_MAX_NUM_SEQS:-}" ]; then
      VLLM_COMMON_CMDLINE+=" --max-num-seqs ${ZENITH_MAX_NUM_SEQS}"
  fi
  # ZENITH_MAX_CUDAGRAPH_CAPTURE_SIZE raises the largest batch vLLM captures a
  # CUDA graph for. vLLM stops at 512, and with MTP a decode step carries
  # (1 + drafts) tokens per sequence, so past c=128 at 3 drafts every decode
  # step ran without a graph. Empty = vLLM's default. submit_lux.slurm.sh
  # sets 2048 for Inkling-MXFP4 on one node. If ZENITH_VLLM_EXTRA_ARGS carries
  # its own --compilation-config, leave this empty and put the capture size in
  # that one instead of passing the flag twice.
  if [ -n "${ZENITH_MAX_CUDAGRAPH_CAPTURE_SIZE:-}" ]; then
      VLLM_COMMON_CMDLINE+=" --compilation-config {\"max_cudagraph_capture_size\":${ZENITH_MAX_CUDAGRAPH_CAPTURE_SIZE}}"
  fi
  # ray is required for multinode; for a single node the mp backend has lower
  # per-step overhead, so only pay for ray when we actually span nodes
  if [ ${NNODES} -gt 1 ]; then
      VLLM_COMMON_CMDLINE+=" --distributed-executor-backend ray"
  else
      VLLM_COMMON_CMDLINE+=" --distributed-executor-backend mp"
  fi

  # ---- MTP (multi-token prediction) ----
  #
  # Speculative decoding using the checkpoint's own next-token-prediction head.
  # This is ON by default because it pays at low and moderate concurrency, for
  # a few percent of KV capacity.
  #
  # It is gated on the checkpoint, not on the model name: the MTP weights have
  # to be on disk or vLLM fails at load. `num_nextn_predict_layers` in
  # config.json is the declaration. gpt-oss has no MTP head and is left alone.
  #
  # Inkling declares it one level down, as mtp_config.num_nextn_predict_layers
  # (8 depth blocks on Inkling and Inkling-Small), which vLLM reads itself
  # (config/speculative.py turns the checkpoint into an inkling_mtp draft
  # model). Inkling MTP has no measured run in this project yet, so a nested
  # declaration only turns MTP on when ZENITH_MTP=1 is set explicitly; once a
  # run has sane acceptance and passes the correctness gate, drop that gate.
  #
  # num_speculative_tokens defaults to the layer count. Asking for more drafts
  # than there are MTP layers is not a speedup; vLLM would have to reuse the
  # single head autoregressively. ZENITH_MTP_TOKENS asks for fewer (capped at
  # the layer count): Inkling builds only the first N depth blocks, one draft
  # forward each, so on a deep head a shorter chain can be the faster one.
  #
  # Set ZENITH_MTP=0 to turn it off -- needed for any baseline-vs-MTP
  # comparison, and at very high concurrency: MTP spends compute slack, and
  # once decode is no longer memory-bound it can be a small loss.
  # prints "<layers> top" or "<layers> mtp_config"
  MTP_DECL=$(python3 -c "
import json,sys
try:
    c = json.load(open('${MODELPATH}/${MODEL}/config.json'))
    n = int(c.get('num_nextn_predict_layers') or 0)
    if n:
        print(n, 'top')
    else:
        print(int((c.get('mtp_config') or {}).get('num_nextn_predict_layers') or 0), 'mtp_config')
except Exception:
    print(0, 'top')
" 2>/dev/null || echo "0 top")
  MTP_LAYERS=${MTP_DECL%% *}
  : "${MTP_LAYERS:=0}"
  MTP_NEEDS_OPT_IN=0
  if [ "${MTP_LAYERS}" -gt 0 ] && [ "${MTP_DECL#* }" = "mtp_config" ] && [ "${ZENITH_MTP:-}" != "1" ]; then
      MTP_NEEDS_OPT_IN=1
      MTP_LAYERS=0
  fi
  if [ "${MTP_LAYERS}" -gt 0 ] && [ -n "${ZENITH_MTP_TOKENS:-}" ] && [ "${ZENITH_MTP_TOKENS}" -lt "${MTP_LAYERS}" ]; then
      MTP_LAYERS=${ZENITH_MTP_TOKENS}
  fi

  # MTP and pipeline parallelism cannot be combined on this vLLM build. Not
  # "are slower together" -- there is no working configuration of the two, and
  # the two ways to ask for it fail differently:
  #
  #   async scheduling ON (the default vLLM picks for MTP, because
  #   EagleModelTypes includes "mtp" and the compatibility check in
  #   config/vllm.py never considers PP):
  #     the last PP stage broadcasts its sampled tokens back to the earlier
  #     stages via _pp_broadcast_prev_sampled_token_ids, which asserts
  #         sampled_token_ids.dim() == 2 and shape[-1] == 1
  #         "PP+async expects sampled_token_ids to have shape [num_reqs, 1]"
  #     MTP makes that tensor [num_reqs, 1+num_spec], so it fires on the first
  #     decode step. EngineDeadError, every in-flight request returns HTTP 500.
  #     Job 46884 died this way.
  #
  #   async scheduling OFF (what we tried next, and it is worse -- it hangs
  #   instead of crashing):
  #     engine/core.py post_step() takes the non-async branch
  #         if self.check_for_draft_tokens and not self.async_scheduling ...:
  #             draft_token_ids = self.model_executor.take_draft_token_ids()
  #     which is a *blocking* collective_rpc to output_rank. PP keeps a
  #     batch_queue of pp_size batches in flight, so at that moment the last
  #     stage is still inside execute_model blocked on irecv_tensor_dict
  #     waiting for activations from stage 0 -- and stage 0 cannot send them,
  #     because the engine that would issue its next execute_model is the very
  #     thing blocked in take_draft_token_ids. Closed cycle, no timeout.
  #     Job 46886 hung exactly here; py-spy on all three showed engine in
  #     take_draft_token_ids -> collective_rpc, Worker_PP1_TP0 in
  #     irecv_tensor_dict, Worker_PP0_TP0 idle on an empty RPC queue. The
  #     server answers /v1/models, prefills at 1040 tok/s, then generates
  #     0.2 tok/s forever with "Running: 1 reqs".
  #
  # post_step's own comment says why: "When using async scheduling we can't get
  # draft token ids in advance, so we update draft token ids in the worker
  # process". Spec decode across PP was designed to collect drafts inside the
  # worker, on the async path. Turning async off does not fall back to a
  # working path, it re-enables one that PP invalidates.
  #
  # So MTP yields and async scheduling stays. MTP is worth more than async
  # scheduling is on its own, but that trade is not on offer -- the actual choice is "PP without MTP" versus "does not run".
  # Note this makes multi-node PP cells non-comparable to single-node cells on
  # equal footing: the 1-node rows have MTP and the PP rows cannot, which is a
  # real handicap on the multi-node side and must be stated wherever the matrix
  # is reported, not silently absorbed into the fabric's column.
  #
  # Worth reporting upstream. The assert path is a genuine bug (the broadcast
  # could carry [num_reqs, 1+num_spec]); the deadlock is arguably a missing
  # guard -- config/vllm.py hard-fails several async+spec combinations already
  # and simply has no case for PP.
  # UPDATE: the assert path above now has a fix, so the suppression is opt-out
  # rather than unconditional. patches/spec_decode_pp_broadcast_width.py moves
  # the sampled-token broadcast to *after* the drafter reduces it, so the wire
  # format stays [num_reqs, 1] and the assert never fires. Note the fix is NOT
  # "relax the assert": the raw tensor's column 0 is the first accepted token,
  # while prev_sampled_token_ids must hold the last, so widening the broadcast
  # would have rewound input_ids on every multi-token acceptance -- silently
  # wrong output that a throughput benchmark would happily report as a win.
  #
  # The deadlock path (async scheduling OFF) is untouched and still fatal, so
  # MTP+PP is only attempted with async scheduling ON.
  #
  # ZENITH_MTP_PP=1 opts in. It stays off by default because the fix is not yet
  # backed by a measured multi-node run; once a PP=2 MTP cell is in the matrix
  # with sane acceptance and output, flip the default here.
  MTP_SUPPRESSED_BY_PP=0
  if [ "${MTP_LAYERS}" -gt 0 ] && [ "${PP}" -gt 1 ] && [ "${ZENITH_MTP:-1}" = "1" ]; then
      if [ "${ZENITH_MTP_PP:-0}" = "1" ]; then
          # Async scheduling is turned off by passing --no-async-scheduling
          # through ZENITH_VLLM_EXTRA_ARGS (there is no dedicated knob), so
          # that string is what has to be inspected here.
          case " ${ZENITH_VLLM_EXTRA_ARGS:-} " in
              *" --no-async-scheduling "*) _MTP_PP_ASYNC_OFF=1 ;;
              *) _MTP_PP_ASYNC_OFF=0 ;;
          esac
          if [ "${_MTP_PP_ASYNC_OFF}" = "1" ]; then
              # Refuse rather than hang. Async-off + MTP + PP is the closed-cycle
              # deadlock described above; it has no timeout and burns the whole
              # walltime looking like a slow server.
              headecho "***   MTP off: ZENITH_MTP_PP=1 needs async scheduling ON"
              headecho "***     (async-off + MTP + PP deadlocks in take_draft_token_ids)"
              MTP_LAYERS=0
              MTP_SUPPRESSED_BY_PP=1
          else
              headecho "***   MTP+PP ENABLED (ZENITH_MTP_PP=1, PP=${PP}) -- EXPERIMENTAL"
              headecho "***     relies on patches/spec_decode_pp_broadcast_width.py"
              headecho "***     + patches/spec_decode_drafter_pp_guard.py"
              headecho "***     verify output quality before quoting throughput"
              if [ "${ZENITH_PP_DEFERRED_RECV:-1}" != "0" ]; then
                  headecho "***     WARNING: hangs in warmup on Lux unless ZENITH_PP_DEFERRED_RECV=0"
              fi
          fi
      else
          headecho "***   MTP off: PP=${PP} (MTP+PP deadlocks/asserts on this vLLM)"
          headecho "***     set ZENITH_MTP_PP=1 to try the patched broadcast path"
          MTP_LAYERS=0
          MTP_SUPPRESSED_BY_PP=1
      fi
  fi

  if [ "${ZENITH_MTP:-1}" = "1" ] && [ "${MTP_LAYERS}" -gt 0 ]; then
      headecho "***   MTP enabled: ${MTP_LAYERS} draft token(s)/step (ZENITH_MTP=0 to disable)"
      # No spaces inside the JSON: VLLM_EXEC_STRING is deliberately unquoted at
      # the podman exec below so that each flag word-splits into its own argv
      # entry. A space here would split the JSON across two arguments and
      # argparse would reject it.
      VLLM_COMMON_CMDLINE+=" --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_LAYERS}}"
      # Async scheduling is deliberately left at vLLM's default (on) here. It
      # is only reachable at PP=1 now, which is the configuration the MTP
      # results were measured in. See the MTP+PP note above for why --no-async-scheduling
      # is not the escape hatch it looks like.
  elif [ "${MTP_SUPPRESSED_BY_PP}" = "1" ]; then
      : # already reported above: the model has MTP weights, PP>1 forbids them
  elif [ "${MTP_NEEDS_OPT_IN}" = "1" ]; then
      headecho "***   MTP off: ${MODEL} has MTP weights (mtp_config), not yet validated here;"
      headecho "***     ZENITH_MTP=1 turns it on"
  elif [ "${ZENITH_MTP:-1}" = "1" ]; then
      headecho "***   MTP unavailable: ${MODEL} declares no num_nextn_predict_layers"
  else
      headecho "***   MTP disabled by ZENITH_MTP=0"
  fi

  # Expert parallelism is deliberately not a knob here. vLLM takes
  # --enable-expert-parallel as a boolean and derives the EP degree as TP*DP,
  # so there is no size to pass; ZENITH_VLLM_EXTRA_ARGS="--enable-expert-parallel"
  # is the whole interface.
  VLLM_EXEC_STRING="${VLLM_ENV_CMDLINE} ${VLLM_COMMON_CMDLINE} ${ZENITH_VLLM_EXTRA_ARGS:-}"
  stdbuf -oL echo "vLLM exec string is: ${PODMAN} exec ${VLLM_EXEC_STRING}"

  # ZENITH_NO_SERVE=1 brings up the containers and the Ray cluster, runs the
  # health checks, and then holds -- without starting a server. It exists for
  # disaggregated prefill (serve_disagg.sh), which needs *two
  # differently-configured* vLLM instances on different nodes rather than the
  # one cluster-wide server this script's whole structure assumes.
  # Rather than teach this script a second serving topology -- the P and D sides
  # differ in role, port, side-channel address and MTP state -- it steps aside
  # and lets serve_disagg.sh podman-exec into the containers it prepared.
  #
  # Everything up to this point still runs, which is the point: the RCCL health
  # check, the KV sizing, and the GPU/fabric setup are all shared, and skipping
  # them would mean a disagg run that fails on a bad node without saying so.
  if [ "${ZENITH_NO_SERVE:-0}" = "1" ]; then
      headecho "*** ZENITH_NO_SERVE=1: cluster is up, not serving."
      headecho "***   run: ./serve_disagg.sh <prefill-host> <decode-host>"
      wait ${CONTAINER_PID}
  else
      # exec vllm
      ${PODMAN} exec ${VLLM_EXEC_STRING}
  fi

else
  # workers just keep their container alive for the duration of the job
  wait ${CONTAINER_PID}
fi
