#!/bin/bash
# Disaggregated prefill: one prefill vLLM and one decode vLLM on separate nodes,
# with KV moved over NIXL/UCX, behind a proxy that speaks one OpenAI endpoint.
#
# WHY THIS TOPOLOGY AND NOT PP/EP
# -------------------------------
# Every other multi-node arrangement (PP, EP) moves *activations* across the
# fabric on every layer of every step, and in our measurements none of them
# beat one node -- twice the hardware, no gain. Disagg is structurally different: it moves the KV cache **once per request**,
# at the prefill/decode boundary, and nothing per-layer. It is the one topology
# whose communication profile our fabric might actually suit.
#
# It also matches the workload. At our house 8:1 shape, ~89% of tokens are
# prefill, so specialising the two phases is worth more here than at 1:1 -- the
# prefiller runs compute-bound with no decode requests stealing its SMs, and the
# decoder runs memory-bound with no prefill chunks inflating its batch latency.
#
# Expect this to be uncertain in both directions. Every published disagg win is
# on NVLink-class interconnect, and a fat per-token KV makes the per-request
# transfer larger than in the published cases. A negative result is a legitimate outcome and should be reported as
# one, not retried until it looks good.
#
# WHAT HAD TO BE FIXED BEFORE THIS COULD RUN AT ALL
# -------------------------------------------------
# vLLM's NIXL connector is in the image and imports cleanly, but every path
# through it died at base_worker.py:519 "RuntimeError: NIXL is not available".
# vllm/distributed/nixl_utils.py dispatches the package name on platform --
# "nixl_rocm" on ROCm, "nixl" elsewhere -- and nixl_rocm existed nowhere: not in
# the image, not on PyPI (PyPI "nixl" is a CUDA meta-package depending on
# nixl-cu12/cu13; "nixl-rocm" 404s; the "rixl" name llm-d's ROCm images use is,
# on PyPI, an unrelated project). build_nixl_rocm.sh builds it from
# ai-dynamo/nixl with -Dwheel_variant=rocm inside this same image;
# run_container.sh stages the resulting wheel. Verified: vLLM reports "NIXL is
# available" and the agent exposes plugins ['POSIX', 'UCX'].
#
# usage:  serve_disagg.sh <jobid> <prefill-hosts> <decode-hosts> [proxy-port]
# Run from the login node against a >=2 node allocation whose containers are
# already up -- either ZENITH_NO_SERVE=1 from the start, or an allocation whose
# server has been stopped. The jobid is explicit rather than read from the
# environment so this works from outside the job step.
#
# xPyD: the host arguments are comma-separated lists, one entry per instance,
# each instance being one whole node at TP=8. So
#     serve_disagg.sh 47018 a7,a8 a9,a11
# is 2 prefillers and 2 decoders across a 4-node allocation. A single host on
# each side is the 1P1D case and is handled by exactly the same code path --
# with one entry per list every derived value below (ports, side-channel
# offsets, the proxy command line) reduces to what the 1P1D version emitted, so
# the validated 2-node configuration is unchanged rather than re-derived.
#
# vLLM's own toy_proxy_server.py already accepts nargs="+" for
# --prefiller-hosts/--prefiller-ports and round-robins over the pools
# (toy_proxy_server.py:95-127), so no proxy-side change is needed; this script
# was the only thing hardcoding 1P1D.
set -u

JOBID="${1:?usage: serve_disagg.sh <jobid> <prefill-hosts> <decode-hosts> [proxy-port]}"
PREFILL_SPEC="${2:?}"
DECODE_SPEC="${3:?}"
PROXY_PORT="${4:-8000}"

IFS=',' read -r -a PREFILL_HOSTS <<< "${PREFILL_SPEC}"
IFS=',' read -r -a DECODE_HOSTS  <<< "${DECODE_SPEC}"
N_PREFILL=${#PREFILL_HOSTS[@]}
N_DECODE=${#DECODE_HOSTS[@]}
if [ "${N_PREFILL}" -lt 1 ] || [ "${N_DECODE}" -lt 1 ]; then
    echo "*** FATAL: need at least one prefill and one decode host" >&2
    exit 1
fi
# The proxy runs on the first prefiller, so that is also the host the benchmark
# driver points at.
PREFILL_HOST="${PREFILL_HOSTS[0]}"
DECODE_HOST="${DECODE_HOSTS[0]}"

MODEL="${ZENITH_MODEL:-Inkling-MXFP4}"
MAX_MODEL_LEN="${ZENITH_MAX_MODEL_LEN:-100000}"
TP="${ZENITH_TP:-8}"
GPU_MEM_UTIL="${ZENITH_GPU_MEM_UTIL:-0.6157}"

# Roles. kv_both is deprecated for NixlConnector; the prefiller produces and the
# decoder consumes, and getting these backwards does not error -- it just never
# transfers anything and the decoder recomputes, which looks like "disagg is
# slow" rather than "disagg is misconfigured".
KV_P='{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
KV_D='{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'

# The NIXL side channel is how the decoder fetches the prefiller's agent
# metadata; it is a separate socket from the OpenAI port and must be reachable
# across nodes, so it binds the node's cluster address rather than the
# "localhost" default. Same class of bug as the DP master address: the default
# is written for a single-node example and silently fails to connect off-node.
SIDE_CHANNEL_PORT="${ZENITH_NIXL_SIDE_CHANNEL_PORT:-5559}"

# ...and passing the *hostname* is not enough, which cost a full startup cycle
# to find. Lux's /etc/hosts lists the Debian-style loopback alias first:
#
#     127.0.1.1 node-a5 node-a5
#     10.0.0.24 node-a5
#
# so VLLM_NIXL_SIDE_CHANNEL_HOST=node-a5 resolves to 127.0.1.1 and the
# ROUTER socket binds there -- confirmed with `ss -ltnp`:
#
#     LISTEN 127.0.1.1:5559 users:(("VLLM::EngineCor",...))
#
# The prefiller then comes up healthy, serves /v1/models, registers KV, and
# advertises an address no other node can reach. The decoder's handshake to
# :5559 gets no reply and dies in recv_multipart with zmq.error.Again
# ("Resource temporarily unavailable"), surfacing as a 500 from the decoder and
# an opaque HTTPStatusError at the proxy -- three layers away from the cause.
#
# So resolve to the routable 10.2.x address explicitly rather than trusting
# name resolution. getent hosts returns the /etc/hosts order, which is exactly
# the wrong answer here, hence the grep for the cluster subnet.
# ZENITH_CLUSTER_NET_RE: extended regex for YOUR cluster network's first two
# octets (default '10\.2\.', the network these runs used).
CLUSTER_NET_RE="${ZENITH_CLUSTER_NET_RE:-10\.2\.}"
resolve_cluster_ip() {
    local host="$1"
    local ip
    ip=$(getent ahostsv4 "${host}" 2>/dev/null \
         | grep -oE "${CLUSTER_NET_RE}[0-9]+\.[0-9]+" | head -1)
    if [ -z "${ip}" ]; then
        echo "*** FATAL: cannot resolve a ${CLUSTER_NET_RE} cluster address for ${host} (set ZENITH_CLUSTER_NET_RE)" >&2
        exit 1
    fi
    echo "${ip}"
}
PREFILL_PORT=8100
DECODE_PORT=8200

# Every instance owns a whole node at TP=8, so the OpenAI port and the NIXL
# side-channel port can be the same number on each -- they never collide. The
# side channel is still offset by role (prefill SIDE_CHANNEL_PORT, decode
# +100) purely so a stray connection to the wrong role fails loudly instead of
# half-working.
#
# Resolve every host's cluster IP up front: the failure mode this guards
# against (binding the 127.0.1.1 loopback alias) applies per instance, and a
# 4-node run that gets 3 right and 1 wrong presents as an intermittent
# handshake timeout rather than a configuration error.
PREFILL_IPS=()
for h in "${PREFILL_HOSTS[@]}"; do PREFILL_IPS+=("$(resolve_cluster_ip "${h}")"); done
DECODE_IPS=()
for h in "${DECODE_HOSTS[@]}"; do DECODE_IPS+=("$(resolve_cluster_ip "${h}")"); done

# MTP must be the SAME on both sides, and this cost a startup cycle to learn.
#
# The original design here was asymmetric and the reasoning was sound on its
# face: MTP drafts tokens for the *decode* phase, the prefiller never decodes,
# so running it only on the decoder saves the compute-bound node a draft
# model's weights and warmup while keeping the MTP gain where it matters.
#
# vLLM forbids it. `speculative_config` is one of the factors in the NIXL
# compatibility hash (nixl/metadata.py:193), so an MTP-off producer and an
# MTP-on consumer compute different hashes and the handshake refuses:
#
#     RuntimeError: NIXL compatibility hash mismatch.
#       Local: f4b4abe11261ff7b...  Remote: 1b43a5b66059f20b...
#
# And it is a real incompatibility, not an over-strict check. With MTP the
# decoder carries an extra MTP layer in its KV cache, and the scheduler's
# `_prefill_backoff()` (nixl/base_scheduler.py) returns
# `num_prefill_lookahead_tokens - 1` so that the prefiller stops short of the
# drafter's lookahead window -- the P side must *know* the D side drafts, or it
# embeds unverified draft tokens in the transferred KV that the decoder will
# never rebuild. The two sides genuinely have to agree.
#
# There is an escape hatch --
#   --kv-transfer-config '{"kv_connector_extra_config":
#                          {"enforce_handshake_compat": false}}'
# -- and it is the wrong move here. It does not make the configurations
# compatible; it stops checking. The result transfers KV built under one
# lookahead assumption into an engine holding another, which surfaces as
# subtly wrong continuations rather than an error -- the same failure mode
# that nearly slipped through on the MTP+PP broadcast patch. Do not set it to
# make a run start.
#
# So: symmetric. ZENITH_MTP=1 puts MTP on both instances, ZENITH_MTP=0 on
# neither. The prefiller pays for a drafter it does not use, which is a real
# cost of disagg on this build and belongs in the measured result rather than
# being configured away.
MTP_ARGS=""
if [ "${ZENITH_MTP:-1}" = "1" ]; then
    MTP_ARGS=' --speculative-config {"method":"mtp","num_speculative_tokens":1}'
fi

# Expert parallelism, off by default. EP=DPxTP, so a single TP=8 instance gains
# nothing from it; it is here for completeness and for the MXFP4 experiments.
# NB it does NOT fix the shared-expert load failure described below -- that was
# the first hypothesis and it is wrong. EP was measured on (workers come up
# named Worker_TP0_EP0, with the experts split across ranks) and the assert
# still fired.
EP_ARGS=""
if [ "${ZENITH_EXPERT_PARALLEL:-0}" = "1" ]; then
    EP_ARGS=" --enable-expert-parallel"
fi

# ---------------------------------------------------------------------------
# VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is set for any *MXFP4* model name.
# For an MXFP4 checkpoint that stores its shared expert packed, this is a
# correctness prerequisite rather than the performance knob its name suggests.
#
# Without it, every worker of such a checkpoint dies during weight load at
#     parameter.py:172 load_merged_column_weight
#     assert param_data.shape == loaded_weight.shape
# which upstream raises bare -- no tensor name, no shapes -- and which reaches
# the operator only as
#     RuntimeError: Engine core initialization failed. Failed core proc(s): {}
# well into a checkpoint read.
#
# Patching the assert to print its operands (job 47025) showed the loaded
# tensor at exactly half the parameter's width on dim 1, which tensor
# parallelism never touches -- the MXFP4 packing ratio (two fp4 values per
# byte) -- with packed_dim=None, i.e. vLLM built the parameter as an
# *unpacked* ModelWeightParameter. With fusion off, the shared expert is
# materialised as a plain dense gate_up_proj expecting unpacked input
# features; the checkpoint supplies the packed MXFP4 tensor, and the two
# cannot meet. With fusion on, the layer is built as the fused MXFP4
# projection the checkpoint was quantised for and the shapes agree.
#
# Keyed off the model name rather than behind a knob. The launcher gates the
# same variable behind ZENITH_AITER_FUSE_SHARED_EXPERTS because there the flag
# is global across every MoE model on the cluster (Inkling on MI355X/MI300A
# are measured with it off); this script only ever serves the one model named
# in ZENITH_MODEL.
AITER_ENV="--env VLLM_ROCM_USE_AITER=1"
case "${MODEL}" in
    *MXFP4*)
        AITER_ENV+=" --env VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1"
        echo "*** MXFP4 checkpoint: AITER shared-expert fusion ON (required for weight load)"
        ;;
esac

common_args() {
    echo "--served-model-name ${MODEL}" \
         "--host 0.0.0.0" \
         "--max-model-len ${MAX_MODEL_LEN}" \
         "--gpu-memory-utilization ${GPU_MEM_UTIL}" \
         "--tensor-parallel-size ${TP}" \
         "--trust-remote-code" \
         "--distributed-executor-backend mp"${EP_ARGS}
}

# Distinct hosts, not instance count: single-node disagg puts both instances on
# one host (ZENITH_PREFILL_GPUS/ZENITH_DECODE_GPUS), so N_PREFILL+N_DECODE
# would label that run "2 nodes" and mislabel the result it produces.
N_HOSTS=$(printf '%s\n' "${PREFILL_HOSTS[@]}" "${DECODE_HOSTS[@]}" | sort -u | wc -l)
echo "*** Disaggregated prefill (${N_PREFILL}P${N_DECODE}D, ${N_HOSTS} node(s))"
echo "***   prefill : ${PREFILL_HOSTS[*]} :${PREFILL_PORT}  (TP=${TP}, kv_producer, MTP ${ZENITH_MTP:-1})"
echo "***   decode  : ${DECODE_HOSTS[*]} :${DECODE_PORT}  (TP=${TP}, kv_consumer, MTP ${ZENITH_MTP:-1})"
echo "***   nixl side channel : ${PREFILL_IPS[*]} :${SIDE_CHANNEL_PORT} (prefill) " \
     "${DECODE_IPS[*]} :$(( SIDE_CHANNEL_PORT + 100 )) (decode)"
echo "***   proxy   : 0.0.0.0:${PROXY_PORT}"

source ./lux_podman_env.sh

# This path is UCX, not RCCL, so none of the RCCL-ANP plumbing applies -- the
# ANP plugin is an NCCL_NET_PLUGIN and UCX does not consult it. The 930 Gbps
# measured for RCCL therefore does NOT carry over; UCX's transport has to be
# established separately, which is what ZENITH_DISAGG_TRANSPORT below selects.
#
# ZENITH_UCX_EXTRA is for diagnosis: UCX reports failures to NIXL as a bare
# nixlBackendError/NIXL_ERR_BACKEND with no reason attached, so the only way to
# learn *why* a transport failed is to turn UCX's own logging up, e.g.
#   ZENITH_UCX_EXTRA="--env UCX_LOG_LEVEL=debug"
# Leave it empty for measurement runs; debug logging is per-operation and will
# distort any throughput number taken with it on.
UCX_EXTRA="${ZENITH_UCX_EXTRA:-}"

# ---------------------------------------------------------------------------
# Why the ANP prepend STAYS for the UCX path, and what the real blocker was.
#
# run_container.sh:425-470 prepends ${ANP_DIR} to LD_LIBRARY_PATH so the
# RCCL-ANP plugin's vendor libionic -- which needs IBVERBS_PRIVATE_34 from
# rdma-core 50 -- resolves against ANP's shipped libibverbs instead of the
# container's rdma-core 61. That is load-bearing for multi-node RCCL
# (73 -> 930 Gbps).
#
# The first diagnosis here was that the prepend also broke UCX, because
# `ibv_devinfo -l` fails with it on:
#
#   ibv_devinfo: /opt/zenith/anp/libibverbs.so.1: version `IBVERBS_PRIVATE_59'
#                not found (required by /usr/lib/.../libibverbs.so.1)
#
# That symptom is real but it is the *tool* mixing stacks -- /usr/bin/ibv_devinfo
# is linked against rdma-core 61 and gets handed ANP's 50 -- not evidence that
# verbs itself is broken. Stripping ANP does make ibv_devinfo list all 8 HCAs,
# which is what made the wrong conclusion look confirmed.
#
# What settled it was testing the fabric directly rather than through a tool
# with its own linkage, on job 47017:
#
#   provider                         ibv_rc_pingpong   ibv_ud_pingpong
#   host, rdma-core 50 + libionic    5,467 Mbit/s      1,631 Mbit/s
#   container, rdma-core 61 in-tree  poll CQ failed -5 poll CQ failed -5
#   container, ANP 50 + libionic     5,537 Mbit/s      (UCX ud_verbs OK)
#
# So the in-tree `ionic` provider in rdma-core 61 cannot drive this Pollara HCA
# at all -- every completion queue poll returns -5 (EIO) regardless of UCX,
# regardless of GID index, for both RC and UD. The vendor libionic that ANP
# ships is not an optimisation here, it is the only working provider. ANP must
# therefore stay first on LD_LIBRARY_PATH, and IBV_CONFIG_DIR must keep pointing
# at ANP's provider list.
#
# The actual blocker was elsewhere, and there were two of them:
#
#   1. The container's Ubuntu libucx0 1.20.0+ds-4ubuntu2 ships NO IB transport
#      module. `dpkg -L libucx0` lists libuct_cma, libuct_rocm, libucm_rocm,
#      libucs_fuse and the perftest helpers -- there is no libuct_ib.so in the
#      image. UCX there is physically incapable of verbs and silently falls
#      back to TCP, which is what the ucp_worker log showed on every endpoint:
#        ucp_worker.c:1912 ... rma_am(tcp/enp105s0) amo_am(tcp/enp105s0)
#      and then loadRemoteMD() fails with a bare NIXL_ERR_BACKEND.
#
#   2. The host's UCX 1.22 at /opt/ucx does have libuct_ib.so, but is built
#      ENABLE_MT=0, and NIXL rejects it: "UCX library does not support
#      multi-threading".
#
# Hence ucx_build/build_ucx_mt.sh: UCX 1.20.0 (matching the container's ABI)
# configured --enable-mt --with-verbs, built inside the container. With that
# plus the ANP provider, NIXL's UCX backend comes up on ud_verbs across all 8
# ionic HCAs (agent metadata 671 bytes for 8 HCAs vs 370 for one).
# ---------------------------------------------------------------------------
CONTAINER_LD=$(srun --jobid="${JOBID}" --nodelist="${PREFILL_HOST}" \
    -N1 -n1 --overlap ${PODMAN} exec node printenv LD_LIBRARY_PATH \
    2>/dev/null | tr -d '\r' | tail -1)
if [ -z "${CONTAINER_LD}" ]; then
    echo "*** FATAL: could not read LD_LIBRARY_PATH from the container on ${PREFILL_HOST}" >&2
    exit 1
fi
LD_ENV=""

# ---------------------------------------------------------------------------
# ZENITH_DISAGG_TRANSPORT selects how NIXL's UCX backend moves KV bytes.
#
#   tcp  (default)  container's own UCX 1.20 over the 10.2.x cluster NIC.
#                   Needs no staged build. Despite being TCP/IP rather than
#                   RDMA, it MEASURED FASTER than the UD RDMA path here.
#
#   ud   MT-enabled UCX 1.20 built by ucx_build/build_ucx_mt.sh, ud_verbs over
#        the 8 ionic HCAs, on ANP's vendor libionic. Real RDMA, but slower --
#        kept because it is the only verbs path that works at all on this
#        fabric, so it is the thing to re-test if the driver gains RC support.
#
# MEASURED, nixl_xfer_test.py, 256 MiB GPU->GPU, a5 -> a6, job 47017:
#
#                          1 stream        4 streams (aggregate)
#   tcp / enp193s0f1np1    4.3 Gbps        17.2 Gbps
#   ud_verbs / ionic_0..3  2.8 Gbps        11.3 Gbps
#
# Both scale linearly with concurrency, which matters because vLLM runs one
# transfer per TP worker (8 per instance), so the per-stream number is not the
# ceiling. Neither is anywhere near the 930 Gbps RCCL gets, and the reason is
# not the wire: the same test with --host-mem (host memory instead of VRAM)
# gives 65.4 Gbps over TCP and 36.7 over UD. So the ~15x gap is the GPU staging
# path, not the network. GPUDirect RDMA is deliberately off on this fabric
# (run_container.sh documents the >=96 MiB hang), so every KV byte is copied
# VRAM -> host -> wire -> host -> VRAM. UCX_RNDV_FRAG_SIZE/FRAG_ALLOC_COUNT
# tuning, including adding an explicit rocm: key (the default has only host:
# and cuda: entries, so ROCm buffers silently take the 512K host default), moved
# this by 0.1 Gbps -- 4.3 -> 4.4. It is structural, not a tuning miss.
#
# Why UCX_NET_DEVICES is set explicitly rather than left at "all": with "all",
# UCX enumerates every interface including the public 216.128.x port and the
# docker bridges, and picks by its own scoring -- which is how it ended up on
# tcp/enp105s0, an interface that is not even the cluster network. The cluster
# fabric is enp193s0f1np1 (10.2.96.x/20); the RoCE HCAs are ionic_0..7.
#
# Why ud_verbs and not rc_verbs: the ionic HCA advertises max_srq=256 but
# rejects the call --
#     rc_iface.c:520 UCX ERROR ionic_0: ibv_create_srq() failed:
#                                       Operation not supported
# -- and UCX's rc_verbs calls ibv_create_srq() unconditionally
# (src/uct/ib/rc/base/rc_iface.c:510-526); there is no UCX_RC_VERBS_* knob to
# turn it off. This reproduces with ANP's provider, with rdma-core 61's, and at
# every GID index, so RC cannot open an iface on this fabric at all. ud_verbs
# contains zero references to srq and comes up cleanly. The cost is that UD is
# a datagram transport: no hardware RDMA-write, so UCX moves KV through its
# active-message path rather than zero-copy RMA. Expect well under the 930 Gbps
# the RCCL path gets, and measure it rather than assuming.
# ---------------------------------------------------------------------------
DISAGG_TRANSPORT="${ZENITH_DISAGG_TRANSPORT:-tcp}"
case "${DISAGG_TRANSPORT}" in
    tcp)
        UCX_ENV="--env UCX_TLS=tcp,self,sm,rocm_copy,rocm_ipc --env UCX_NET_DEVICES=${ZENITH_CLUSTER_NIC:-enp193s0f1np1}"
        echo "*** UCX transport: TCP over ${ZENITH_CLUSTER_NIC:-enp193s0f1np1} (correctness path, NOT RDMA)"
        ;;
    ud)
        UCX_MT="${ZENITH_UCX_MT_DIR:-/tmp/ucx_mt_install}"
        UCX_ENV="--env UCX_TLS=ud_verbs,self,sm,rocm_copy,rocm_ipc"
        UCX_ENV+=" --env UCX_NET_DEVICES=${ZENITH_UCX_HCAS:-ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1}"
        # GID 1 is the routable RoCE v2 address (GID 0 is link-local fe80::),
        # and is what NCCL_IB_GID_INDEX is already set to in the container.
        UCX_ENV+=" --env UCX_IB_GID_INDEX=${ZENITH_UCX_GID_INDEX:-1}"
        # Order matters twice over: ANP first so the vendor libionic provider
        # wins (rdma-core 61's in-tree ionic fails every CQ poll with -5), then
        # the MT UCX so libucp/libuct resolve to the build that has libuct_ib
        # and ENABLE_MT=1, then the container's own path.
        LD_ENV="--env LD_LIBRARY_PATH=/opt/zenith/anp:${UCX_MT}/lib:${CONTAINER_LD}"
        echo "*** UCX transport: ud_verbs over the ionic HCAs (RDMA), UCX from ${UCX_MT}"
        ;;
    *)
        echo "*** FATAL: ZENITH_DISAGG_TRANSPORT must be 'tcp' or 'ud', got '${DISAGG_TRANSPORT}'" >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Single-node disagg: both instances share one host, so they must be pinned to
# disjoint GPU sets or they will each try to claim all 8 and the second one
# OOMs during KV allocation.
#
#   ZENITH_PREFILL_GPUS=0,1,2,3 ZENITH_DECODE_GPUS=4,5,6,7 ZENITH_TP=4
#
# Both HIP_VISIBLE_DEVICES *and* CUDA_VISIBLE_DEVICES have to be set, to the
# same value. run_container.sh:521-523 sets both, and the ROCm runtime aborts
# outright if they disagree:
#     Conflicting visibility of agent-3 between HIP_VISIBLE_DEVICES and
#     CUDA_VISIBLE_DEVICES
# which kills the worker with exit 134 rather than falling back to either one.
#
# A model whose bf16 checkpoint does not fit in 4 x 288 GiB has to run its
# MXFP4 build here, with ZENITH_GPU_MEM_UTIL raised to match. TP (4 vs 8) is
# then a second moving variable against a multi-node xPyD run -- read it as
# node-count scaling at fixed quantization, not as a single-variable
# comparison.
#
# Naming, because the label misleads: "4P4D" here means 4 GPUs prefill / 4 GPUs
# decode -- ONE kv_producer and ONE kv_consumer, both TP=4. It is a 1P1D
# topology by instance count (see the "1P1D, 1 node(s)" banner below), so it
# has exactly one (prefill, decode) pair and a single warmup request suffices.
# It is NOT four producer instances and four consumer instances.
gpu_env() {
    local gpus="$1"
    [ -z "${gpus}" ] && return 0
    echo "--env HIP_VISIBLE_DEVICES=${gpus} --env CUDA_VISIBLE_DEVICES=${gpus}"
}

SERVE_PIDS=()
launch_instance() {
    local role="$1" host="$2" ip="$3" port="$4" sc_port="$5" idx="$6"
    local kv_cfg nixl_env
    if [ "${role}" = "prefill" ]; then
        kv_cfg="${KV_P}"; nixl_env="$(gpu_env "${ZENITH_PREFILL_GPUS:-}")"
    else
        kv_cfg="${KV_D}"; nixl_env="$(gpu_env "${ZENITH_DECODE_GPUS:-}")"
    fi
    nixl_env+=" ${UCX_ENV} ${LD_ENV} ${UCX_EXTRA}"
    nixl_env+=" --env VLLM_NIXL_SIDE_CHANNEL_HOST=${ip}"
    nixl_env+=" --env VLLM_NIXL_SIDE_CHANNEL_PORT=${sc_port}"
    srun --jobid="${JOBID}" --nodelist="${host}" -N1 -n1 --overlap \
        ${PODMAN} exec ${nixl_env} ${AITER_ENV} node \
        vllm serve "/app/models/${MODEL}" $(common_args) \
        --port "${port}" --kv-transfer-config "${kv_cfg}"${MTP_ARGS} \
        > "disagg_${role}${idx}.${JOBID}.log" 2>&1 &
    SERVE_PIDS+=($!)
}

for i in "${!PREFILL_HOSTS[@]}"; do
    launch_instance prefill "${PREFILL_HOSTS[$i]}" "${PREFILL_IPS[$i]}" \
        "${PREFILL_PORT}" "${SIDE_CHANNEL_PORT}" "${i}"
done
for i in "${!DECODE_HOSTS[@]}"; do
    launch_instance decode "${DECODE_HOSTS[$i]}" "${DECODE_IPS[$i]}" \
        "${DECODE_PORT}" "$(( SIDE_CHANNEL_PORT + 100 ))" "${i}"
done

# Every instance must be up before the proxy is useful -- the proxy round-robins
# blindly, so one lagging decoder does not degrade throughput, it 500s every
# Nth request. 90 x 30 s = 45 min matches ZENITH_STARTUP_TRIES; all instances
# load the checkpoint independently and in parallel, so this is not N x the
# single-node load time.
N_TOTAL=$(( N_PREFILL + N_DECODE ))
for i in $(seq 1 90); do
    UP=0
    for h in "${PREFILL_HOSTS[@]}"; do
        curl -sf -m 5 "http://${h}:${PREFILL_PORT}/v1/models" >/dev/null 2>&1 && UP=$(( UP + 1 ))
    done
    for h in "${DECODE_HOSTS[@]}"; do
        curl -sf -m 5 "http://${h}:${DECODE_PORT}/v1/models" >/dev/null 2>&1 && UP=$(( UP + 1 ))
    done
    if [ "${UP}" -eq "${N_TOTAL}" ]; then
        echo "*** all ${N_TOTAL} instances up after $((i*30))s"
        break
    fi
    for pid in "${SERVE_PIDS[@]}"; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "*** FATAL: an instance exited during startup; see disagg_*.${JOBID}.log"
            exit 1
        fi
    done
    sleep 30
done

if [ "${UP:-0}" -ne "${N_TOTAL}" ]; then
    echo "*** FATAL: only ${UP:-0}/${N_TOTAL} instances up after 45 min" >&2
    exit 1
fi

# The proxy is vLLM's own reference implementation, copied out of the image
# (tests/v1/kv_connector/nixl_integration/toy_proxy_server.py) rather than
# reimplemented -- it already handles the two-step "prefill with max_tokens=1,
# then hand the request to the decoder" dance and its streaming semantics.
#
# --prefiller-ports takes one port per host even though they are all identical
# (each instance owns a node), because the parser pairs the two lists by index
# and hard-errors on a length mismatch (toy_proxy_server.py:117-123).
P_PORTS=(); for _ in "${PREFILL_HOSTS[@]}"; do P_PORTS+=("${PREFILL_PORT}"); done
D_PORTS=(); for _ in "${DECODE_HOSTS[@]}"; do D_PORTS+=("${DECODE_PORT}"); done

echo "*** starting proxy on :${PROXY_PORT} (${N_PREFILL}P${N_DECODE}D)"
exec srun --jobid="${JOBID}" --nodelist="${PREFILL_HOST}" -N1 -n1 --overlap \
    ${PODMAN} exec node python /tmp/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
    --host 0.0.0.0 --port "${PROXY_PORT}" \
    --prefiller-hosts "${PREFILL_HOSTS[@]}" --prefiller-ports "${P_PORTS[@]}" \
    --decoder-hosts "${DECODE_HOSTS[@]}" --decoder-ports "${D_PORTS[@]}"
