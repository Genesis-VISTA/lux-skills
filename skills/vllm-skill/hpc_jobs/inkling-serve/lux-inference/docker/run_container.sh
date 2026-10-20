#!/bin/bash
#
# starts up ray within a container, spanning large multinode inference run
# ray communicates across the virtual cluster over ethernet,
# RCCL communicates over the high speed fabric
#
# this script is fabric-, arch- and scheduler-agnostic. It probes the host at
# run time and configures itself for whichever of the supported sites it finds:
#
#   site A  MI300A / Tuolumne : gfx942 APU, HPE Slingshot (/dev/cxi*, hsi0),
#                               4 GPUs per node, libfabric provider "cxi",
#                               XNACK=1 (host and device share HBM)
#   site B  MI355X / Lux      : gfx950 dGPU, AMD Pollara RoCE
#                               (/dev/infiniband/uverbs*, ionic_N),
#                               8 GPUs per node, libfabric provider "verbs"
#   site C  MI250X / RZVernal : gfx90a dGPU, HPE Slingshot (/dev/cxi*, hsi0),
#                               4 cards = 8 GCDs per node, provider "cxi"
#
# Nothing below is keyed to a site name: the fabric comes from which device
# nodes exist, the GPU count from KFD, and XNACK from the detected arch.
# Everything that used to be hardcoded for site A (the hsi0 / enp129s0
# interface names, the cxi device list, FI_PROVIDER=cxi, 4 GPUs) is derived.
#

# arg checking
if [ $# != 5 ]; then
    echo "Usage: $0 <head_ipaddr> --head|--worker </path/to/models> </path/to/datasets> <multistate>"
    exit 1
fi

# host addr, node type and directory holding our LLMs
HEAD_NODE_ADDRESS="$1"  # IP of the machine designated as the head (do NOT list worker machine IPs here)
NODE_TYPE="$2"          # should be --head or --worker
MODELPATH="$3"          # directory holding the model weights
DATASETSPATH="$4"	# directory holding the datasets for training
MULTISTATE="$5"		# whether numnodes=1 or >1

# arg checking on node type
if [ "${NODE_TYPE}" != "--head" ] && [ "${NODE_TYPE}" != "--worker" ]; then
    echo "Error: Node type must be --head or --worker"
    exit 1
fi

# arg checking on model path
if [ ! -d "${MODELPATH}" ]; then
    echo "Error: Model path directory ${MODELPATH} does not exist"
    exit 1
fi

# arg checking on datasets path
if [ ! -d "${DATASETSPATH}" ]; then
    echo "Error: Datasets path directory ${DATASETSPATH} does not exist"
    exit 1
fi

# whether it's a single or multinode run
if [ "${MULTISTATE}" != "single" ] && [ "${MULTISTATE}" != "multi" ]; then
    echo "Error: multistate must be either single or multi"
    exit 1
fi

# podman storage / runroot configuration for this site
SCRIPTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPTDIR}/lux_podman_env.sh"

# ---------------------------------------------------------------------------
# fabric + topology autodetection
# ---------------------------------------------------------------------------

# how many GPUs does this node actually have?
#   4 on MI300A, 8 on MI355X, 8 on a 4x MI250X node (each card is two GCDs and
#   KFD/rocm-smi/HIP all enumerate a GCD as its own device)
if command -v rocm-smi >/dev/null 2>&1; then
    NUM_GPUS=$(rocm-smi --showid --csv 2>/dev/null | grep -c '^card')
fi
if [ -z "${NUM_GPUS}" ] || [ "${NUM_GPUS}" -eq 0 ]; then
    NUM_GPUS=$(ls -d /sys/class/kfd/kfd/topology/nodes/*/ 2>/dev/null | \
               while read -r n; do
                   [ "$(cat "${n}/properties" 2>/dev/null | \
                        awk '/^simd_count/{print $2}')" != "0" ] && echo x
               done | wc -l)
fi
: "${NUM_GPUS:=8}"

# ZENITH_USE_GPUS caps how many of this node's GPUs the container may see, for
# sub-node experiments (e.g. "how does a model do on 4 GPUs instead of 8?").
# The allocation stays a whole exclusive node -- we are measuring GPU count, so
# sharing the node with another job would put its memory-bandwidth and power
# draw straight into the number. This restricts *visibility*, not ownership.
#
# It caps rather than sets: asking for more GPUs than the node has would hand
# HIP a device ordinal that does not exist, which surfaces as an invalid-device
# error from inside a worker rather than anything naming this variable.
if [ -n "${ZENITH_USE_GPUS:-}" ] && [ "${ZENITH_USE_GPUS}" -ge 1 ] 2>/dev/null; then
    if [ "${ZENITH_USE_GPUS}" -lt "${NUM_GPUS}" ]; then
        echo "  ZENITH_USE_GPUS : restricting ${NUM_GPUS} -> ${ZENITH_USE_GPUS} GPUs"
        NUM_GPUS=${ZENITH_USE_GPUS}
    elif [ "${ZENITH_USE_GPUS}" -gt "${NUM_GPUS}" ]; then
        echo "  ZENITH_USE_GPUS : ignoring ${ZENITH_USE_GPUS} (node has only ${NUM_GPUS})"
    fi
fi
GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))

# GPU architecture, used to decide XNACK and a few RCCL knobs, the default
# image tarball name, and the image/node arch check below. rocminfo is often
# not on PATH in a batch job (it comes from a module), so fall back to KFD:
# gfx_target_version is major*10000 + minor*100 + stepping, and the stepping
# is printed in hex (90010 -> gfx90a, 90402 -> gfx942, 90500 -> gfx950).
GPU_ARCH=$(rocminfo 2>/dev/null | awk '/gfx[0-9]/{print $2; exit}')
if [ -z "${GPU_ARCH}" ]; then
    GPU_ARCH=$(awk '/^gfx_target_version/ && $2 > 0 {
                   printf "gfx%d%d%x\n", int($2/10000), int($2/100)%100, $2%100; exit}' \
               /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null | head -1)
fi
: "${GPU_ARCH:=unknown}"

# --- high speed fabric ---
# sites A and C: HPE Slingshot presents /dev/cxi0.. and an hsi0 IP interface
# site B:        Pollara RoCE presents /dev/infiniband/uverbs0.. and ionic_N HCAs
CXI_DEVICES=$(ls /dev/cxi* 2>/dev/null)
IB_DEVICES=$(ls /sys/class/infiniband/ 2>/dev/null)

DEVICE_ARGS=""
if [ -n "${CXI_DEVICES}" ]; then
    FABRIC="slingshot"
    FI_PROV="cxi"
    for d in ${CXI_DEVICES}; do
        DEVICE_ARGS+=" --device ${d}"
    done
    # the data plane address lives on the slingshot interface
    MY_HSI_ADDRESS=$(ip -o -4 addr show hsi0 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
elif [ -n "${IB_DEVICES}" ]; then
    FABRIC="roce"
    FI_PROV="verbs"
    # pass the whole uverbs/rdma_cm set through
    DEVICE_ARGS+=" --device /dev/infiniband"
    # restrict RCCL to the HCAs that are actually ACTIVE
    ACTIVE_HCAS=""
    for d in ${IB_DEVICES}; do
        st=$(cat /sys/class/infiniband/${d}/ports/1/state 2>/dev/null)
        case "${st}" in *ACTIVE*) ACTIVE_HCAS+="${d}:1," ;; esac
    done
    ACTIVE_HCAS="${ACTIVE_HCAS%,}"
    # A node can carry ACTIVE RDMA NICs that are not the GPU fabric. On lux
    # (MI355X) the frontend NIC appears as bnxt_re0, which sorts ahead of
    # ionic_0..7: it landed in NCCL_IB_HCA, and the GID probe below ran on it
    # and chose index 3 (its IPv4 RoCE v2 slot), which is an *empty* slot on
    # ionic, so every QP failed to reach RTR. When any Pollara HCA (ionic_*) is
    # active, those are the GPU fabric, and RCCL-ANP drives nothing else, so
    # keep only them. Nodes without ionic HCAs keep every ACTIVE HCA, as before.
    # ZENITH_IB_HCA (a plain "dev:port,..." list) overrides the selection.
    IONIC_HCAS=""
    for h in ${ACTIVE_HCAS//,/ }; do
        case "${h}" in ionic_*) IONIC_HCAS+="${h}," ;; esac
    done
    [ -n "${IONIC_HCAS}" ] && ACTIVE_HCAS="${IONIC_HCAS%,}"
    ACTIVE_HCAS="${ZENITH_IB_HCA:-${ACTIVE_HCAS}}"
    # Derive the RoCE v2 GID index rather than hardcoding it. This used to be a
    # literal 3 with the comment "RoCEv2 GID", which is correct on
    # Mellanox/NVIDIA hardware and wrong here: on these AMD Pensando/Pollara
    # ionic NICs, index 3 is an *empty* slot, so every QP transition to RTR
    # fails. Reproducible outside NCCL entirely --
    #     ib_write_bw -d ionic_0 -x 3  ->  "Failed to modify QP 4096 to RTR"
    #     ib_write_bw -d ionic_0 -x 1  ->  38,609 MB/s (309 Gb/s)
    # The gid_attrs/types/<i> file names the protocol for each slot, so ask.
    # Prefer a global (non-fe80::) RoCE v2 GID over a link-local one, since the
    # link-local entry only works within a single L2 segment.
    FIRST_HCA="${ACTIVE_HCAS%%:*}"
    ROCE_GID_INDEX=""
    ROCE_GID_FALLBACK=""
    for i in 0 1 2 3 4 5 6 7; do
        gt="/sys/class/infiniband/${FIRST_HCA}/ports/1/gid_attrs/types/${i}"
        [ -r "${gt}" ] || continue
        case "$(cat "${gt}" 2>/dev/null)" in
            *"RoCE v2"*) ;;
            *) continue ;;
        esac
        gid=$(cat "/sys/class/infiniband/${FIRST_HCA}/ports/1/gids/${i}" 2>/dev/null)
        case "${gid}" in
            fe80:*) [ -z "${ROCE_GID_FALLBACK}" ] && ROCE_GID_FALLBACK="${i}" ;;
            *) ROCE_GID_INDEX="${i}"; break ;;
        esac
    done
    [ -z "${ROCE_GID_INDEX}" ] && ROCE_GID_INDEX="${ROCE_GID_FALLBACK}"
    if [ -z "${ROCE_GID_INDEX}" ]; then
        echo "WARNING: no RoCE v2 GID found on ${FIRST_HCA}; falling back to index 3"
        ROCE_GID_INDEX=3
    fi
    ROCE_GID_INDEX="${ZENITH_IB_GID_INDEX:-${ROCE_GID_INDEX}}"
    # NCCL_IB_GID_INDEX is one index for every HCA, so say so if it is not a
    # RoCE v2 slot on all of them (the probe above only looks at the first)
    for h in ${ACTIVE_HCAS//,/ }; do
        gt="/sys/class/infiniband/${h%%:*}/ports/1/gid_attrs/types/${ROCE_GID_INDEX}"
        [ -r "${gt}" ] || continue
        case "$(cat "${gt}" 2>/dev/null)" in
            *"RoCE v2"*) ;;
            *) echo "WARNING: RoCE GID index ${ROCE_GID_INDEX} is not RoCE v2 on ${h%%:*}" ;;
        esac
    done
else
    FABRIC="ethernet"
    FI_PROV="tcp"
    echo "WARNING: no Slingshot (/dev/cxi*) and no RDMA (/sys/class/infiniband)"
    echo "WARNING: falling back to TCP -- multinode performance will be poor"
fi

# --- control plane (ray) interface ---
# pick the highest-speed UP ethernet interface that has an IPv4 address and is
# not loopback/docker/virtual. On site A this lands on enp129s0, on site B it
# lands on the 10.x management/data interface. The Slingshot IP interfaces are
# excluded by name: hsi* on LLNL's TOSS, hsn* on HPE's own naming.
pick_eth () {
    local best="" bestspeed=-1
    for ifc in /sys/class/net/*; do
        n=$(basename "${ifc}")
        case "${n}" in lo|docker*|veth*|br-*|virbr*|hsi*|hsn*) continue ;; esac
        [ "$(cat ${ifc}/operstate 2>/dev/null)" = "up" ] || continue
        ip -o -4 addr show "${n}" >/dev/null 2>&1 || continue
        addr=$(ip -o -4 addr show "${n}" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)
        [ -n "${addr}" ] || continue
        sp=$(cat ${ifc}/speed 2>/dev/null); [ -n "${sp}" ] || sp=0
        # prefer RFC1918 (cluster-internal) over the routable/public interface
        case "${addr}" in 10.*|192.168.*|172.1[6-9].*|172.2[0-9].*|172.3[01].*) sp=$((sp + 1000000)) ;; esac
        if [ "${sp}" -gt "${bestspeed}" ]; then bestspeed=${sp}; best=${n}; fi
    done
    echo "${best}"
}
ETH_IFACE="${ZENITH_ETH_IFACE:-$(pick_eth)}"
if [ -z "${ETH_IFACE}" ]; then
    echo "Error: could not determine an ethernet interface for the ray control plane"
    exit 1
fi
MY_ETH_ADDRESS=$(ip -o -4 addr show "${ETH_IFACE}" | awk '{print $4}' | cut -d/ -f1 | head -1)
# on a slingshot-less site the data plane address is just the ethernet one
: "${MY_HSI_ADDRESS:=${MY_ETH_ADDRESS}}"

# --- per-arch GPU settings ---
# XNACK (unified/page-migrating memory) is right for the MI300A APU where the
# CPU and GPU share HBM. On a discrete card (MI355X gfx950, MI250X gfx90a) it
# forces the XNACK-enabled code object variant and costs performance for no
# benefit, so the default arm covers both of those.
case "${GPU_ARCH}" in
    gfx942*) XNACK=1 ;;
    *)       XNACK=0 ;;   # gfx950, gfx90a, anything else discrete
esac

MYHOSTNAME=$(hostname)
echo "=== zenith run_container.sh on ${MYHOSTNAME} ==="
echo "  gpu arch        : ${GPU_ARCH} (${NUM_GPUS} GPUs, HIP_VISIBLE_DEVICES=${GPU_LIST})"
echo "  fabric          : ${FABRIC} (libfabric provider ${FI_PROV})"
[ "${FABRIC}" = "roce" ] && echo "  active HCAs     : ${ACTIVE_HCAS}"
[ "${FABRIC}" = "roce" ] && echo "  RoCE GID index  : ${ROCE_GID_INDEX}"
echo "  ray interface   : ${ETH_IFACE} -> ${MY_ETH_ADDRESS}"
echo "  data plane addr : ${MY_HSI_ADDRESS}"
echo "  HSA_XNACK       : ${XNACK}"

# ---------------------------------------------------------------------------
# clean up any partial container left behind from a failed prior process
# ---------------------------------------------------------------------------
# Only this launcher's own named container is removed (never `podman rm -af`, which
# would also kill other users'/jobs' containers on a shared node). The name is
# "node" unless ZENITH_CONTAINER_PREFIX is set, then <prefix>-<hostname suffix>.
CNAME="${ZENITH_CONTAINER_PREFIX:+${ZENITH_CONTAINER_PREFIX}-$(hostname | sed "s/.*-//")}"; CNAME="${CNAME:-node}"
${PODMAN} rm -f ${CNAME} 2>/dev/null || true
${PODMAN} stop ${CNAME} 2>/dev/null || true

# define a function to cleanup on EXIT signal
cleanup() {
    ${PODMAN} stop ${CNAME} || true
    ${PODMAN} rm --force ${CNAME} || true
}
trap cleanup EXIT

# command setup for head or worker node
# libamd_smi belt-and-braces (see the matching comment in the Dockerfile).
# The image now ships $ROCM_HOME/lib/libamd_smi.so pointing at the _rocm_sdk_core
# runtime copy, but repair it here too so an older image still serves: if the
# unversioned name resolves into _rocm_sdk_devel, any process that imports both
# torch and amdsmi maps two copies of the library, the second amdsmi_init()
# enumerates zero devices, and torch.cuda.device_count() drops to 0 -- vLLM then
# aborts with "Failed to infer device type".
SMI_FIXUP='
ROCMLIB="$(ls -d /opt/rocm*/lib 2>/dev/null | head -1)"
CORESMI="$(ls /opt/venv/lib/python*/site-packages/_rocm_sdk_core/lib/libamd_smi.so.* 2>/dev/null | head -1)"
if [ -n "$ROCMLIB" ] && [ -n "$CORESMI" ] && \
   ! readlink -f "$ROCMLIB/libamd_smi.so" 2>/dev/null | grep -q _rocm_sdk_core; then
    ln -sf "$CORESMI" "$ROCMLIB/libamd_smi.so" && \
      echo "lux: repointed libamd_smi.so -> $CORESMI"
fi
'

# kpack shadowing (see the matching comment in the Dockerfile). The kpack
# loader takes the most specific torch_<arch>[:<features>].kpack that exists,
# not one that holds the kernel; the gfx90a wheel's :xnack+/- archives hold only
# libtorch_rocshmem.so, so on an MI250X every torch kernel launch fails with
# hipErrorInvalidImage. Move such archives aside when a generic one exists, for
# images built before the Dockerfile dropped them; a no-op otherwise.
KPACK_FIXUP='
for k in /opt/venv/lib/python*/site-packages/*/.kpack/*:*.kpack; do
    [ -e "$k" ] && [ -e "${k%%:*}.kpack" ] || continue
    mkdir -p /tmp/kpack_shadowed && mv "$k" /tmp/kpack_shadowed/ && \
      echo "lux: moved shadowing kpack $(basename "$k") aside"
done
'

# Same belt-and-braces for the vLLM ROCm source patches in patches/. The image
# applies them at build time, but apply them here too so an image built before a
# given patch existed still serves correctly -- ${SCRIPTDIR}/patches is bind
# mounted over /opt/zenith_patches below, so this always runs the current
# versions. Every patch script is idempotent and exits 0 when already applied,
# so this whole block is a no-op on an up-to-date image.
#
# Currently (the *.py files in patches/; the Dockerfile keeps the same list).
# On gfx950 the MI250X ones listed in GFX950_SKIP_PATCHES after this list are
# skipped unless ZENITH_GFX950_ALL_PATCHES=1:
#   rocm_aiter_ray_env_refresh.py  AITER toggles are cached at import time, so
#                                a Ray actor never sees the driver's
#                                VLLM_ROCM_USE_AITER=1 and the MoE layer dies
#                                with "AITER MoE is not enabled". Multinode
#                                only -- single-node mp runs are unaffected.
#   spec_decode_drafter_pp_guard.py  vLLM builds the spec-decode drafter only
#                                on the last PP rank, then reads self.drafter
#                                back on all of them. With MTP (now the
#                                default) and PP>1, every other rank dies at
#                                startup with "'GPUModelRunner' object has no
#                                attribute 'drafter'". PP>1 only -- at PP=1
#                                every rank is the last rank and the guards
#                                evaluate exactly as before.
#   spec_decode_pp_broadcast_width.py  With PP + async scheduling + MTP, the
#                                last stage's sampled-token broadcast asserts
#                                width 1 and the engine dies on the first
#                                decode step. Broadcasts the last *accepted*
#                                token per request (simply widening the
#                                broadcast would feed earlier stages the wrong
#                                token and silently corrupt output).
#   moe_backend_aiter_mxfp4_activation_aliases.py  Adds aiter_mxfp4_mxfp4 to
#                                the --kernel-config moe_backend choices, so
#                                an MXFP4 MoE can be asked to run W4A4;
#                                without it argparse rejects the value.
#   nixl_block_major_alignment_slack.py  NIXL's block-major KV check is an
#                                exact size equality, which fails when the
#                                allocator rounds the pool up for alignment
#                                (a few KiB). Disaggregated prefill dies at KV
#                                registration without it.
#   inkling_split_kv_empty_row_guard.py  With CUDA graphs on, a padded decode
#                                slot (cache_seqlens 0) makes Inkling's split-KV
#                                stage 2 store 0/0 = NaN, and fused_moe_kernel
#                                then faults on a host address. Needs 3+
#                                concurrent requests and a KV length that
#                                selects split-KV; --enforce-eager never pads.
#   inkling_attn_output_zero_init.py  The same padded slot under the
#                                single-pass decode kernel is never written,
#                                so it keeps torch.empty graph-pool garbage;
#                                NaN there faults fused_moe_kernel the same
#                                way (7 concurrent on Tioga). Needs both.
#   inkling_mtp_pp_embed_share.py  With PP>1 the V2 runner's
#                                maybe_share_target_embed skips InklingMTP,
#                                so its embed_tokens stays None and the last
#                                PP rank dies at startup ("'NoneType' object
#                                has no attribute 'weight'"). PP>1 + MTP only.
#   pp_broadcast_deferred_recv.py  V2 runner, PP>1: earlier PP ranks posted
#                                the sampled-token recv right after their
#                                forward. On gfx90a the spinning RCCL kernel
#                                slowed the next forward to ~40%, the stages
#                                serialized, and CUDA graphs lost to eager
#                                from c=4 up. Posts the recv when it is
#                                consumed instead. PP=1 has no PPHandler.
#   inkling_single_pass_kv_bounds.py  Inkling's single-pass attention (every
#                                MTP verify step) walked the whole context on
#                                512-token sliding-window layers; ~12x on
#                                those layers at 8k context. Bit-identical.
#   inkling_verify_split_kv.py   MTP verify on global layers used one
#                                workgroup per (request, KV head) over the
#                                whole context; now split-KV + merge, 655 ->
#                                46 us at B=1. INKLING_VERIFY_SPLIT_KV=0 off.
#   inkling_gate_fp32_splitk.py  Router fp32-out GEMM at T <= 64 via split-K
#                                Triton instead of hipBLASLt (58-206 -> 8-16
#                                us). On by default only on gfx90a;
#                                INKLING_GATE_SPLITK=1/0 forces it.
#   inkling_sconv_rsag_blocks.py Inkling's sconv RS/AG on dim-0 slabs
#                                ((T,tp,C) -> (tp,T,C)) instead of dim -1:
#                                row copies replace three strided element
#                                transposes per site (286 -> 226 us at T=512).
#   inkling_sconv_zero_copy.py   Drops most remaining copies around the
#                                sconv RS/AG and residual add + RMSNorm
#                                (13-18% per site). INKLING_SCONV_ZC=0 off.
#   inkling_sconv_zero_copy_ca.py  Sconv RS/AG via the custom all-reduce for
#                                T <= INKLING_SCONV_CA_RSAG_MAX_T (default 0
#                                = off). Needs VLLM_ROCM_CUSTOM_AR_GFX90A=1.
#   inkling_draft_tp_argmax.py   MTP drafter argmax over vocab shards, no
#                                full-logits gather. Bit-exact.
#                                INKLING_DRAFT_TP_ARGMAX=0 off.
#   inkling_mtp_input_proj_tp.py MTP drafter input projection split across
#                                TP ranks by tokens for T >=
#                                INKLING_MTP_INPUT_PROJ_TP (default 512).
#   vllm_custom_ar_gfx90a.py     VLLM_ROCM_CUSTOM_AR_GFX90A=1 (default 0)
#                                lets vLLM's ROCm custom all-reduce run on
#                                MI250X (upstream gates it to gfx94/95).
#   inkling_moe_sink_fused_sum.py  Prefill-size sink-expert add folded into
#                                the routed moe_sum. Bit-exact. Needs
#                                rocm_triton_moe_sum.py.
#   rocm_triton_moe_output_alias.py  ROCm TritonExperts writes into the
#                                caller's output, skipping an [M, H] copy.
#   rocm_triton_moe_sum.py       Triton moe_sum for M >= 1024 on ROCm, with
#                                an optional fused addend. Bit-exact.
#   rocm_triton_moe_w2_config.py Per-entry "w2" GEMM2 tiles / A-stationary
#                                GEMM2 from the tuned MoE config; entries
#                                without "w2" are unchanged.
#   rocm_triton_moe_g1act.py     Per-entry "g1act" overlay: GEMM1 with
#                                SiLU-and-mul in the epilogue. Bit-exact;
#                                entries without "g1act" are unchanged.
#   inkling_gate_select_rows.py  Inkling router top-k, row-blocked on one
#                                wavefront (211 -> 37 us at T=8192).
#                                Bit-identical.
#   inkling_mixed_flat_grid.py   Inkling attention for mixed prefill +
#                                decode steps on a flat grid of the
#                                (request, query block) pairs that exist.
#                                Bit-identical.
#   inkling_prefill_attn_bq16.py Inkling prefill attention (chunks of 2048+
#                                queries) with 16-query programs: global
#                                layer 3.0 -> 1.2 ms per 7.6k prompt.
#                                Bit-identical.
#   inkling_compute_logits_local.py  Inkling compute_logits_local, so
#                                --enable-batch-sharded-sampling runs.
#                                Inert without the flag.
#   zz_hl_overlay.py             Copies the two tuned files in
#                                patches/hl_overlay/vllm over the installed
#                                vllm (AITER IPC all-gather for every TP
#                                all-gather size, and the guarded RAHS Inkling
#                                attention). gfx950 only, and fatal there if it
#                                does not apply, so a run never serves the
#                                untuned kernels by accident; on gfx942/gfx90a
#                                it is skipped. Runs last (zz_).
#
# On gfx950 the loop skips the MI250X patches in GFX950_SKIP_PATCHES below.
# They were written and measured on tioga and have never run on MI355X; they
# change Inkling's MI355X code paths without an MI355X test.
# ZENITH_GFX950_ALL_PATCHES=1 applies them anyway. Three of them
# (inkling_mixed_flat_grid, inkling_prefill_attn_bq16, rocm_inkling_rel_attn_dec)
# edit fa4_rel_attention.py, one of the overlay's two files. The overlay's
# fa4_rel_attention.py is the Hyperloom rewrite merged onto the file with those
# three applied: on gfx950 it runs the rewrite, on other archs the three
# patches' kernels. hl_overlay/BASE.sha256 lists both exports it accepts (the
# zenith_v6 image's file and the v8 gfx950 image's, which bakes the three in).
#
# ZENITH_HL_OVERLAY picks what zz_hl_overlay.py does on gfx950: 1 (default)
# copies the overlay and makes any failure fatal; 0 skips it (stock kernels);
# partial copies the rewrites whose installed file still matches an export and
# prints a SKIP line for the others instead of failing (for an image whose
# overlay files match neither export).
ZENITH_HL_OVERLAY="${ZENITH_HL_OVERLAY:-1}"
case "${ZENITH_HL_OVERLAY}" in
    0|1|partial) ;;
    *) echo "lux: ZENITH_HL_OVERLAY must be 1, 0 or partial (got '${ZENITH_HL_OVERLAY}')" >&2; exit 1 ;;
esac
GFX950_SKIP_PATCHES="inkling_compute_logits_local inkling_draft_tp_argmax
    inkling_gate_select_rows inkling_gate_token_parallel inkling_mixed_flat_grid
    inkling_moe_sink_fused_sum inkling_mtp_input_proj_tp inkling_prefill_attn_bq16
    inkling_rel_proj_mfma16 inkling_sconv_zero_copy inkling_sconv_zero_copy_ca
    inkling_sconv_zero_copy_sres inkling_sconv_zero_copy_xrsag
    pp_balanced_decode_split rocm_inkling_decode_tgemm rocm_inkling_dense_rocblas
    rocm_inkling_rel_attn_dec rocm_moe_align_triton rocm_triton_moe_g1act
    rocm_triton_moe_output_alias rocm_triton_moe_sum rocm_triton_moe_w2_config
    rocm_triton_moe_w2_sum_prefill rocm_triton_moe_zdec_join vllm_custom_ar_gfx90a"
PATCH_FIXUP='
for _p in /opt/zenith_patches/*.py; do
    [ -f "$_p" ] || continue
    if [ "'"${GPU_ARCH}"'" = gfx950 ] && [ "'"${ZENITH_GFX950_ALL_PATCHES:-0}"'" != 1 ]; then
        _n="${_p##*/}"
        case " '"$(echo ${GFX950_SKIP_PATCHES})"' " in
            *" ${_n%.py} "*)
                echo "lux: skipping $_p (MI250X patch, not validated on gfx950; ZENITH_GFX950_ALL_PATCHES=1 applies it)"
                continue ;;
        esac
    fi
    if [ "${_p##*/}" = zz_hl_overlay.py ]; then
        if [ "'"${GPU_ARCH}"'" != gfx950 ]; then
            echo "lux: skipping $_p (tuned for gfx950; this node is '"${GPU_ARCH}"')"
            continue
        fi
        if [ "'"${ZENITH_HL_OVERLAY}"'" = 0 ]; then
            echo "lux: skipping $_p (ZENITH_HL_OVERLAY=0; serving the untuned kernels)"
            continue
        fi
        ZENITH_HL_OVERLAY='"${ZENITH_HL_OVERLAY}"' python "$_p" || { echo "lux: FATAL -- $_p failed; not serving"; exit 1; }
        continue
    fi
    python "$_p" || echo "lux: WARNING -- patch $_p did not apply"
done
unset _p _n
'

# Install the nixl_rocm wheel if one has been staged on the host, so the NIXL
# KV connector (disaggregated prefill) is usable.
#
# vLLM ships the NIXL connector in the image and it imports fine, but every code
# path that touches it dies at base_worker.py:519 "RuntimeError: NIXL is not
# available", because vllm/distributed/nixl_utils.py dispatches on platform:
#
#     return "nixl_rocm" if current_platform.is_rocm() else "nixl"
#
# and there is no nixl_rocm anywhere -- not in the image, and not on PyPI. The
# PyPI "nixl" is a CUDA meta-package (depends on nixl-cu12/nixl-cu13), and
# "nixl-rocm" 404s. The only source is a source build of ai-dynamo/nixl with
# -Dwheel_variant=rocm; build_nixl_rocm.sh does that inside this same image and
# drops the wheel in nixl_build/. Staging a host-built artifact into the running
# container is the same pattern already used for librccl-anp.so above, which is
# why this does not require an image rebuild.
#
# Note the wheel's *distribution* name is nixl_cu12 even for the ROCm build --
# -Dwheel_variant renames the importable module dir (nixl_rocm/) but not the
# project name in pyproject.toml. The filename is misleading; the contents are
# ROCm/HIP, built against /opt/rocm-7.14.0 with hipify-perl, and the agent
# reports plugins ['POSIX', 'UCX'].
#
# Absent-by-default: no wheel staged means this is a silent no-op and NIXL stays
# unavailable exactly as before. --no-deps is mandatory -- the wheel's declared
# deps would drag CUDA torch over the ROCm one.
NIXL_FIXUP='
for _w in /opt/zenith_nixl/*.whl; do
    [ -f "$_w" ] || continue
    python -c "import nixl_rocm" 2>/dev/null && break
    pip install --root-user-action=ignore --no-cache-dir --no-deps --quiet "$_w" \
        && echo "lux: installed $(basename "$_w") (nixl_rocm)" \
        || echo "lux: WARNING -- nixl wheel $_w did not install"
    break
done
unset _w
'
# Only mount the directory if it exists and holds a wheel; podman errors on a
# missing source path, and creating an empty dir on every node just to satisfy
# the mount would be worse than a conditional.
NIXL_MOUNT=""
if compgen -G "${SCRIPTDIR}/nixl_build/*.whl" >/dev/null 2>&1; then
    NIXL_MOUNT="-v ${SCRIPTDIR}/nixl_build:/opt/zenith_nixl:ro"
else
    NIXL_FIXUP=""
fi

RAY_START_CMD="ray start --block --disable-usage-stats --log-style=pretty --object-store-memory 10000000000"
# leave a couple of cores for the raylet/GCS itself. The image caps the idle
# worker pool these CPUs would otherwise prestart (RAY_prestart_worker_first_driver,
# RAY_num_workers_soft_limit; see the Dockerfile's runtime defaults).
RAY_CPUS=$(( $(nproc) - 8 )); [ ${RAY_CPUS} -lt 1 ] && RAY_CPUS=1
if [ "${NODE_TYPE}" == "--head" ]; then
    # both single and multi node: the head participates with all of its GPUs.
    # (as of ray 2.44+ the head's own memory usage is no longer excessive)
    RAY_START_CMD+=" --head --node-ip-address=${MY_ETH_ADDRESS} --port=6379"
    RAY_START_CMD+=" --num-cpus=${RAY_CPUS} --num-gpus=${NUM_GPUS}"
else
    RAY_START_CMD+=" --address=${HEAD_NODE_ADDRESS}:6379 --node-ip-address=${MY_ETH_ADDRESS}"
    RAY_START_CMD+=" --num-cpus=${RAY_CPUS} --num-gpus=${NUM_GPUS}"
fi

# container image: prefer a pre-loaded image in local storage.
# ZENITH_IMAGE_TAR selects a candidate build (build_image.sh with VLLM_HASH set
# writes inference_image_vllm<sha8>.tar) for an A/B without swapping production.
# ZENITH_IMAGE_TAG=<tag> (as given to build_image.sh) selects inference_image_<arch>_<tag>.tar.
# The default is named for this node's GPU arch (build_image.sh writes
# inference_image_<arch>[_<tag>].tar); a tree with only a tarball from before
# that naming falls back to the plain inference_image[_<tag>].tar.
IMAGE_TAR_TAG=""
[ -n "${ZENITH_IMAGE_TAG:-}" ] && [ "${ZENITH_IMAGE_TAG}" != "latest" ] && IMAGE_TAR_TAG="_${ZENITH_IMAGE_TAG}"
IMAGE_TAR="${SCRIPTDIR}/inference_image_${GPU_ARCH%%:*}${IMAGE_TAR_TAG}.tar"
[ -e "${IMAGE_TAR}" ] || IMAGE_TAR="${SCRIPTDIR}/inference_image${IMAGE_TAR_TAG}.tar"
IMAGE_TAR="${ZENITH_IMAGE_TAR:-${IMAGE_TAR}}"
IMAGE_REF="${ZENITH_IMAGE_REF:-docker-archive:${IMAGE_TAR}}"
echo "  image           : ${IMAGE_REF}"

# Image arch vs node arch. Each image is compiled for exactly one GPU arch, and
# the wrong one does not fail at load or import: HIP only reports "no kernel
# image is available" (or hangs a collective) at the first kernel launch, after
# Ray and the health-check setup. Two checks, cheapest first:
#   1. an arch-named tarball (inference_image_gfxNNN*.tar) that names another
#      arch -- before podman spends minutes loading ~61 GB;
#   2. inside the container, the image's own PYTORCH_ROCM_ARCH (set from
#      TRAIN_ARCHS at build time), which also covers the plain-named tarballs
#      and ZENITH_IMAGE_REF. Folded into the container command (ARCH_CHECK).
# ZENITH_SKIP_ARCH_CHECK=1 turns both off; an undetected node arch skips them.
NODE_ARCH="${GPU_ARCH%%:*}"
ARCH_CHECK=""
if [ "${ZENITH_SKIP_ARCH_CHECK:-0}" != "1" ] && [ "${NODE_ARCH}" != "unknown" ]; then
    TAR_ARCH=$(basename "${IMAGE_REF#docker-archive:}" | grep -o -E '^inference_image_gfx[0-9a-f]+' | sed 's/^inference_image_//')
    if [ -n "${TAR_ARCH}" ] && [ "${TAR_ARCH}" != "${NODE_ARCH}" ]; then
        echo "Error: image ${IMAGE_REF} is built for ${TAR_ARCH}, but this node's GPUs are ${NODE_ARCH}."
        echo "Error: build one with ./build_image.sh ${NODE_ARCH}, or point ZENITH_IMAGE_TAR at it."
        echo "Error: (ZENITH_SKIP_ARCH_CHECK=1 overrides this check.)"
        exit 1
    fi
    ARCH_CHECK='
case ";${PYTORCH_ROCM_ARCH:-};" in
    *";'"${NODE_ARCH}"';"*|";;") ;;
    *) echo "lux: ERROR image is built for PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}, but this node is '"${NODE_ARCH}"'."
       echo "lux: ERROR build the image for '"${NODE_ARCH}"' (./build_image.sh '"${NODE_ARCH}"'), or set ZENITH_SKIP_ARCH_CHECK=1."
       exit 1 ;;
esac
'
fi

# Image vs this script's environment. The static container environment (RCCL,
# torch and Ray settings, cache dirs, MAX_JOBS, LD_PRELOAD) is baked into the
# image, not passed below. An image built before that (no ZENITH_IMAGE_ENV)
# would start without it -- and with the old image's TORCH_NCCL_BLOCKING_WAIT=1
# and Ray's worker prestart storm -- so refuse it; rebuild the image.
ENV_CHECK='
if [ -z "${ZENITH_IMAGE_ENV:-}" ]; then
    echo "lux: ERROR this image predates the baked-in runtime environment (no ZENITH_IMAGE_ENV)."
    echo "lux: ERROR rebuild it (./build_image.sh <arch>)."
    exit 1
fi
'

# ---------------------------------------------------------------------------
# environment passed into the container
#
# Only what has to be worked out on the node: one shared block, then one block
# per fabric. Everything static is an image default (Dockerfile, "runtime
# defaults" and the RCCL / libfabric sections); the ZENITH_* knobs below
# override an image default only when they are set.
# ---------------------------------------------------------------------------
# shared, every arch and fabric
CONTAINER_ENV="--env RAY_ADDRESS_HOST=${MY_ETH_ADDRESS}"
CONTAINER_ENV+=" --env HIP_VISIBLE_DEVICES=${GPU_LIST}"
CONTAINER_ENV+=" --env ROCR_VISIBLE_DEVICES=${GPU_LIST}"
CONTAINER_ENV+=" --env CUDA_VISIBLE_DEVICES=${GPU_LIST}"
CONTAINER_ENV+=" --env HSA_XNACK=${XNACK}"             # per arch, see XNACK above
CONTAINER_ENV+=" --env FI_PROVIDER=${FI_PROV}"         # per fabric, probed above
CONTAINER_ENV+=" --env NCCL_SOCKET_IFNAME=${ETH_IFACE}"
CONTAINER_ENV+=" --env GLOO_SOCKET_IFNAME=${ETH_IFACE}"
# Ray's idle worker pool cap (image default 16; MI300A_PORTABILITY §2.1 bug 5)
[ -n "${ZENITH_RAY_MAX_IDLE_WORKERS:-}" ] && CONTAINER_ENV+=" --env RAY_num_workers_soft_limit=${ZENITH_RAY_MAX_IDLE_WORKERS}"
# 0 makes PATCH_FIXUP undo patches/pp_broadcast_deferred_recv.py (needed for
# MTP + PP on Lux; set by lux_tuned_defaults.sh, see that patch)
[ -n "${ZENITH_PP_DEFERRED_RECV:-}" ] && CONTAINER_ENV+=" --env ZENITH_PP_DEFERRED_RECV=${ZENITH_PP_DEFERRED_RECV}"
[ -n "${ZENITH_GFX950_ALL_PATCHES:-}" ] && CONTAINER_ENV+=" --env ZENITH_GFX950_ALL_PATCHES=${ZENITH_GFX950_ALL_PATCHES}"
CONTAINER_ENV+=" --env ZENITH_HL_OVERLAY=${ZENITH_HL_OVERLAY}"
# ZENITH_NCCL_DMABUF=0: register GPU buffers through peer-memory, not dma-buf
[ -n "${ZENITH_NCCL_DMABUF:-}" ] && CONTAINER_ENV+=" --env NCCL_DMABUF_ENABLE=${ZENITH_NCCL_DMABUF}"
# AITER JIT ninja fan-out ceiling (image default 32; see the Dockerfile)
[ -n "${ZENITH_MAX_JOBS:-}" ] && CONTAINER_ENV+=" --env MAX_JOBS=${ZENITH_MAX_JOBS}"

# per fabric: roce (Lux, MI355X), slingshot (tuolumne MI300A, tioga MI250X),
# and plain ethernet as the fallback.
#
# Where AMD's RCCL-ANP net plugin and the libionic it needs live inside the
# container. The default is the copy this package ships in patches/anp (built
# on Lux against the hosts' libionic-amd 61; see patches/anp/README.md), which
# reaches the container through the /opt/zenith_patches mount below. Older
# images staged their own at /opt/zenith/anp; ZENITH_ANP_DIR=/opt/zenith/anp
# selects that one.
ANP_DIR="${ZENITH_ANP_DIR:-/opt/zenith_patches/anp}"
# The image's own LD_LIBRARY_PATH, which ANP_DIR is prepended to rather than
# replacing -- torch, rocm, and the sdk wheels all have to stay reachable.
CONTAINER_LD_PATH="/opt/venv/lib/python3.14/site-packages/torch/lib:/opt/aws-ofi-rccl/lib:/opt/libfabric/lib:/opt/libcxi/lib:/opt/rocm/lib:/opt/venv/lib/python3.14/site-packages/_rocm_sdk_core/lib:/opt/venv/lib/python3.14/site-packages/_rocm_sdk_libraries/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"

FABRIC_ENV=""
if [ "${FABRIC}" = "roce" ]; then
    # ---- RoCE: RCCL over the Pollara HCAs ----
    FABRIC_ENV+=" --env NCCL_IB_DISABLE=0"
    FABRIC_ENV+=" --env NCCL_IB_HCA=${ACTIVE_HCAS}"
    FABRIC_ENV+=" --env NCCL_IB_GID_INDEX=${ROCE_GID_INDEX}"  # probed above, not 3
    # GPUDirect RDMA is ON by default (the image's NCCL_NET_GDR_LEVEL=3, with
    # the plugin's GDR flush disabled here); ZENITH_NET_GDR_LEVEL=0 turns it
    # off (host-staged RDMA). This
    # whole block is cross-node only: on one node RCCL uses P2P/IPC, not NET.
    #
    # Root cause of the GDR hang (2026-10-04, 2 x 8 MI355X): after each GDR
    # receive on the Simple protocol, RCCL asks the net plugin to flush, and
    # RCCL-ANP (amd-anp v1.4.0, net_ib.cc anpNetFlush) does that with an RDMA
    # READ on a loopback "flush" QP. On the ionic (Pollara AINIC) NICs that READ
    # never completes, so the receiver waits forever. LL/LL128 do not flush,
    # which is why small all-reduces worked and the hang showed up at 96 MiB
    # (where the tuner switches to Simple) and at every send/recv size (p2p is
    # always Simple). It is not the ROCm 7.2.2-plugin / 7.14-image mismatch:
    # the fully matched host stack (ROCm 7.2.2 RCCL + ANP + rccl-tests) hangs
    # identically. RCCL's own IB transport already disables the GDR flush by
    # default for AINIC; ANP does not, so any non-zero ZENITH_NET_GDR_LEVEL does
    # it here.
    #
    # With GDR level 3 + flush off, 16 ranks, zenith_v6 image (host-staged in
    # brackets):
    #   all_reduce 1 GiB busbw 2,972 Gbps (971)                    3.06x
    #   send/recv per GPU, 8 cross-node pairs: 383 Gbps (110)      3.5x
    #   10 min whole-tensor integrity stress (all_reduce, batch and blocking
    #   send/recv, 4 KiB-1 GiB): 0 bad of 1,578,272 checks.
    # NCCL_NCHANNELS_PER_NET_PEER=4 is what takes send/recv from ~220 to ~380
    # Gbps per GPU (8 gives the same; all-reduce unaffected).
    # Default since 2026-10-04 after: a 48 min random-data stress (0 bad of
    # 10.05 M whole-tensor checks, no hangs), 17 fresh health-check boots at
    # 2,727-2,849 Gbps, all channels GDRDMA, and Inkling bf16 + MXFP4 TP8 x PP2
    # 2-node serving (30/30, gsm8k within noise, throughput neutral). One health
    # check of 24 read 109.6 Gbps and was not reproduced.
    # Do NOT add NCCL_IB_QPS_PER_CONNECTION=4 + NCCL_IB_SPLIT_DATA_ON_QPS=1:
    # that hangs at 1 MiB.
    ROCE_GDR_LEVEL="${ZENITH_NET_GDR_LEVEL:-3}"
    [ -n "${ZENITH_NET_GDR_LEVEL:-}" ] && FABRIC_ENV+=" --env NCCL_NET_GDR_LEVEL=${ROCE_GDR_LEVEL}"
    case "${ROCE_GDR_LEVEL}" in
        0|LOC) ;;
        *)  FABRIC_ENV+=" --env NCCL_GDR_FLUSH_DISABLE=1"
            FABRIC_ENV+=" --env NCCL_NCHANNELS_PER_NET_PEER=${ZENITH_NCHANNELS_PER_NET_PEER:-4}" ;;
    esac
    # Do NOT swap in the host's rdma-core libionic (the cross-node NIXL recipe
    # does that, patches/nixl): ANP then finds no device and RCCL falls back
    # to NET/Socket (~75 Gbps).
    #
    # the cray tuning profile is slingshot-only; do not set OFI_NCCL_PLATFORM
    #
    # Bypass the aws-ofi-nccl plugin entirely and use RCCL's own IB transport.
    # On this fabric libfabric selects verbs;ofi_rxm, and the plugin then fails
    # endpoint allocation with EOPNOTSUPP ("Couldn't allocate endpoint. RC:
    # -95"), after which RCCL silently degrades to NET/Socket -- i.e. TCP. With
    # the plugin out of the way RCCL logs "Assigned NET plugin IB to comm" and
    # carries real RDMA. Measured 2026-09-23; see RESULTS.md.
    #
    # Set ZENITH_ROCE_USE_OFI=1 to go back to the plugin (e.g. to retest after
    # an aws-ofi-nccl bump).
    #
    # Dropping the OFI plugin gets us onto RCCL's generic IB transport, which is
    # necessary but NOT sufficient: generic IB completes exactly one collective
    # per communicator and then dies with `ibv_poll_cq() -> -5` from the proxy
    # thread, at any message size. These NICs need AMD's own RCCL net plugin,
    # librccl-anp.so ("accelerated network path"), which the ROCm 7.2.2 hosts
    # ship and the container base image does not. Staged into the image from
    # /opt/rocm-7.2.2 at build time, together with the rdma-core 50 libionic it
    # is linked against -- the vendor libionic needs IBVERBS_PRIVATE_34, so it
    # cannot load against the container's own rdma-core 61.
    #
    # With ANP: 10/10 repeated all-reduces, all sizes 4 KiB..64 MiB, and
    # 1,048.6 Gbps at 64 MiB vs 73 Gbps on the TCP fallback. See RESULTS.md,
    # "the second collective, at any size".
    #
    # Set ZENITH_ROCE_USE_ANP=0 to fall back to generic IB (single-collective
    # only -- useful for reproducing the bug, not for running work).
    if [ "${ZENITH_ROCE_USE_OFI:-0}" = "1" ]; then
        FABRIC_ENV+=" --env OFI_NCCL_PROTOCOL=${ZENITH_OFI_PROTOCOL:-SENDRECV}"
    elif [ "${ZENITH_ROCE_USE_ANP:-1}" = "1" ] && [ -n "${ANP_DIR}" ]; then
        FABRIC_ENV+=" --env NCCL_NET_PLUGIN=${ANP_DIR}/librccl-anp.so"
        FABRIC_ENV+=" --env NCCL_TUNER_PLUGIN=none"
        FABRIC_ENV+=" --env IBV_CONFIG_DIR=${ANP_DIR}/libibverbs.d"
        # ANP_DIR must come first so its libionic (and, in older image-staged
        # copies, the rdma-core 50 libibverbs it needed) wins over the
        # container's own.
        FABRIC_ENV+=" --env LD_LIBRARY_PATH=${ANP_DIR}:${CONTAINER_LD_PATH}"
        # No silent socket fallback across nodes. If librccl-anp.so fails to
        # load or finds no ionic device, RCCL quietly uses NET/Socket (~237
        # Gbps at the PP boundary instead of ~2,900). Two guards:
        #  * NCCL_NET=RCCL-ANP: RCCL may only assign the ANP plugin to a
        #    communicator, so a failed load is a comm-init error, not TCP.
        #    This covers vLLM's own communicators, not just the health check.
        #  * ZENITH_HEALTH_MIN_PP_GBPS: ray_health_rccl.py fails (and the
        #    launcher refuses to serve) if the measured stage handoff is below
        #    1500 Gbps with GDR / 500 without, which also catches a loss of
        #    GPUDirect RDMA. See SOCKET_FALLBACK_GBPS / MIN_PP_GBPS there.
        # ZENITH_ALLOW_SOCKET_FALLBACK=1 drops both, for debugging.
        if [ "${MULTISTATE}" = "multi" ] && [ "${ZENITH_ALLOW_SOCKET_FALLBACK:-0}" != "1" ]; then
            if [ "${ANP_DIR}" = "/opt/zenith_patches/anp" ] && [ ! -f "${SCRIPTDIR}/patches/anp/librccl-anp.so" ]; then
                echo "ERROR: ${SCRIPTDIR}/patches/anp/librccl-anp.so is missing; cross-node RCCL"
                echo "       would fall back to TCP sockets. Restore patches/anp, point"
                echo "       ZENITH_ANP_DIR at another copy, or set ZENITH_ALLOW_SOCKET_FALLBACK=1."
                exit 1
            fi
            FABRIC_ENV+=" --env NCCL_NET=${ZENITH_NCCL_NET:-RCCL-ANP}"
            case "${ROCE_GDR_LEVEL}" in
                0|LOC) _MIN_PP=500 ;;
                *)     _MIN_PP=1500 ;;
            esac
            FABRIC_ENV+=" --env ZENITH_HEALTH_MIN_PP_GBPS=${ZENITH_HEALTH_MIN_PP_GBPS:-${_MIN_PP}}"
        fi
    else
        FABRIC_ENV+=" --env NCCL_NET_PLUGIN=none"
        FABRIC_ENV+=" --env NCCL_TUNER_PLUGIN=none"
    fi
elif [ "${FABRIC}" = "slingshot" ]; then
    # ---- Slingshot: RCCL over cxi through the aws-ofi plugin ----
    # The FI_CXI_* tuning (including FI_CXI_OFLOW_BUF_SIZE) is an image default.
    FABRIC_ENV+=" --env NCCL_IB_DISABLE=1"          # cxi, not verbs
    FABRIC_ENV+=" --env OFI_NCCL_PLATFORM=cray"
    # SENDRECV, not rdma. On tuolumne (SHS 13.1, libfabric 2.6, MI300A) the
    # rdma protocol initializes and picks Libfabric/GDRDMA, then every
    # collective dies with "NET/OFI fi_writedata failed; RC: -38" (ENOSYS)
    # and torch reports ncclSystemError / a hang. SENDRECV runs at line rate:
    # 179 Gbps per-GPU cross-node send/recv, 735 Gbps all-reduce busbw on
    # 4 nodes x 4 GPUs. Measured 2026-10-05.
    FABRIC_ENV+=" --env OFI_NCCL_PROTOCOL=${ZENITH_OFI_PROTOCOL:-SENDRECV}"
    # Name the net plugin explicitly. The image sets NCCL_NET_PLUGIN_PATH,
    # which is the *search directory*; whether RCCL also needs the plugin
    # named has varied across RCCL versions, and when the load fails RCCL
    # falls back to NET/Socket with only an INFO-level message. Slingshot is
    # where that fallback has actually been observed (~0.9 Gbps all-reduce on
    # tuolumne), so set it here only -- the RoCE path is working today and is
    # deliberately left alone.
    FABRIC_ENV+=" --env NCCL_NET_PLUGIN=${ZENITH_NCCL_NET_PLUGIN:-/opt/aws-ofi-rccl/lib/libnccl-net.so}"
    # Slingshot job credentials. Where the workload manager hands the job its
    # own VNI(s) (SLINGSHOT_VNIS etc. in the job env), libfabric's cxi provider
    # has to use those -- an LNI on any other VNI is refused by the cxi service.
    # Pass through whichever of the four are set, and point FI_CXI_DEFAULT_VNI
    # at the first VNI (the list is comma-separated; strip any ":<count>").
    # Unset on the host (as on tuolumne today): the image defaults stand,
    # VNI=1 / SVC_IDS=1.
    for _v in SLINGSHOT_VNIS SLINGSHOT_SVC_IDS SLINGSHOT_DEVICES SLINGSHOT_TCS; do
        [ -n "${!_v:-}" ] && FABRIC_ENV+=" --env ${_v}=${!_v}"
    done
    if [ -n "${SLINGSHOT_VNIS:-}" ]; then
        _vni="${SLINGSHOT_VNIS%%,*}"; _vni="${_vni%%:*}"
        FABRIC_ENV+=" --env FI_CXI_DEFAULT_VNI=${_vni}"
        echo "  slingshot vni   : ${_vni} (SLINGSHOT_VNIS=${SLINGSHOT_VNIS}, SVC_IDS=${SLINGSHOT_SVC_IDS:-image default})"
    fi
    # ZENITH_HOST_FABRIC_DIR=<dir> (staged by stage_host_fabric.sh) swaps the
    # image's source-built libfabric/libcxi/plugin for the host's own, in case
    # those do not match the host cxi driver. Host libs first, their RHEL-only
    # deps last -- see the layout note in stage_host_fabric.sh. The tuner plugin
    # is the image's and links the image's libfabric, so it is turned off.
    if [ -n "${ZENITH_HOST_FABRIC_DIR:-}" ]; then
        FABRIC_ENV+=" --env NCCL_NET_PLUGIN=/opt/hostfabric/librccl-net.so"
        FABRIC_ENV+=" --env NCCL_TUNER_PLUGIN=none"
        FABRIC_ENV+=" --env LD_LIBRARY_PATH=/opt/hostfabric:${CONTAINER_LD_PATH}:/opt/hostfabric/deps"
        HOST_FABRIC_MOUNT="-v ${ZENITH_HOST_FABRIC_DIR}:/opt/hostfabric:ro"
        echo "  host fabric     : ${ZENITH_HOST_FABRIC_DIR} -> /opt/hostfabric"
    fi
else
    # ---- plain ethernet: RCCL over sockets ----
    FABRIC_ENV+=" --env NCCL_IB_DISABLE=1"
fi

# escape hatches for debugging a hang: ZENITH_DISABLE_P2P=1 / ZENITH_DISABLE_IB=1
# (after the fabric blocks, so they win over them)
[ "${ZENITH_DISABLE_P2P:-0}" = "1" ] && FABRIC_ENV+=" --env NCCL_P2P_DISABLE=1"
[ "${ZENITH_DISABLE_IB:-0}"  = "1" ] && FABRIC_ENV+=" --env NCCL_IB_DISABLE=1"

# start up ray within the container
echo "starting up container on ${MYHOSTNAME} with ray command line ${RAY_START_CMD}"
${PODMAN} run \
   -it \
   --log-level=error \
   --name=${CNAME} \
   --network=host \
   --group-add=video \
   --group-add=render \
   --ipc=host \
   --pid=host \
   `# no --shm-size here: podman rejects it under --ipc=host, and it would be` \
   `# redundant anyway -- the container gets the host's /dev/shm directly, which` \
   `# is what vLLM's multiproc workers need for their shared tensors.` \
   --cap-add=SYS_PTRACE \
   --cap-add=IPC_LOCK \
   --ulimit memlock=-1:-1 \
   --ulimit stack=67108864:67108864 \
   --security-opt seccomp=unconfined \
   --device /dev/kfd \
   --device /dev/dri \
   ${DEVICE_ARGS} \
   --privileged  \
   ${CONTAINER_ENV} \
   ${FABRIC_ENV} \
   `# ZENITH_CONTAINER_EXTRA_ENV: extra container-wide env, space-separated` \
   `# NAME=VALUE. Last, so it overrides the image and everything above.` \
   $(for _kv in ${ZENITH_CONTAINER_EXTRA_ENV:-}; do printf -- "--env %s " "${_kv}"; done) \
   -v $MODELPATH:/app/models \
   -v $DATASETSPATH:/app/datasets \
   `# mount the patch dir read-only so PATCH_FIXUP works even against an image` \
   `# built before a patch was added; the image also bakes them in at` \
   `# /opt/zenith_patches, and this mount simply shadows that with the same files.` \
   -v ${SCRIPTDIR}/patches:/opt/zenith_patches:ro \
   `# nixl_rocm wheel, if one has been staged by build_nixl_rocm.sh. Expands to` \
   `# nothing when nixl_build/ holds no wheel -- see NIXL_FIXUP above.` \
   ${NIXL_MOUNT} \
   ${HOST_FABRIC_MOUNT:-} \
   ${IMAGE_REF} \
   bash -lc "${ARCH_CHECK} ${ENV_CHECK} ${SMI_FIXUP} ${KPACK_FIXUP} ${PATCH_FIXUP} ${NIXL_FIXUP} exec ${RAY_START_CMD}"
