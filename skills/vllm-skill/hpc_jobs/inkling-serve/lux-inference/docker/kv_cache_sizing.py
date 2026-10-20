#!/usr/bin/env python3
"""
Size the vLLM KV cache to a fixed number of GB per GPU.

vLLM has no "give me N GB of KV cache" flag. It takes --gpu-memory-utilization
(a fraction of total VRAM) and spends that budget on, in order:

    weights + non-torch overhead + peak activation//graph memory + KV cache

so the KV cache is the *remainder*. To pin the KV cache at a target size we
have to solve for the utilization fraction:

    util = (weights_per_gpu + overhead + target_kv) / total_vram_per_gpu

The original requirement was to reserve, on MI355X, the same amount of KV cache
that half of an MI300A's VRAM would give -- 128 GB HBM3 per APU, so 64 GB per
GPU -- and because MI355X has 288 GiB per GPU that same 64 GB is a very
different *fraction* of the card, hence solving for the fraction rather than
reusing a utilization number.

That 64 GB target is still what the MI355X runs use, but it does not survive on
the MI300A: an MI300A's HBM is the node's system memory, so a 4-GPU node has
only ~502 GiB total for weights + KV + host-side everything. Asking for 64 GB
of KV per GPU there works out to utilization ~0.84, which claims 85% of node
RAM and gets the job killed by the batch system's memory cgroup before vLLM
ever finishes loading. Adding nodes does NOT help, because the KV target is
per-GPU: only the weights term shrinks. ZENITH_TARGET_KV_GB=32 is the value
that runs on Tuolumne today. See MI300A_PORTABILITY.md.

Three things this script gets asked to be careful about, all of which have
already cost a job:

  * total_vram_per_gpu is per-*machine*. There is no default: pass --vram-bytes
    or let it be detected from the KFD topology. An earlier version defaulted
    to the MI355X figure, which sized MI300A runs against a card 2.25x too
    large and asked vLLM for utilization 0.40 -- a budget smaller than the
    weights, so the run OOMed during model loading.

  * weights_per_gpu is the *resident* footprint, not the file size. MXFP4
    checkpoints do not stay packed in HBM. How much they inflate depends on
    which MoE backend the arch selects -- see PACKED_INFLATION below.

  * on an APU (gfx942) the utilization fraction is drawn from a pool shared
    with the host, so a "safe" fraction of VRAM can still be an unsafe fraction
    of the node. This script does not model host-side demand; it only sizes the
    GPU budget.

Usage:
    python3 kv_cache_sizing.py --model /path/to/model --tp 8 [--pp 1] \
        [--target-kv-gb 64] [--max-model-len 100000] [--vram-bytes N]
"""

import argparse
import glob
import json
import math
import os
import struct
import sys

# Known cards, for error messages and for sanity-checking a detected value.
# The MI250X figure is per *GCD*, not per package: an MI250X is two dies of
# 64 GiB each and every layer above (KFD, HIP, torch, vLLM) counts them as two
# independent devices, so a "4x MI250X" node presents 8 GPUs of 64 GiB.
MI355X_VRAM_BYTES = 309220868096   # 288 GiB, gfx950, 8 per node
MI300A_VRAM_BYTES = 137438953472   # 128 GiB, gfx942, 4 per node
MI250X_VRAM_BYTES = 68719476736    # 64 GiB per GCD, gfx90a, 8 GCDs per node

# Empirical non-KV, non-weight floor per GPU for a vLLM ROCm worker: HIP context,
# RCCL buffers, the torch caching allocator's fragmentation headroom, CUDA/HIP
# graph capture, and peak activation for the largest batch. 12 GiB is a
# deliberately conservative number for large-MoE + 100k context workloads.
#
# NB: this was calibrated on MI355X, a discrete card. On MI300A the HBM is
# shared with the host, so Ray's object store and the page cache draw from the
# same pool; if you see host-side pressure there, raise it rather than lower it.
DEFAULT_OVERHEAD_GIB = 12.0

GIB = 1024 ** 3
GB = 10 ** 9

# Resident-HBM inflation over on-disk bytes, measured against real vLLM startup
# logs on MI355X (see RESULTS.md for the runs these came from):
#
#   Inkling-MXFP4  TP8 PP1  disk 65.91 GiB/GPU -> loaded 85.56 GiB/GPU  (1.298x)
#   Inkling-MXFP4  TP8 PP2  disk 32.96 GiB/GPU -> loaded 44.19 GiB/GPU  (1.341x)
#   Inkling-bf16   TP8 PP2  disk 110.89 GiB    -> loaded 116.12 GiB     (1.047x)
#
# The whole-model factor is not a constant, because it depends on how much of
# the checkpoint is 4-bit. Splitting by dtype does give a near-constant: the
# bf16 tensors are close to pass-through, and the packed-U8 (fp4, two values
# per byte) tensors are what inflate. On gfx950 vLLM/AITER does not keep MXFP4
# experts in the checkpoint's packed layout -- it unpacks and re-blocks them
# for the fused-MoE kernel, and pads each TP shard -- so the resident footprint
# is larger than the file, and grows slightly as TP*PP shards it further.
#
# Backing the bf16 part out of the MXFP4 numbers implies 1.346x (PP1) and
# 1.396x (PP2) on the U8 bytes alone. We use the larger, both because multinode
# is the case that matters and because the error is one-sided: overestimating
# weights costs KV cache, underestimating them costs the whole job to an OOM
# halfway through model loading.
#
# IMPORTANT: 1.40 is backend-specific, not arch-independent. It was measured on
# the AITER MXFP4 path, which only gfx950 selects. gfx942 and gfx90a fall to
# Mxfp4MoeBackend.EMULATION (OCP_MXQuantizationEmulationTritonExperts), which
# keeps the packed layout resident and dequantizes per call; the Tuolumne run
# measured 27.9-29.8 GiB loaded against 26.4 GiB on disk, i.e. ~1.06-1.13x on
# the whole model rather than ~1.30x. Leaving 1.40 in place there is therefore
# conservative in the safe direction -- it over-reserves and the excess simply
# becomes unused headroom instead of KV cache. Do not lower it without a log
# from the arch in question.
PACKED_INFLATION = 1.40     # U8 / packed-4-bit tensors
UNPACKED_INFLATION = 1.05   # bf16/fp16/fp32 tensors (per-shard padding only)

# safetensors dtype -> is it a packed sub-byte payload that vLLM UNPACKS?
# U8 is how quark/OCP-MX checkpoints store fp4 (two values per byte); I32/U32
# is how AWQ/GPTQ-style checkpoints pack theirs. Everything else is a real
# floating-point tensor that loads roughly 1:1.
#
# I8 is deliberately NOT in this set. Native OCP MXFP4 stored as I8 plus
# companion F8_E8M0 scale tensors is already the layout gfx950's AITER kernels
# consume, so vLLM keeps it packed instead of unpacking and re-blocking it the
# way it does a quark U8 checkpoint; inflating it by 1.40 would over-reserve
# weights that never materialize.
PACKED_DTYPES = {"U8", "U32", "I32", "U4", "I4"}


def safetensors_dtype_bytes(path):
    """
    Byte count per dtype in one safetensors shard, read from its header only.

    Layout: u64 little-endian header length, then that many bytes of JSON, then
    the tensor data. Each entry carries dtype/shape/data_offsets, so we can size
    every tensor without reading (or having enough RAM for) the payload.
    """
    out = {}
    with open(path, "rb") as fh:
        (hdr_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hdr_len))
    for name, meta in header.items():
        if name == "__metadata__" or not isinstance(meta, dict):
            continue
        off = meta.get("data_offsets")
        if not off:
            continue
        out[meta.get("dtype", "?")] = out.get(meta.get("dtype", "?"), 0) + (
            off[1] - off[0])
    return out


def weight_bytes(model_dir):
    """
    (bytes_on_disk, estimated_resident_bytes, per-dtype breakdown).

    On-disk is what the shards weigh; resident is what vLLM will actually hold
    in HBM after unpacking and re-blocking. They are NOT the same for quantized
    checkpoints -- see PACKED_INFLATION above. Sizing off the disk number is
    what put a 3-node MI300A MXFP4 run over 1.0 utilization while the arithmetic
    said 0.90.
    """
    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shards:
        sys.exit(f"no *.safetensors found in {model_dir}")

    disk = 0
    by_dtype = {}
    for f in shards:
        disk += os.path.getsize(f)
        try:
            for dt, n in safetensors_dtype_bytes(f).items():
                by_dtype[dt] = by_dtype.get(dt, 0) + n
        except Exception as exc:  # unreadable/odd header: fall back below
            print(f"warning: could not read safetensors header of "
                  f"{os.path.basename(f)}: {exc}", file=sys.stderr)

    counted = sum(by_dtype.values())
    if counted == 0:
        # No header parsed at all. Apply the packed multiplier to everything --
        # the conservative choice, since guessing low here ends in an OOM.
        print("warning: no safetensors headers parsed; assuming the whole "
              "checkpoint inflates like packed 4-bit", file=sys.stderr)
        return disk, disk * PACKED_INFLATION, {}

    resident = 0.0
    for dt, n in by_dtype.items():
        resident += n * (PACKED_INFLATION if dt in PACKED_DTYPES
                         else UNPACKED_INFLATION)
    # shard file overhead (headers, alignment) is not tensor data; scale the
    # estimate to the full on-disk size so nothing is silently dropped
    if counted < disk:
        resident += (disk - counted) * UNPACKED_INFLATION
    return disk, resident, by_dtype


def detect_vram_bytes():
    """
    VRAM per GPU from the KFD topology, or 0 if it cannot be determined.

    Mirrors detect_vram_bytes() in lux_container_launcher.sh. simd_count==0
    marks the CPU-only topology nodes; heap_type 1 is HEAP_TYPE_FB_PUBLIC, the
    device framebuffer. We take the largest such bank across all GPU nodes.
    """
    best = 0
    for node in sorted(glob.glob("/sys/class/kfd/kfd/topology/nodes/*/")):
        try:
            with open(os.path.join(node, "properties")) as fh:
                props = dict(
                    ln.split(None, 1) for ln in fh.read().splitlines() if ln)
            if props.get("simd_count", "0").strip() == "0":
                continue
        except OSError:
            continue
        for bank in glob.glob(os.path.join(node, "mem_banks", "*", "properties")):
            try:
                with open(bank) as fh:
                    bp = dict(
                        ln.split(None, 1) for ln in fh.read().splitlines() if ln)
            except OSError:
                continue
            if bp.get("heap_type", "").strip() != "1":
                continue
            try:
                best = max(best, int(bp.get("size_in_bytes", "0").strip()))
            except ValueError:
                pass
    return best


def node_shape(vram_per_gpu):
    """
    (gpus_per_node, human-readable node kind) for a given per-GPU VRAM size.

    Used only to phrase the "you need N nodes" message. Per-GPU VRAM does not
    order these machines monotonically -- an MI250X GCD is the *smallest* device
    here at 64 GiB and yet comes 8 to a node, more than the MI300A's 4 -- so
    match against the known sizes rather than thresholding. The KFD-reported
    size is a little under the nominal capacity (carve-outs), hence the window.
    """
    known = [
        (MI250X_VRAM_BYTES, 8, "4xMI250X (8 GCDs)"),
        (MI300A_VRAM_BYTES, 4, "4xMI300A"),
        (MI355X_VRAM_BYTES, 8, "8xMI355X"),
    ]
    for nominal, per_node, kind in known:
        if 0.88 * nominal <= vram_per_gpu <= 1.02 * nominal:
            return per_node, kind
    # Unrecognized card: 8 GPUs per node is the common case, and saying so
    # explicitly beats silently attributing the count to a card this is not.
    return 8, f"unknown {vram_per_gpu/GIB:.0f} GiB GPUs, assuming 8/node"


def load_config(model_dir):
    path = os.path.join(model_dir, "config.json")
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        if not os.path.isdir(model_dir):
            sys.exit(f"error: no such model directory: {model_dir}\n"
                     f"       check ZENITH_MODEL and ZENITH_MODELPATH")
        sys.exit(f"error: {path} not found -- {model_dir} does not look like "
                 f"a HuggingFace checkpoint")
    except json.JSONDecodeError as exc:
        sys.exit(f"error: {path} is not valid JSON: {exc}")


def aiter_moe_is_safe(cfg, tp):
    """
    Can AITER's fused-MoE path handle this model at this tensor-parallel width?

    AITER shuffles the expert weights into a blocked layout before the fused
    kernel runs, and that shuffle asserts the last dimension is a multiple of
    32 (`assert x.shape[-1] % BK == 0`). The expert intermediate dim is sharded
    by TP, so a model can be fine at TP=1 and fail at TP=8:

        gpt-oss-20b intermediate_size=2880 -> TP=8 gives 360, 360 % 32 == 8
        Inkling     intermediate_size=3072 -> TP=8 gives 384, 384 % 32 == 0

    When it is unsafe the run dies well into model loading with a bare
    "AssertionError: 360 % 32 == 8" from inside a worker process, which is a
    long way from anything that names the real cause. Detect it up front and
    let the caller set VLLM_ROCM_USE_AITER_MOE=0 instead -- that keeps AITER
    attention/rmsnorm and only drops the fused-MoE kernel.

    This check is arch-independent on purpose. It is only load-bearing where
    AITER actually runs (CDNA3/gfx942 and CDNA4/gfx950); on CDNA2/gfx90a vLLM
    disables AITER wholesale, so the answer there is moot rather than wrong.

    Returns (safe: bool, reason: str).
    """
    tcfg = cfg.get("text_config", cfg)

    n_experts = (tcfg.get("num_local_experts") or tcfg.get("num_experts")
                 or tcfg.get("n_routed_experts") or 0)
    if not n_experts:
        return True, "not an MoE model"

    inter = (tcfg.get("moe_intermediate_size")
             or tcfg.get("intermediate_size") or 0)
    if not inter:
        return True, "no expert intermediate size in config"

    if inter % tp:
        return True, f"intermediate {inter} not divisible by TP={tp}; vLLM will object first"

    shard = inter // tp
    if shard % 32:
        return (False,
                f"expert intermediate {inter}/TP{tp} = {shard}, "
                f"{shard} % 32 == {shard % 32} (AITER fused-MoE needs a multiple of 32)")
    return True, f"expert intermediate {inter}/TP{tp} = {shard}, 32-aligned"


def kv_bytes_per_token(cfg):
    """
    Per-token KV cache footprint across ALL layers, in bytes.

    Handles the two layouts we care about here:
      * plain dense/MoE transformer  (gpt-oss)
      * Inkling's hybrid full-attention + sliding-window-attention stack
    vLLM allocates the sliding-window layers at the (much smaller) window size
    rather than the full context, but it still allocates a block table for them;
    we account for the full-attention layers exactly and treat SWA layers at
    their window, which is what the v1 hybrid allocator does.

    The output of this function only drives the printed "concurrent sequences"
    line; `util` is computed from --target-kv-gb and never touches it.
    """
    text = cfg.get("text_config", cfg)

    n_layers = text.get("num_hidden_layers")
    head_dim = text.get("head_dim")
    n_kv = text.get("num_key_value_heads")

    if head_dim is None:
        head_dim = text["hidden_size"] // text["num_attention_heads"]

    # KV is stored in bf16/fp16 by default (2 bytes), for K and V both
    dtype_bytes = 2
    per_layer_per_token = 2 * n_kv * head_dim * dtype_bytes

    # --- Inkling hybrid: local_layer_ids are sliding-window layers ---
    local_ids = text.get("local_layer_ids")
    if local_ids is not None:
        swa_kv = text.get("swa_num_key_value_heads", n_kv)
        swa_hd = text.get("swa_head_dim", head_dim)
        n_swa = len(local_ids)
        n_full = n_layers - n_swa
        window = text.get("sliding_window_size", 512)
        return {
            "full_layers": n_full,
            "swa_layers": n_swa,
            "full_per_token": n_full * per_layer_per_token,
            "swa_per_token": n_swa * 2 * swa_kv * swa_hd * dtype_bytes,
            "swa_window": window,
        }

    # --- gpt-oss style: layer_types lists sliding vs full ---
    layer_types = text.get("layer_types")
    if layer_types:
        n_swa = sum(1 for t in layer_types if "sliding" in t)
        n_full = len(layer_types) - n_swa
        return {
            "full_layers": n_full,
            "swa_layers": n_swa,
            "full_per_token": n_full * per_layer_per_token,
            "swa_per_token": n_swa * per_layer_per_token,
            "swa_window": text.get("sliding_window", 128),
        }

    return {
        "full_layers": n_layers,
        "swa_layers": 0,
        "full_per_token": n_layers * per_layer_per_token,
        "swa_per_token": 0,
        "swa_window": 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, required=True, help="tensor parallel size")
    ap.add_argument("--pp", type=int, default=1, help="pipeline parallel size")
    ap.add_argument("--target-kv-gb", type=float, default=64.0,
                    help="KV cache GB per GPU (default 64, the MI355X value; "
                         "MI300A needs 32 -- see the module docstring)")
    ap.add_argument("--max-model-len", type=int, default=100000)
    ap.add_argument("--vram-bytes", type=int, default=0,
                    help="VRAM per GPU in bytes. Default: detect from the KFD "
                         "topology. There is deliberately no hardcoded "
                         "fallback -- see the module docstring.")
    ap.add_argument("--overhead-gib", type=float, default=DEFAULT_OVERHEAD_GIB)
    ap.add_argument("--no-inflate", action="store_true",
                    help="size weights from raw on-disk bytes, skipping the "
                         "quantization inflation model (debugging only)")
    args = ap.parse_args()

    total = args.vram_bytes
    vram_source = "--vram-bytes"
    if total <= 0:
        total = detect_vram_bytes()
        vram_source = "KFD topology"
    if total < GIB:
        sys.exit(
            "error: could not determine VRAM per GPU (KFD topology gave "
            f"{total}). Pass --vram-bytes explicitly:\n"
            f"       {MI250X_VRAM_BYTES}  MI250X (64 GiB per GCD)\n"
            f"       {MI300A_VRAM_BYTES} MI300A (128 GiB)\n"
            f"       {MI355X_VRAM_BYTES} MI355X (288 GiB)\n"
            "       Refusing to guess -- sizing against the wrong card "
            "surfaces as an out-of-memory failure during model load, which "
            "names nothing.")

    world = args.tp * args.pp
    cfg = load_config(args.model)
    disk_total, resident_total, by_dtype = weight_bytes(args.model)
    weights_total = disk_total if args.no_inflate else resident_total
    weights_per_gpu = weights_total / world

    target_kv = args.target_kv_gb * GB
    overhead = args.overhead_gib * GIB

    util = (weights_per_gpu + overhead + target_kv) / total

    kv = kv_bytes_per_token(cfg)
    # tokens the target KV buys, accounting for the hybrid layout. SWA layers
    # cost their window, not the full sequence, once a sequence exceeds it.
    per_tok_full = kv["full_per_token"] / args.tp   # KV heads shard over TP
    per_tok_swa = kv["swa_per_token"] / args.tp
    kv_per_gpu_total = target_kv * world  # aggregate pool across the deployment

    # a single max-length sequence costs:
    seq_cost = per_tok_full * args.max_model_len + per_tok_swa * min(
        args.max_model_len, kv["swa_window"])
    # ...but that is the per-*pipeline-stage* cost; each PP stage holds 1/pp of layers
    seq_cost_per_stage = seq_cost / args.pp

    total_kv_pool = target_kv * args.tp  # per pipeline stage, across its TP ranks
    concurrent = total_kv_pool / seq_cost_per_stage if seq_cost_per_stage else 0

    print(f"model              : {args.model}")
    print(f"parallelism        : TP={args.tp} PP={args.pp}  world={world} GPUs")
    print(f"weights on disk    : {disk_total/GB:8.1f} GB total")
    if by_dtype:
        parts = ", ".join(f"{dt} {n/GB:.1f} GB"
                          for dt, n in sorted(by_dtype.items(),
                                              key=lambda kv: -kv[1]))
        print(f"  dtype breakdown  : {parts}")
    if args.no_inflate:
        print(f"weights resident   : {weights_total/GB:8.1f} GB total"
              f"  ->  {weights_per_gpu/GB:6.1f} GB per GPU  [--no-inflate]")
    else:
        print(f"weights resident   : {weights_total/GB:8.1f} GB total"
              f"  ->  {weights_per_gpu/GB:6.1f} GB per GPU"
              f"  ({weights_total/disk_total:.3f}x disk)")
    print(f"target KV per GPU  : {target_kv/GB:8.1f} GB"
          f"   ({kv_per_gpu_total/GB:.0f} GB aggregate over {world} GPUs)")
    print(f"assumed overhead   : {overhead/GIB:8.1f} GiB per GPU")
    print(f"VRAM per GPU       : {total/GIB:8.1f} GiB   (from {vram_source})")
    print()
    print(f"layers             : {kv['full_layers']} full-attention,"
          f" {kv['swa_layers']} sliding-window (window={kv['swa_window']})")
    print(f"KV per token/GPU   : {per_tok_full/args.tp/1024:8.1f} KiB full +"
          f" {per_tok_swa/args.tp/1024:.1f} KiB swa")
    print(f"cost of one {args.max_model_len}-token seq"
          f" : {seq_cost_per_stage/GB:.2f} GB per pipeline stage")
    print(f"=> concurrent {args.max_model_len}-token sequences: {concurrent:.1f}")
    print()

    if util >= 1.0:
        print(f"*** INFEASIBLE: required utilization {util:.3f} >= 1.0")
        print(f"*** weights+overhead+KV = "
              f"{(weights_per_gpu+overhead+target_kv)/GIB:.0f} GiB "
              f"> {total/GIB:.0f} GiB per GPU")
        headroom = total - overhead - target_kv
        if headroom <= 0:
            print(f"*** overhead + KV alone ({(overhead+target_kv)/GIB:.0f} GiB) "
                  f"exceeds a {total/GIB:.0f} GiB GPU -- no node count fixes "
                  f"this; lower --target-kv-gb.")
            sys.exit(2)
        min_gpus = math.ceil(weights_total / headroom)
        # Report the node count for the machine actually being sized, not for
        # whichever one this was first written on. --vram-bytes is how the
        # caller says "size me for an MI300A", and an MI300A node has 4 GPUs;
        # answering in 8-GPU nodes there is off by 2x in the direction that
        # under-provisions. Note that a 64 GiB device is an MI250X *GCD* and
        # those come 8 to a node, so per-GPU VRAM alone does not order the
        # machines -- match on the known sizes instead of thresholding.
        gpus_per_node, node_kind = node_shape(total)
        print(f"*** need at least {min_gpus} GPUs "
              f"({math.ceil(min_gpus/gpus_per_node)} nodes of {node_kind}) "
              f"to fit weights + {args.target_kv_gb:.0f}GB KV each")
        print(f"*** NB: that is the arithmetic minimum, at utilization ~1.0."
              f" Add a node for allocator headroom.")
        sys.exit(2)

    aiter_ok, aiter_why = aiter_moe_is_safe(cfg, args.tp)
    print(f"AITER fused MoE    : {'ok' if aiter_ok else 'DISABLE'} -- {aiter_why}")
    print()

    print(f"--gpu-memory-utilization {util:.4f}")
    # emit shell-sourceable lines for the launcher
    print(f"ZENITH_GPU_MEM_UTIL={util:.4f}")
    print(f"ZENITH_AITER_MOE={1 if aiter_ok else 0}")


if __name__ == "__main__":
    main()
