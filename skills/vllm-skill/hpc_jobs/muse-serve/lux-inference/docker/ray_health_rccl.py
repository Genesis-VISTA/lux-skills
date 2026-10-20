"""
RCCL health check shaped like the vLLM server the launcher is about to start.

It builds the communicators vLLM builds for the same TP x PP x DP layout, and
only those, and drives each one the way vLLM does:

  TP groups   all-reduce, 1 GiB per GPU x 10   (intra-node XGMI/Infinity Fabric)
  PP groups   activation handoff down the pipeline chain, vLLM-style: plain
              isend/irecv, so a middle stage has a recv from the previous stage
              and a send to the next one in flight at once   (inter-node fabric)
  DP/EP       all-reduce on the DP groups, and all-gather + reduce-scatter on
              the EP groups, only when DP > 1. That is vLLM's default MoE
              all2all backend (allgather_reducescatter); it never calls
              all_to_all, and a 16-rank RCCL all_to_all over cxi failed with
              ncclInProgress on Tioga (f61Lzs5a27GB, DP2 x TP8). Set
              ZENITH_HEALTH_EP_ALLTOALL=1 to also run the all_to_all.

What it deliberately does NOT build any more is one RCCL communicator over the
whole world. The previous version all-reduced over every GPU in the job, a
communicator vLLM never creates, and on Tioga (MI250X, ROCm 7.14) that one comm
fails at init -- RCCL's intra-node IPC (hipIpcOpenMemHandle, 'invalid device
pointer') breaks in comms that mix intra-node and inter-node peers -- while the
TP-per-node + PP-across-nodes comms vLLM uses are fine. The check failed on 16
and 32 GPUs for a reason that could not affect serving. Conversely, it passed
straight over the fault that did stop serving there: a PP chain hung on the
cxi path once a handoff grew past a few tens of KiB (job f618XyXKjXs5;
MI250X_PORTABILITY.md section 8.1 -- the image's FI_CXI_OFLOW_BUF_SIZE=2MB
parsed as 2 bytes, now 2097152). The handoff sizes below start at one token so
that kind of hang is caught here, in seconds, rather than on a user's second
request after the weights are loaded. (With the current image and host fabric
the world comm no longer fails either -- section 8.2 -- but vLLM still never
builds one, so neither does this check.)

Ray Train's own process group is gloo, for the same reason: with the default
nccl backend, any collective on it (a barrier, an all_gather_object) would
build the world RCCL comm this check exists to avoid.

A hang anywhere is turned into a failure by a per-phase watchdog
(ZENITH_HEALTH_TIMEOUT seconds, default 180), which names the phase.
"""

import os
import sys
import socket
import threading
import time

import torch
import torch.distributed as dist
import ray
from ray.train.torch import TorchTrainer, TorchConfig, get_device
from ray.train import ScalingConfig, RunConfig, FailureConfig
from lux_ray_util import cpu_budget_per_worker

# Below this, RCCL is not on the high-speed fabric at all. 1 GbE line rate is
# ~1.0 Gbps and 10 GbE is ~10; a management-network fallback lands in that
# band, while every real fabric measured here is at least an order of magnitude
# above it. ~70 Gbps on Pollara means RCCL is on TCP sockets over the NICs' IP
# interfaces -- not caught by this threshold, so check the transport with
# ZENITH_NCCL_DEBUG=INFO.
# This is a fallback detector, not a performance gate -- do not raise it into
# the range where a merely slow-but-working fabric would trip it. It is applied
# to the aggregate PP handoff bandwidth across one stage boundary (all TP lanes
# at once), which is what a node pair's fabric actually carries.
SOCKET_FALLBACK_GBPS = 10.0

# The hard floor, by contrast, IS a gate: below it the check fails and the
# launcher refuses to serve. run_container.sh sets it on the paths where the
# fabric's healthy figure is known -- on Lux (Pollara RoCE, RCCL-ANP) 1500 Gbps
# with GPUDirect RDMA and 500 host-staged. Measured there (TP8 x PP2, 8192-token
# handoff): 2,836-2,932 Gbps with GDR, 235-237 Gbps on NET/Socket, so a socket
# fallback or a silent loss of GDR trips it and a healthy fabric has ~2x margin.
# 0 (the default) disables it; ZENITH_HEALTH_MIN_PP_GBPS=0 does the same on Lux.
MIN_PP_GBPS = float(os.environ.get("ZENITH_HEALTH_MIN_PP_GBPS", "0") or 0)

# Activation handoff sizes, in tokens of a bf16 hidden state. 1 token is a
# decode step; 8192 is a long prefill chunk. 7 and 33 sit either side of the
# ~84 KiB threshold where the Tioga chain hang starts.
PP_HIDDEN = 8192
PP_TOKENS = (1, 7, 33, 512, 8192)


def vllm_groups(world, tp, pp, dp):
    """
    vLLM's rank groups for TP x PP x DP, tiled over `world` (more than one
    instance only for ZENITH_NO_SERVE disagg layouts). Mirrors
    vllm/distributed/parallel_state.py: all_ranks.reshape(-1, DP, PP, TP), with
    PCP = 1. Returns {name: [rank lists]}; ranks are vLLM ranks, i.e.
    node_rank * gpus_per_node + local_rank.
    """
    r = torch.arange(world).reshape(-1, dp, pp, tp)
    groups = {"tp": r.reshape(-1, tp).tolist()}
    if pp > 1:
        groups["pp"] = r.transpose(2, 3).reshape(-1, pp).tolist()
    if dp > 1:
        groups["dp"] = r.transpose(1, 3).reshape(-1, dp).tolist()
        groups["ep"] = r.transpose(1, 2).reshape(-1, dp * tp).tolist()
    return groups


class Watchdog:
    """Turns a hang into a worker exit that Ray Train reports as a failure."""

    def __init__(self, timeout_s, who):
        self.timeout_s, self.who = timeout_s, who
        self.done = threading.Event()

    def phase(self, label):
        self.done.set()
        self.done = threading.Event()
        done = self.done

        def watch():
            if not done.wait(self.timeout_s):
                print(f"*** {self.who}: HANG in {label} (> {self.timeout_s:.0f} s)", flush=True)
                os._exit(3)
        threading.Thread(target=watch, daemon=True).start()

    def stop(self):
        self.done.set()


def rccl_health_func(config):
    tp, pp, dp, gpn = config["tp"], config["pp"], config["dp"], config["gpus_per_node"]
    ctx = ray.train.get_context()
    rank, world = dist.get_rank(), dist.get_world_size()
    device = get_device()
    torch.cuda.set_device(device)
    host = socket.gethostname()

    # vLLM rank of every Ray Train rank. Exchanged over gloo; no RCCL yet.
    me = (ctx.get_node_rank(), ctx.get_local_rank())
    everyone = [None] * world
    dist.all_gather_object(everyone, me)
    vrank = [n * gpn + l for n, l in everyone]
    if sorted(vrank) != list(range(world)):
        raise RuntimeError(f"Ray Train placement is not {world // gpn} nodes x {gpn} GPUs: "
                           f"(node_rank, local_rank) = {everyone}")
    global_of = {v: g for g, v in enumerate(vrank)}
    my_v = vrank[rank]

    # every rank must call new_group for every group, in the same order
    mine = {}
    for name, lists in vllm_groups(world, tp, pp, dp).items():
        for ranks in lists:
            g = dist.new_group(ranks=[global_of[v] for v in ranks], backend="nccl")
            if my_v in ranks:
                mine[name] = (g, ranks)

    dog = Watchdog(config["timeout_s"], f"rank {my_v} ({host} GPU {me[1]})")
    out = {"vrank": my_v, "host": host}

    # ---- TP: all-reduce, as before but per TP group instead of world-wide ----
    g, ranks = mine["tp"]
    n = len(ranks)
    x = torch.ones((config["size_mb"] << 20) // 4, device=device)
    dog.phase(f"TP all-reduce (group {ranks[0]}..{ranks[-1]})")
    dist.all_reduce(x, group=g)                               # comm init + warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(config["iterations"]):
        dist.all_reduce(x, group=g)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / config["iterations"]
    out["tp_busbw"] = x.numel() * 4 * 2 * (n - 1) / n / dt * 8 / 1e9 if n > 1 else None
    del x

    # ---- PP: chain handoff, vLLM's order (recv from previous, send to next) ----
    if pp > 1:
        g, ranks = mine["pp"]
        stage = ranks.index(my_v)
        prev = global_of[ranks[stage - 1]] if stage > 0 else None
        nxt = global_of[ranks[stage + 1]] if stage + 1 < pp else None
        for tokens in PP_TOKENS:
            numel = tokens * PP_HIDDEN
            send = torch.full((numel,), float(stage + 1), dtype=torch.bfloat16, device=device)
            recv = torch.empty(numel, dtype=torch.bfloat16, device=device)
            dog.phase(f"PP handoff {tokens} tokens ({numel * 2 >> 10} KiB), stage {stage} of {pp}")

            def handoff():
                hs = []
                if prev is not None:
                    hs.append(dist.irecv(recv, src=prev, group=g))
                if nxt is not None:
                    hs.append(dist.isend(send, dst=nxt, group=g))
                for h in hs:
                    h.wait()
            # One handoff in flight per pair at a time, as in vLLM, which waits on
            # its previous PP send before issuing the next. work.wait() on an
            # NCCL op does not block the CPU, so without the synchronize a loop
            # queues every iteration's send at once -- a pattern vLLM never
            # produces.
            handoff()
            torch.cuda.synchronize()
            dist.barrier()                                        # gloo
            iters = 5
            t0 = time.perf_counter()
            for _ in range(iters):
                handoff()
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / iters
            if prev is not None:
                if not bool(torch.all(recv == float(stage)).item()):
                    raise RuntimeError(f"PP handoff {tokens} tokens: wrong values from stage {stage - 1}")
                out["pp_gbps"] = numel * 2 / dt * 8 / 1e9          # last (largest) size wins
                out["pp_stage"] = stage
            del send, recv

    # ---- DP / EP, only when the layout has them ----
    if dp > 1:
        g, ranks = mine["dp"]
        x = torch.ones(16 << 20, device=device)
        dog.phase("DP all-reduce")
        dist.all_reduce(x, group=g)
        torch.cuda.synchronize()
        g, ranks = mine["ep"]
        x = torch.ones(1 << 20, device=device)
        y = torch.empty(len(ranks) * (1 << 20), device=device)
        dog.phase("EP all-gather")
        dist.all_gather_into_tensor(y, x, group=g)
        torch.cuda.synchronize()
        dog.phase("EP reduce-scatter")
        dist.reduce_scatter_tensor(x, y, group=g)
        torch.cuda.synchronize()
        if not bool(torch.all(x == float(len(ranks))).item()):
            raise RuntimeError("EP all-gather + reduce-scatter: wrong values")
        if os.environ.get("ZENITH_HEALTH_EP_ALLTOALL", "0") == "1":
            x = torch.ones(len(ranks) * (1 << 20), device=device)
            y = torch.empty_like(x)
            dog.phase("EP all-to-all")
            dist.all_to_all_single(y, x, group=g)
            torch.cuda.synchronize()
        del x, y

    dog.stop()
    results = [None] * world
    dist.all_gather_object(results, out)                     # gloo
    if rank == 0:
        report(results, config)
    ray.train.report({"status": "READY"})


def report(results, config):
    tp, pp, dp = config["tp"], config["pp"], config["dp"]
    tpbw = [r["tp_busbw"] for r in results if r.get("tp_busbw") is not None]
    if tpbw:
        print(f"TP all-reduce (TP={tp}, {config['size_mb']} MiB): bus bandwidth "
              f"min {min(tpbw):.1f} / max {max(tpbw):.1f} Gbps over {len(tpbw)} GPUs")
    if pp > 1:
        print(f"PP chain handoff (PP={pp}, {len(PP_TOKENS)} sizes {PP_TOKENS[0]}..{PP_TOKENS[-1]} "
              f"tokens x {PP_HIDDEN} bf16): all stages completed, values correct")
        worst = None
        for s in range(1, pp):
            per_gpu = [r["pp_gbps"] for r in results if r.get("pp_stage") == s]
            agg = sum(per_gpu) / max(1, len(results) // (tp * pp * dp))   # per instance
            print(f"  stage {s - 1} -> {s}: {agg:.1f} Gbps aggregate over {tp} TP lanes "
                  f"(per GPU min {min(per_gpu):.1f} Gbps, {PP_TOKENS[-1]}-token handoff)")
            worst = agg if worst is None else min(worst, agg)
        if worst is not None and worst < SOCKET_FALLBACK_GBPS:
            print(f"*** WARNING: {worst:.2f} Gbps across a stage boundary is at ethernet "
                  f"line rate, not fabric speed. RCCL has almost certainly "
                  f"fallen back to its TCP sockets transport.")
            print("***   Re-run with ZENITH_NCCL_DEBUG=INFO and grep the log "
                  "for 'NET/': 'via NET/RCCL-ANP' (Pollara RoCE) or "
                  "'NET/OFI ... provider: cxi' (Slingshot) is healthy, "
                  "'NET/Socket' confirms the fallback.")
            print("***   Inference will still work and still produce correct "
                  "output; every inter-node transfer will just be ~100x "
                  "slower than the hardware can do.")
        if worst is not None and worst < MIN_PP_GBPS:
            print(f"*** FABRIC FLOOR: {worst:.1f} Gbps across a stage boundary is below "
                  f"ZENITH_HEALTH_MIN_PP_GBPS={MIN_PP_GBPS:g}. RCCL is on NET/Socket or "
                  f"has lost GPUDirect RDMA; grep the log for 'via NET/' to see which.")
            raise RuntimeError(f"PP handoff {worst:.1f} Gbps < floor {MIN_PP_GBPS:g} Gbps")
    if dp > 1:
        print(f"DP all-reduce and EP all-gather + reduce-scatter (DP={dp}, EP={dp * tp}): completed")


def main():
    if len(sys.argv) < 4:
        print("Usage: python ray_health_rccl.py <nnodes> <cpus_per_node> <gpus_per_node> [tp pp dp]")
        sys.exit(1)

    nnodes = int(sys.argv[1])
    cpus_per_node = int(sys.argv[2])
    gpus_per_node = int(sys.argv[3])
    tp, pp, dp = (int(v) for v in sys.argv[4:7]) if len(sys.argv) >= 7 else (gpus_per_node, nnodes, 1)

    total_workers = nnodes * gpus_per_node
    # The launcher refuses to serve a layout that does not fill the allocation,
    # but ZENITH_NO_SERVE disagg layouts tile it with several instances; test
    # those as tiles. Anything else (TP across nodes, a remainder) falls back to
    # the default layout rather than inventing groups vLLM would not build.
    if total_workers % (tp * pp * dp) or tp > gpus_per_node:
        print(f"TP={tp} x PP={pp} x DP={dp} does not tile {nnodes} x {gpus_per_node} GPUs "
              f"within nodes; checking the default TP={gpus_per_node} x PP={nnodes} instead.")
        tp, pp, dp = gpus_per_node, nnodes, 1

    # Ray workers are forked by the raylet, not by this driver, so `podman exec
    # --env NCCL_...` (the launcher's NCCL_DEBUG_ARGS, a diagnostic override)
    # would otherwise never reach the ranks that build the comms
    fwd = {k: v for k, v in os.environ.items()
           if k.startswith(("NCCL_", "RCCL_", "FI_", "HSA_", "ZENITH_"))}
    ray.init(address="auto", ignore_reinit_error=True, runtime_env={"env_vars": fwd})

    # see lux_ray_util: must be sized against ray's CPUs, not the node's
    cpus_per_worker = cpu_budget_per_worker(cpus_per_node, total_workers)

    trainer = TorchTrainer(
        train_loop_per_worker=rccl_health_func,
        train_loop_config={"size_mb": 1024, "iterations": 10, "tp": tp, "pp": pp, "dp": dp,
                           "gpus_per_node": gpus_per_node,
                           "timeout_s": float(os.environ.get("ZENITH_HEALTH_TIMEOUT", "180"))},
        scaling_config=ScalingConfig(
            num_workers=total_workers,
            use_gpu=True,
            resources_per_worker={"CPU": cpus_per_worker, "GPU": 1}
        ),
        torch_config=TorchConfig(backend="gloo"),
        run_config=RunConfig(failure_config=FailureConfig(max_failures=0)),
    )

    print(f"Testing vLLM's communicators for TP={tp} x PP={pp} x DP={dp} on "
          f"{total_workers} workers across {nnodes} nodes "
          f"({cpus_per_worker} CPU + 1 GPU each)...")
    # Measured reference points, so the numbers can be read against something
    # real rather than against a nominal link rate:
    #
    #   TP all-reduce, 8x MI355X over XGMI                     3068 Gbps bus bw
    #
    # The PP handoff figure is new with this version of the check and has no
    # healthy reference yet. The old check's inter-node numbers (2727-2849 Gbps
    # on Pollara RoCE with GPUDirect RDMA, 936-958 host-staged) were all-reduce
    # bus bandwidths over the whole world and are not comparable to it. The
    # thing that is unambiguously a fault is the sub-10 Gbps band.
    try:
        trainer.fit()
    except Exception as exc:                                 # noqa: BLE001
        print(f"RCCL HEALTH CHECK FAILED: {exc}")
        print("  a 'HANG in PP handoff' line above means the PP chain itself hangs "
              "(MI250X_PORTABILITY.md section 8); vLLM would hang the same way")
        print(f"cluster resources: {ray.cluster_resources()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
