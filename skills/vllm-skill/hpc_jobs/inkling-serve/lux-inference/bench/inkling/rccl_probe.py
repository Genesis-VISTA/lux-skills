#!/usr/bin/env python3
"""2-node RCCL all-reduce / send-recv bandwidth sweep (torchrun, 8 procs/node).

Prints algbw and busbw (busbw = algbw * 2(n-1)/n for all-reduce) per size.
A per-size watchdog prints HANG and hard-exits if a collective does not finish,
so a GDR stall is reported instead of wedging the job.

Run inside the containers of a 2-node cluster brought up with ZENITH_NO_SERVE=1
(containers + Ray + health check, no vLLM), one command per node, same port.
Copy this file into each container first (podman cp):

  podman exec <container> env NCCL_COMM_BLOCKING=1 PROBE_TAG=gdr0 NCCL_NET_GDR_LEVEL=0 \
    torchrun --nnodes 2 --node_rank <0|1> --nproc_per_node 8 \
             --master_addr <headnode> --master_port 29600 rccl_probe.py

Env: PROBE_SIZES_MIB (default 1,16,64,96,128,256,1024), PROBE_ITERS (10),
PROBE_HANG_S (90). Expected on zenith_v6 with GDR off (2026-10-04): 1 GiB
all-reduce busbw ~944 Gbps (algbw ~504); send/recv ~109 Gbps per GPU.

GDR (README §10 gotcha 5) is the launcher's default (ZENITH_NET_GDR_LEVEL=3);
the NCCL_NET_GDR_LEVEL=0 above turns it off for this probe only. To probe GDR,
drop that override (the container already has NCCL_NET_GDR_LEVEL=3
NCCL_GDR_FLUSH_DISABLE=1 NCCL_NCHANNELS_PER_NET_PEER=4), or pass those three
to podman exec in a cluster started with ZENITH_NET_GDR_LEVEL=0. Expected (2026-10-04, 16 ranks): 1 GiB all-reduce busbw
~2,970 Gbps; send/recv ~383 Gbps per GPU. NCCL_NET_GDR_LEVEL=3 without
NCCL_GDR_FLUSH_DISABLE=1 hangs (watchdog prints HANG). In a container not
started by the launcher, also pass TORCH_NCCL_BLOCKING_WAIT=0: the image bakes
in 1, which adds a ~1.1 ms floor to every send/recv (the launcher sets 0).
NCCL_COMM_BLOCKING=1 is required: the image sets NCCL_COMM_BLOCKING=0, and with
that the first send/recv fails with "NCCL operation in progress" / "Unknown
NCCL error". To see the transport (ZENITH_NCCL_DEBUG reaches only the
launcher's health check), add NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET and
look for GDRDMA on every "via NET/RCCL-ANP" line.
"""
import os, sys, time, threading
import torch, torch.distributed as dist

sizes_mib = [int(s) for s in os.environ.get("PROBE_SIZES_MIB", "1,16,64,96,128,256,1024").split(",")]
iters = int(os.environ.get("PROBE_ITERS", "10"))
hang_s = float(os.environ.get("PROBE_HANG_S", "90"))
dist.init_process_group("nccl")
r, n = dist.get_rank(), dist.get_world_size()
lr = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
tag = os.environ.get("PROBE_TAG", "")
gpn = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
# PROBE_HOG_FRAC=0.77: take that fraction of the GPU's total memory before any
# comm exists, as vLLM's weights + KV cache do (on an MI300A APU that is system
# memory too); PROBE_PIN_GB pins that much host memory per rank on top
hog = []
if float(os.environ.get("PROBE_HOG_FRAC", "0")) > 0:
    free, total = torch.cuda.mem_get_info()
    want = int(total * float(os.environ["PROBE_HOG_FRAC"])) - (total - free)
    while want > (1 << 30):
        hog.append(torch.ones(1 << 30, dtype=torch.uint8, device="cuda")); want -= 1 << 30
if float(os.environ.get("PROBE_PIN_GB", "0")) > 0:
    hog.append(torch.empty(int(float(os.environ["PROBE_PIN_GB"]) * (1 << 30)), dtype=torch.uint8).pin_memory())
if hog and r == 0:
    free, total = torch.cuda.mem_get_info()
    print(f"[probe {tag}] holding {len(hog)} buffers; GPU free {free / 2**30:.1f} of {total / 2**30:.1f} GiB", flush=True)
# barriers on gloo, so that PROBE_VLLM=1 never builds a world-sized RCCL communicator
cpu = dist.new_group(backend="gloo")
# PROBE_VLLM=1: only the communicators vLLM TPxPP builds -- a TP group per node
# and a PP group per local rank -- in place of world all_reduce/sendrecv
vllm = os.environ.get("PROBE_VLLM", "0") == "1"
tps = [dist.new_group(list(range(k * gpn, (k + 1) * gpn))) for k in range(n // gpn)]
if r == 0:
    print(f"[probe {tag}] world={n} NCCL_NET_GDR_LEVEL={os.environ.get('NCCL_NET_GDR_LEVEL')} "
          f"LD_PRELOAD={os.environ.get('LD_PRELOAD')}", flush=True)

done = threading.Event()
def watchdog(label):
    if not done.wait(hang_s):
        print(f"[probe {tag}] rank {r}: HANG in {label} (> {hang_s:.0f} s)", flush=True)
        os._exit(3)

def timed(label, fn):
    done.clear(); t = threading.Thread(target=watchdog, args=(label,), daemon=True); t.start()
    fn(); torch.cuda.synchronize()          # warmup
    dist.barrier(group=cpu); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / iters
    done.set(); return dt

for mib in sizes_mib:
    x = torch.ones(mib * 2**20 // 2, dtype=torch.bfloat16, device="cuda")
    g, gn, what = (tps[r // gpn], gpn, "tp_all_reduce") if vllm else (None, n, "all_reduce")
    dt = timed(f"{what} {mib} MiB", lambda: dist.all_reduce(x, group=g))
    alg = x.numel() * 2 / dt * 8 / 1e9; bus = alg * 2 * (gn - 1) / gn
    ok = bool(torch.allclose(x[:8].float(), torch.full((8,), float(gn) ** (iters + 1), device="cuda")))  # value check
    if r == 0:
        print(f"[probe {tag}] {what} {mib:5d} MiB  {dt*1e3:8.2f} ms  algbw {alg:8.1f} Gbps  busbw {bus:8.1f} Gbps  values_ok={ok}", flush=True)
    del x
# cross-node point-to-point: rank i (node 0) <-> rank i+8 (node 1), all 8 pairs at once
half = n // 2
for mib in ([] if vllm else sizes_mib):
    x = torch.ones(mib * 2**20 // 2, dtype=torch.bfloat16, device="cuda")
    peer = r + half if r < half else r - half
    def sr():
        ops = [dist.P2POp(dist.isend, x, peer), dist.P2POp(dist.irecv, torch.empty_like(x), peer)]
        for q in dist.batch_isend_irecv(ops): q.wait()
    dt = timed(f"sendrecv {mib} MiB", sr)
    if r == 0:
        print(f"[probe {tag}] sendrecv  {mib:5d} MiB  {dt*1e3:8.2f} ms  per-GPU uni-dir {x.numel()*2/dt*8/1e9:8.1f} Gbps (x8 pairs per node)", flush=True)
    del x
# vLLM-style pipeline handoff: plain (unbatched) isend/irecv of activation-sized
# tensors on a PP subgroup, at an odd element offset inside a larger caching-
# allocator block. Inkling-MXFP4 under vLLM failed on its first PP isend on
# tuolumne (jobs f3cQ27TiMRWX, f3cQz7wrEWmV: "NET/OFI Unable to register memory
# (type = 1) ... RC: -22"; type 1 is NCCL_PTR_HOST, i.e. RCCL's own pinned host
# buffers, registered while connecting). This case alone did not reproduce it
# (f3cQsDZxDwzo); vLLM also runs with NCCL_CUMEM_ENABLE=0, so pass that too.
if os.environ.get("PROBE_PP", "1") == "1":
    stages = n // gpn
    groups = [dist.new_group([g + s * gpn for s in range(stages)]) for g in range(gpn)]
    pg, stage = groups[lr], r // gpn
    hidden = int(os.environ.get("PROBE_PP_HIDDEN", "6144"))
    # element offsets into the send/recv allocations; PROBE_PP_OFFSETS=0,0 aligns both
    xo, yo = (int(v) for v in os.environ.get("PROBE_PP_OFFSETS", "777,333").split(","))
    for tokens in [int(t) for t in os.environ.get("PROBE_PP_TOKENS", "1,7,33,512,2048,8192").split(",")]:
        pool = torch.ones((tokens + 2) * hidden + 1024, dtype=torch.bfloat16, device="cuda")
        x = pool[xo:xo + tokens * hidden]                   # not page aligned by default
        y = torch.empty(tokens * hidden + yo, dtype=torch.bfloat16, device="cuda")[yo:]
        # PROBE_PP_MODE: send_first (default), recv_first (vLLM's per-step order), or
        # batch (send and recv in one NCCL group via batch_isend_irecv)
        mode = os.environ.get("PROBE_PP_MODE", "send_first")
        def handoff():
            ops = []
            if stage > 0: ops.append(dist.P2POp(dist.irecv, y, r - gpn, group=pg))
            if stage + 1 < stages: ops.append(dist.P2POp(dist.isend, x, r + gpn, group=pg))
            if mode == "send_first": ops.reverse()
            if mode == "batch":
                hs = dist.batch_isend_irecv(ops)
            else:
                hs = [op.op(op.tensor, op.peer, group=pg) for op in ops]
            for h in hs: h.wait()
        dt = timed(f"pp_handoff {tokens} tokens", handoff)
        ok = stage == 0 or bool(torch.all(y == 1).item())
        if r == gpn:
            print(f"[probe {tag}] pp_handoff {tokens:5d} tok  {x.numel()*2/1024:9.1f} KiB  {dt*1e6:9.1f} us  values_ok={ok}", flush=True)
        del pool, x, y
dist.barrier(group=cpu)
if r == 0:
    print(f"[probe {tag}] DONE", flush=True)
dist.destroy_process_group()
