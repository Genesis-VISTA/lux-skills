# Times one context-parallel attention kernel at one point, fwd+bwd, as RingX's benchmark/benchmark_ringX_func.py
# (MIT, jqyin/ringX-attention) does: random bf16 (b, s, H, d) inputs on every rank, 1 warm-up iteration, then
# --num_iter iterations timed with CUDA events. One rank per GPU, launched by torchrun. One kernel per launch:
# job.lux.slurm runs one torchrun per kernel and point, so an out-of-memory failure ends only that run. Prints the
# upstream log lines (benchmark/utils.py parses them) and one RESULT line for summarize.py. --seq_length is tokens
# per GPU, as upstream.
import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist

BASELINES = {"ring": "ring_flash_attn_func", "zigzag": "zigzag_ring_flash_attn_func", "stripe": "stripe_flash_attn_func"}


def load(name):
    if name in BASELINES:
        import fa_rng_fix  # noqa: F401  (Lux: rng_state for the ring-flash-attn baselines on ROCm FA 2.8.3)
        import ring_flash_attn
        return getattr(ring_flash_attn, BASELINES[name])
    import ringX_attn
    return getattr(ringX_attn, f"{name}_attn_func")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", required=True, help="ringX1..ringX4o, or a baseline: ring, zigzag, stripe")
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--seq_length", type=int, required=True, help="tokens per GPU")
    p.add_argument("--num_heads", type=int, default=32)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--num_iter", type=int, default=5)
    p.add_argument("--causal", action="store_true")
    a = p.parse_args()

    dist.init_process_group("nccl", timeout=timedelta(seconds=int(os.environ.get("RINGX_PG_TIMEOUT", "600"))))
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")  # torchrun's LOCAL_RANK; every GPU is visible
    torch.cuda.set_device(device)
    assert a.seq_length % (2 * world_size) == 0
    assert a.head_dim % 8 == 0
    f = load(a.kernel)

    shape = (a.batch_size, a.seq_length, a.num_heads, a.head_dim)
    q, k, v = (torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    dout = torch.randn(shape, device=device, dtype=torch.bfloat16)
    if rank == 0:
        print(f"Algo: {a.kernel}\nBatch size: {a.batch_size}\nSequence length: {a.seq_length}\n"
              f"Number of heads: {a.num_heads}\nHead dimension: {a.head_dim}")
        print(f"# {f.__name__}")
        print(f"ngpus: {world_size}, causal: {a.causal}, batch: {a.batch_size}, seqlen: {a.seq_length}, "
              f"num_heads: {a.num_heads}, head_dim: {a.head_dim}", flush=True)

    def step():
        q.grad = k.grad = v.grad = None
        out = f(q, k, v, causal=a.causal, window_size=(-1, -1), alibi_slopes=None, deterministic=False,
                return_attn_probs=False, group=dist.group.WORLD)
        out.backward(dout)

    step()  # warm-up
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(a.num_iter):
        step()
    end.record()
    torch.cuda.synchronize(device=device)
    sec = begin.elapsed_time(end) / 1000.0
    if rank == 0:
        print(f"{a.num_iter / sec:.6f} iter/s, {sec:.3f} sec")
        print(f"RESULT kernel={a.kernel} causal={int(a.causal)} b={a.batch_size} s={a.seq_length} H={a.num_heads} "
              f"d={a.head_dim} ngpus={world_size} iters={a.num_iter} sec={sec:.4f} "
              f"peak_mem_gib={torch.cuda.max_memory_allocated(device) / 2**30:.1f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
