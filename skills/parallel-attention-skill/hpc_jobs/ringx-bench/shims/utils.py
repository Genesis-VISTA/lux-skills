# log/set_seed as in zhuzilin/ring-flash-attention test/utils.py (MIT), which RingX's tests import; RingX ships no
# test/utils.py.
import random
import torch
import torch.distributed as dist


def set_seed(rank, seed=42):
    seed = rank + seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def log(msg, a, rank0_only=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank0_only:
        if rank == 0:
            print(f"{msg}: max {a.abs().max().item():.3g}, mean {a.abs().mean().item():.3g}", flush=True)
        return
    for i in range(world_size):
        if i == rank:
            if rank == 0:
                print(f"{msg}:")
            print(f"[{rank}] max {a.abs().max().item():.3g}, mean {a.abs().mean().item():.3g}", flush=True)
        dist.barrier()
