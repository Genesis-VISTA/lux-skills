#!/usr/bin/env python3
"""
Move Inkling's sconv channel shards with block permutes instead of element
transposes around the TP reduce-scatter / all-gather.

THE BUG AS IT PRESENTS

_sconv_add_norm (vllm/models/inkling/amd/model.py) runs twice per decoder layer:
RS(delta, dim=-1) -> sconv on the channel shard -> AG(dim=-1). CudaCommunicator
implements dim=-1 as movedim(0, -1).contiguous() before the RS, movedim back
plus .contiguous() after it, and a movedim/reshape copy after the AG: three
element-granular transposes of a (T, 4096) bf16 tensor per site. They run as
at::native::elementwise_kernel_manual_unroll<128, 8> (strided, not vectorized),
11.9 us per call at c64 in the 2-node profile (f61hNXK1vtUo, 5120 calls), with
42 sites per stage pass and more in the drafter.

THE FIX

Rank r's shard is channels [r*C, (r+1)*C), C = hidden / tp. Viewing delta as
(T, tp, C) and permuting to (tp, T, C) puts each rank's shard in one contiguous
dim-0 slab, so RS(dim=0) returns the (T, C) shard directly and AG(dim=0) returns
(tp, T, C), which one permute turns back into (T, hidden). The copies move
C*2 = 1 KiB rows, which the vectorized copy kernel handles. The shard
assignment, and so the sconv weight and state sharding, is unchanged. RS/AG on
dim 0 skip the movedim calls in CudaCommunicator. The result equals the old
path up to the RCCL summation order (the shard offsets inside the RCCL buffer
change).
"""
import argparse
import importlib.util
import os
import sys

TARGET = "vllm/models/inkling/amd/model.py"
MARKER = "patch inkling_sconv_rsag_blocks.py"

OLD = '''    # RCCL RS -> shard sconv -> AG -> fused add(+rmsnorm).
    shard = tensor_model_parallel_reduce_scatter(delta, dim=-1)
    shard = sconv(shard.contiguous(), positions)
    full = tensor_model_parallel_all_gather(shard, dim=-1)
'''

NEW = '''    # RCCL RS -> shard sconv -> AG -> fused add(+rmsnorm).
    # (patch inkling_sconv_rsag_blocks.py) Rank r's shard is channels
    # [r*C, (r+1)*C); permuting (T, tp, C) -> (tp, T, C) makes it a dim-0 slab,
    # so RS/AG run on dim 0 and the copies move whole C-wide rows instead of
    # transposing element by element.
    tp = get_tensor_model_parallel_world_size()
    if tp > 1 and delta.dim() == 2:
        t, h = delta.shape
        c = h // tp
        slabs = delta.view(t, tp, c).transpose(0, 1).reshape(tp * t, c)
        shard = tensor_model_parallel_reduce_scatter(slabs, dim=0)
        shard = sconv(shard, positions)
        full = tensor_model_parallel_all_gather(shard, dim=0)
        full = full.view(tp, t, c).transpose(0, 1).reshape(t, h)
    else:
        shard = tensor_model_parallel_reduce_scatter(delta, dim=-1)
        shard = sconv(shard.contiguous(), positions)
        full = tensor_model_parallel_all_gather(shard, dim=-1)
'''

SITES = [("_sconv_add_norm RS/AG", OLD, NEW)]


def find_target():
    """Locate the model module inside whichever site-packages vLLM lives in.

    Deliberately does NOT `import vllm` (see rocm_aiter_ray_env_refresh.py):
    importing it touches the GPU at module scope and fails in a build container.
    """
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        sys.exit("vllm is not installed; run this inside the container")
    root = os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))
    path = os.path.join(root, TARGET)
    if not os.path.exists(path):
        sys.exit(f"expected {path} to exist; vLLM layout changed")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report status without writing")
    ap.add_argument("--path", help="patch this file instead of the installed "
                    "vLLM's (e.g. a copy bind-mounted over the image file)")
    args = ap.parse_args()

    path = args.path or find_target()
    with open(path) as fh:
        src = fh.read()

    if MARKER in src:
        print(f"already patched: {path}")
        return 0

    patched = src
    problems = []
    for name, old, new in SITES:
        n = patched.count(old)
        if n != 1:
            problems.append(f"{name} (matched {n}x, expected 1)")
            continue
        patched = patched.replace(old, new, 1)

    if problems:
        print(f"ERROR: patch target not found in {path}:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print("vLLM changed Inkling's _sconv_add_norm; check whether it still "
              "does RS(dim=-1) -> sconv -> AG(dim=-1), then re-derive or retire "
              "this patch", file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {path}")
        return 0

    with open(path, "w") as fh:
        fh.write(patched)
    print(f"patched: {path}  (Inkling sconv RS/AG on dim-0 slabs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
