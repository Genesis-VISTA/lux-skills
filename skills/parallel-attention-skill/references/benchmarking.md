# Benchmarking RingX kernels

Paths are in [jqyin/ringX-attention](https://github.com/jqyin/ringX-attention) (read at `d1e53d7`). Everything below
comes from a static reading of that code, except the "Lux (MI355X)" section, which was run on one node.

## What is upstream

| Path | What it does |
|---|---|
| `benchmark/benchmark_ringX_func.py` | Times one RingX kernel against the three ring-flash-attention baselines on random inputs |
| `benchmark/utils.py` | Parses `log.algo*` files in the current directory into time and TFLOPS/GPU |
| `script/submit_frontier.sh` | Frontier Slurm job: a 4-point batch/sequence sweep for one algorithm |
| `script/setup_dist_vars.sh` | Maps Slurm variables to `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT=29501` |
| `script/plot.ipynb` | The paper's plots |
| `test/test_ringX{1,1o_varlen,2,3}_attn_func.py` | Correctness tests against `flash_attn_func` (see `integration.md`) |

Requirements: PyTorch, FlashAttention 2 (the README says "tested with v2.7"; the paper used 2.6.3), RingX
(`pip install -e .` at the repo root), and `ring_flash_attn` (`pip install ring-flash-attn`, from
[zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention); the paper used v0.1.1). These are
known-working references, not pins.

## `benchmark_ringX_func.py`

| Flag | Default | Meaning |
|---|---|---|
| `--module` | required | Module name. The script runs `importlib.import_module(m)` and then `getattr(module, f"{m}_func")` |
| `--batch_size` | 32 | Batch |
| `--seq_length` | 128 | Sequence length **per GPU** (s/N). Global length = seq_length × world size |
| `--num_heads` | 8 | Heads, used for Q, K and V alike (no GQA) |
| `--head_dim` | 64 | Head dimension; must be a multiple of 8 |
| `--causal` | off | Causal masking |
| `--forward_only` | off | Time the forward only (under `no_grad`); the default times forward+backward |
| `--num_iter` | 10 | Timed iterations |
| `--profile` | off | torch.profiler trace (see below) |

Fixed in the code:

- dtype bf16;
- `seq_length % (2 * world_size) == 0` (asserted);
- `dist.init_process_group("nccl")` from env (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`);
- device `cuda:0`, so **each rank must see exactly one GPU**;
- `group=dist.group.WORLD`, so the CP size is the whole job.

Each kernel gets 1 warm-up fwd+bwd, then `num_iter` iterations timed with CUDA events. Rank 0 prints:

```
Algo: ringX1_attn
Batch size: 32 ... Head dimension: 128
# ringX1_attn_func
ngpus: 64, causal: False, batch: 32, seqlen: 8192, num_heads: 32, head_dim: 128
<x> iter/s, <t> sec
# zigzag_ring_flash_attn_func
...
```

The kernels run in the order ringX, `zigzag_ring_flash_attn_func`, `stripe_flash_attn_func`,
`ring_flash_attn_func`.

### Upstream issues to work around

These come from reading the code. On Lux, the shim (1) worked, the zigzag assert (2) was confirmed, and the
iteration fix (3) was applied (`--num_iter 5`). Items 4 and 5 were not exercised.

1. **Import path.** `submit_frontier.sh` sets `PYTHONPATH=../ringX_attn` and passes `--module ringX1_attn`. The
   modules start with `from .utils import get_default_args` (a relative import), so importing `ringX1_attn` as a
   top-level module should fail. Passing `--module ringX_attn.ringX1_attn` does not help either: the script then
   looks up `ringX_attn.ringX1_attn_func`. A workaround that leaves upstream untouched is one shim module per
   kernel (it works on Lux; the failure of the original form was not tried):

   ```bash
   pip install -e /path/to/ringX-attention
   mkdir -p shims
   for a in ringX1 ringX1o ringX2 ringX2o ringX3 ringX4 ringX4o; do
     echo "from ringX_attn import ${a}_attn_func" > shims/${a}_attn.py
   done
   export PYTHONPATH=$PWD/shims        # then --module ringX1_attn works as the script expects
   ```

2. **Bi-directional runs stop at the zigzag baseline.** `zigzag_ring_flash_attn_func` asserts `causal == True`
   ("zigzag ring is meaningless for causal=False"), and `stripe_flash_attn_func` does the same. Without
   `--causal`, the ringX timing prints and then the script raises before `ring_flash_attn_func`, the bi-directional
   baseline. Use a local copy of the script whose kernel list is `[ringX_attn_func, ring_flash_attn_func]` for
   bi-directional runs.
3. **TFLOPS assumes 5 iterations.** `benchmark/utils.py` hardcodes `iters = 5`, but `--num_iter` defaults to 10. At
   the default, the TFLOPS it prints are half the real value. Pass `--num_iter 5`, or edit `iters` to match. The
   relative ordering of kernels is unaffected.
4. **Baselines get contiguous inputs.** The ring baselines receive the same random per-rank tensors; they are not
   zigzag-arranged. That is fine for timing, because the inputs are random, but the outputs are not comparable.
5. `--profile` needs `--num_iter` ≥ 15, because the schedule is wait=5, warmup=5, active=5. Traces go to
   `./benchmark/logs/<kernel>/rank_<r>/`, relative to the working directory, for TensorBoard. Times printed in a
   profiled run include profiler overhead; do not compare them with unprofiled runs.

## Setting your dimensions

Map the model's attention onto the flags:

| Model | Flag |
|---|---|
| micro-batch per GPU | `--batch_size` |
| context length / CP size | `--seq_length` (must be divisible by 2 × CP size) |
| query heads per TP rank | `--num_heads` |
| hidden / heads | `--head_dim` |
| causal (decoders) | `--causal` |
| dtype | bf16 only (edit `dtype` in `benchmark()` for anything else) |
| CP size | number of ranks in the job (`--ntasks`) |

- **GQA / MQA.** The script uses one head count for Q, K and V. The kernels allocate buffers with `empty_like(k)`,
  and FlashAttention supports fewer KV heads, so GQA is structurally fine. forge passes un-repeated KV to the ring
  kernel. To time it, edit `k`/`v` in a local copy to `num_kv_heads`. TODO: GQA is not exercised by any upstream
  test or benchmark.
- **ViT example.** The paper's bi-directional sweep used 32 heads × 128 and paired batch/sequence points, the
  same as `submit_frontier.sh`:

  ```bash
  batch_sizes=(32 64 128 256); seq_lengths=(8192 4096 2048 1024); num_heads=32; head_dim=128; causal=false
  ```

- **Llama3 8B.** 32 query heads, 8 KV heads, head_dim 128, causal, micro-batch 1. Source:
  `train/configs/xforge/llama3-8b-*.yml` in forge.

Use the CP size and the per-GPU length you will train with. The kernel ranking changes with both (see
`algorithm-selection.md`).

## Running on Frontier (upstream as is)

```bash
cd ringX-attention/script
sbatch submit_frontier.sh ringX1_attn          # the algorithm is the job's only argument; edit the sweep inside
```

The settings below are **Frontier-specific**; do not copy them to another machine.

- `#SBATCH -A stf218 -N 64 -q debug -t 0:20:00`;
- `module use /sw/aaims/crusher/modulefiles; module load xforge` (the AAIMS environment);
- `NCCL_SOCKET_IFNAME=hsn`, `FI_PROVIDER=cxi` and the `FI_CXI_*` variables, `FI_MR_CACHE_MONITOR=disabled`,
  `NCCL_NET_GDR_LEVEL=3`, `NCCL_CROSS_NIC=1`, `NCCL_ALGO=Tree` (Slingshot, aws-ofi-rccl plugin);
- `srun --network=disable_rdzv_get ... --ntasks-per-gpu=1 --gpus-per-node=8 --gpu-bind=closest -c7` (one GCD per
  task, which is what makes `cuda:0` correct).

Output is appended to `script/log.algo-<algo>_n<nodes>_<jobid>`.

## Adapting to another cluster

The parts that carry over are the Python script, one rank per GPU, and `setup_dist_vars.sh`. Replace the rest:

| Concern | Frontier | What to do elsewhere |
|---|---|---|
| Scheduler | Slurm `sbatch`, `-A stf218 -q debug` | Your account, partition and QoS; PBS/LSF need their own rank variables in place of `SLURM_PROCID`/`SLURM_LOCALID`/`SLURM_NTASKS` |
| Launcher | `srun`, 8 tasks/node, `setup_dist_vars.sh` | `srun` with one task per GPU, or `torchrun --nproc_per_node=<gpus>` (it sets RANK/WORLD_SIZE/MASTER_* itself) |
| One GPU per rank | `--gpu-bind=closest --ntasks-per-gpu=1` | If every rank sees all GPUs, set `HIP_VISIBLE_DEVICES=$LOCAL_RANK` (ROCm) or `CUDA_VISIBLE_DEVICES=$LOCAL_RANK` (CUDA) before Python, because the script always uses `cuda:0` |
| GPU stack | ROCm 6.1.3, MI250X | ROCm: a ROCm PyTorch wheel and a ROCm FlashAttention build. CUDA: the standard wheels. The `torch.cuda` / `"nccl"` names work on both |
| Collectives | RCCL + aws-ofi-rccl over Slingshot (`cxi`) | RCCL on AMD, NCCL on NVIDIA, each with your fabric's plugin and interface variables |
| `MASTER_ADDR` | first hostname | Use an address on the interface the job's traffic goes over |

### Lux (MI355X), tested on one node

Run on 2026-10-23, on one node (8x MI355X): RingX `d1e53d7`, ring-flash-attn 0.1.8, inside the Primus image
`/lustre/orion/world-shared/stf218/junqi/primus/primus-v26.5.sif`. That image has torch 2.12 (ROCm 7.15),
FlashAttention 2.8.3 with forward and backward kernels, and RCCL 2.30.4. No pip install into the image is needed.
Put the packages on `PYTHONPATH`:

```bash
W=$PWD                                 # holds ringX-attention/ (git clone), pydeps/ and shims/
apptainer exec $SIF python -m pip install --no-deps --target $W/pydeps ring-flash-attn   # via the ORNL proxy
export APPTAINERENV_PYTHONPATH=$W/ringX-attention:$W/pydeps:$W/shims   # ringX_attn, ring_flash_attn, shims
```

Lux-specific requirements found on the run:

- **Request the GPUs.** `#SBATCH --gpus-per-node=8` is required, together with `--exclusive`. Without it, the
  job's steps see no GPU ("ProcessGroupNCCL is only supported with GPUs, no GPUs found!"), even on an exclusive node.
- **rng_state for the baselines.** ring-flash-attn 0.1.8 calls `_flash_attn_backward` without `rng_state`. FA
  2.8.3 on ROCm then fails ("Cannot access data pointer of Tensor that doesn't have storage"), as a 1-GPU check
  confirmed. RingX passes `torch.empty(2)` and is not affected. Patch the baselines in a local copy of the
  benchmark, before `from ring_flash_attn import ...`:

  ```python
  import torch, flash_attn.flash_attn_interface as fai
  import ring_flash_attn.ring_flash_attn as _r, ring_flash_attn.zigzag_ring_flash_attn as _z, ring_flash_attn.stripe_flash_attn as _s
  _orig = fai._flash_attn_backward
  def _flash_attn_backward(dout, q, k, v, out, softmax_lse, dq, dk, dv, dropout_p, softmax_scale, causal,
                           window_size_left, window_size_right, softcap, alibi_slopes, deterministic, rng_state=None):
      if rng_state is None:
          rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)
      return _orig(dout, q, k, v, out, softmax_lse, dq, dk, dv, dropout_p, softmax_scale, causal,
                   window_size_left, window_size_right, softcap, alibi_slopes, deterministic, rng_state)
  for m in (_r, _z, _s):
      m._flash_attn_backward = _flash_attn_backward   # explicit signature: get_default_args must see the names
  ```

- The import shim (issue 1) and the bi-directional kernel list (issue 2) are needed as described.
- The FA 2.8.3 private functions are torch custom ops whose signature reads `(*args, **kwargs)`. RingX's
  `get_default_args` still gets the full argument list through `_init_fn`.
- Every rank prints a "c10d::broadcast_: an autograd kernel was not registered" warning in backward. It is harmless:
  RingX does its own backward, and the gradients match the reference (`integration.md`).

The launcher used the NCCL variables of this repo's `primus-skill` job (`hpc_jobs/primus-smoke/job.lux.slurm`):

```bash
#SBATCH -A <account> -p batch -N 1 --exclusive --gpus-per-node=8
IFACE=bond0
export MASTER_ADDR=$(ip -4 addr show $IFACE | grep -oP '(?<=inet )[0-9.]+' | head -1)   # 1 node; multi-node: query the first host
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
# APPTAINERENV_*, not --env, which comma-splits NCCL_IB_HCA
export APPTAINERENV_NCCL_SOCKET_IFNAME=$IFACE APPTAINERENV_GLOO_SOCKET_IFNAME=$IFACE
export APPTAINERENV_NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
export APPTAINERENV_NCCL_IB_GID_INDEX=1 APPTAINERENV_USING_AINIC=1 APPTAINERENV_RCCL_USE_AMD_SMI_LIB=1
srun -N1 -n8 --ntasks-per-node=8 --cpus-per-task=16 bash -c "
  export APPTAINERENV_RANK=\$SLURM_PROCID APPTAINERENV_LOCAL_RANK=\$SLURM_LOCALID APPTAINERENV_WORLD_SIZE=\$SLURM_NTASKS
  export APPTAINERENV_MASTER_ADDR=$MASTER_ADDR APPTAINERENV_MASTER_PORT=$MASTER_PORT
  export APPTAINERENV_HIP_VISIBLE_DEVICES=\$SLURM_LOCALID      # benchmark only: it uses cuda:0. Omit for the tests (cuda:rank)
  exec apptainer exec --rocm --bind /lustre/orion $SIF python benchmark_ringX_func_lux.py \
    --module ringX4_attn --causal --batch_size 1 --seq_length 16384 --num_heads 32 --head_dim 128 --num_iter 5" \
  >> log.algo-ringX4_attn_n1_$SLURM_JOB_ID
```

`--cpus-per-task=16` (128 / 8) worked. The RCCL defaults were left alone; no `NCCL_ALGO`/GDR tuning was tried.

**Measured, one node, 8 GPUs, bf16, fwd+bwd, 5 timed iterations, one run each** (TFLOPS/GPU from
`benchmark/utils.py`). These are single measurements. Repeat them before relying on them.

| Attention | Point (b, s/N, heads × d) | Kernel | Time (5 it) | TFLOPS/GPU |
|---|---|---|---|---|
| causal | 1, 16384, 32 × 128 | **ringX3** | 1.532 s | 366 |
| | | ringX2 | 1.551 s | 362 |
| | | ringX4 | 1.678 s | 334 |
| | | ring | 1.65 s | 340 |
| | | zigzag ring | 1.72–1.92 s | 292–326 |
| | | stripe | 1.85–1.90 s | 296–303 |
| bi-directional | 32, 8192, 32 × 128 | **ringX1** | 13.90 s | 342 |
| | | ring | 36.92 s | 129 |
| bi-directional | 128, 2048, 32 × 128 | **ringX1** | 6.14 s | 258 |
| | | ring | 26.86 s | 59 |
| causal | 64, 4096, 32 × 128 | **ringX3** | 4.56 s | 579 |
| | | ringX4 | 7.28 s | 363 |
| | | ringX2 | 9.10 s | 290 |
| | | ring | 14.1–21.9 s | 121–187 |
| | | zigzag ring | 21.1–27.3 s | 97–125 |
| | | stripe | 20.1–35.7 s | 74–132 |
| bi-directional | 64, 8192, 32 × 128 | **ringX1** | 27.35 s | 347 |
| | | ring | 45.77 s | 208 |

(The baseline ranges are over the three causal runs, which each re-time the baselines.)

At batch 64, the causal point had to drop to 4096 tokens/GPU. At b=64 with 16384 tokens/GPU (131K global), ringX3
runs out of memory all-gathering KV (a 128 GiB allocation, on 288 GB GPUs), and ringX2/ringX4 fail with "CUDA
driver error: 1" inside `update_out_and_lse` (presumably memory too). The baselines vary up to 1.7x between
repeats at b=64, so repeat them.

**The TFLOPS column is not attention FLOPs.** `benchmark/utils.py` counts 3 × (8·b·S·h² + 4·b·S²·h). That includes
projection FLOPs (8·b·S·h²) the kernel does not do, and it does not halve the S² term for causal attention. For a
like-for-like number, count 4·b·S²·H·d × 3.5 (forward + 2.5x backward), halved for causal, and compare with
single-GPU `flash_attn_func` on the same image (one MI355X, FA 2.8.3, bf16, fwd+bwd):

| `flash_attn_func`, one GPU | Real attention TFLOPS |
|---|---|
| causal, b=1, s=16384 | 374 |
| causal, b=64, s=4096 | 246 |
| causal, b=1, s=131072 | 448 |
| bi-directional, b=32 or 64, s=8192 | 464 |
| bi-directional, b=8, s=65536 | 471 |

Counted this way, on 8 GPUs:
- ringX1 at b=32/64, s/N=8192 reaches 354/360 TFLOPS/GPU, about 77% of the 464 ceiling (ring: 133/215).
- ringX3 reaches 201 at causal b=1, s/N=16K, and 270 at b=64, s/N=4K.

For bi-directional attention, ringX1 is already at a plateau at b=32 (342 → 347 by the utils formula). Batch
size helps the causal kernels more.

On one node, ringX1 is 2.7–4.4x faster than ring attention for bi-directional attention. For causal attention at
this point, ringX2/ringX3 are ~7% faster than ring and 11–20% faster than zigzag. ringX4 does not win on a single
MI355X node, which differs from the "one node → ringX4" heuristic. Benchmark at your own point before choosing.

### Lux reference results: scaling to 16 nodes, sequence sweep, hardware FLOP check

Run on 2026-10-23 with the job `hpc_jobs/ringx-bench` (`scripts/bench.sh`, `NODES=1..16`), same software as above.
The context-parallel group is every GPU of the job (8 per node, up to 128). bf16, fwd+bwd, 32 heads × 128, 5 timed
iterations, one run each. The numbers are **real** TFLOPS/GPU (attention FLOPs only, halved for causal); `%` is of
single-GPU `flash_attn_func` at the same per-GPU shape. Outputs are in `runs/ringx.<jobid>/summary.txt`.

**Weak scaling at 64K tokens/GPU, batch 1** (jobs 364897–364901; global sequence 512K to 8M tokens):

| Nodes (GPUs) | ringX2 | ringX3 | ringX4 | ring | zigzag | stripe | best ringX / best baseline |
|---|---|---|---|---|---|---|---|
| 1 (8) | 235 | **441** (102%) | 407 | 231 | 400 | 345 | 1.10x |
| 2 (16) | 191 | **384** | 333 | 120 | 343 | 167 | 1.12x |
| 4 (32) | 225 | **445** | 410 | 96 | 306 | 112 | 1.46x |
| 8 (64) | 217 | **438** | 403 | 111 | 299 | 113 | 1.47x |
| 16 (128) | 217 | OOM | **398** (92%) | 154 | 278 | timed out (40 min) | 1.44x |

| Nodes (GPUs) | bi-directional ringX1 | ring | ringX1 / ring |
|---|---|---|---|
| 1 (8) | **446** (95%) | 434 | 1.03x |
| 2 (16) | **386** | 343 | 1.12x |
| 4 (32) | **441** | 321 | 1.37x |
| 8 (64) | **433** | 331 | 1.31x |
| 16 (128) | **432** (93%) | 336 | 1.29x |

**Weak scaling at 16K tokens/GPU** (jobs 364848–364853; causal batch 1, bi-directional batch 8):

| Nodes (GPUs) | ringX2 | ringX3 | ringX4 | ring | zigzag | stripe | ringX1 (bidir) | ring (bidir) |
|---|---|---|---|---|---|---|---|---|
| 1 (8) | **199** | 198 | 189 | 189 | 164 | 167 | **347** | 310 |
| 2 (16) | 116 | **238** | 193 | 122 | 135 | 129 | **176** | 72 |
| 4 (32) | 182 | **398** | 300 | 128 | 140 | 133 | **335** | 74 |
| 8 (64) | 177 | **393** | 303 | 19 | 139 | 50 | **325** | 78 |
| 16 (128) | 170 | **392** | 300 | failed | 138 | 26 | **321** | timed out |

What the scaling shows:

- **The RingX kernels weak-scale almost flat from 1 to 16 nodes.** At 64K, ringX4 goes 407 → 398 and ringX1
  446 → 432. The ring-attention baselines collapse across nodes: ring and stripe fall to 20–150, and zigzag holds
  only about 140 (16K) or 280 (64K).
- **2 nodes is a dip for every kernel** (ringX1 at 16K: 347 → 176 → 335). Do not extrapolate from 2 nodes.
- **ringX3 is fastest wherever it fits, and its memory grows with the GPU count.** Its peak is the full gathered
  KV: at 64K/GPU, 39, 71, 135 and 263 GiB on 1, 2, 4 and 8 nodes, and out of memory on 16 (288 GB GPUs). At 16K/GPU
  it reaches 130 GiB on 16 nodes. The measured peaks fit about 16 × b × S × H × d bytes, with S the global
  sequence (bf16 K and V, fp32 dK and dV, and buffers).
- **ringX4 is the fallback beyond ringX3's memory limit.** It stays at 12.5 GiB (64K) and 3.1 GiB (16K) at every
  node count, at 87–92% of ringX3's speed at 64K/GPU and about 75% at 16K/GPU. On Lux: ringX3 while
  16·b·S·H·d bytes fits in GPU memory next to the model, then ringX4.
- **ringX2 stays at about half of the single-GPU rate** at every scale (see the hardware check below for why).
- **Longer sequences per GPU scale better** for every kernel: per-GPU compute grows as s², the messages as s.

**Sequence-length sweep on one node at constant tokens per GPU × batch** (job 364868, causal, 8 GPUs):

| Tokens/GPU × batch | ringX2 | ringX3 | ringX4 | 1-GPU `flash_attn_func` |
|---|---|---|---|---|
| 4K × 16 | 131 | **163** (67%) | 131 | 244 |
| 8K × 8 | 173 | **260** (81%) | 215 | 323 |
| 16K × 4 | 206 | **363** (93%) | 318 | 391 |
| 32K × 2 | 224 | **426** (97%) | 393 | 437 |
| 64K × 1 | 237 | **443** (104%) | 412 | 425 |
| 128K × 1 | 245 | **456** (101%) | 434 | 451 |

Bi-directional ringX1 on the same sweep: 273 (63%) at 4K × 16, 399 (86%) at 16K × 4, 448 (95%) at 64K × 1, and 460
(97%) at 128K × 1. Throughput rises with the per-GPU sequence for two reasons. FlashAttention itself is faster at
long sequences (244 → 451 on one GPU), and the communication share falls. **For the same tokens per GPU, prefer a
longer sequence and a smaller batch.** At 64K and above, ringX3 and ringX1 are at the single-GPU rate (95–104%).

**Hardware FLOP check with rocprofv3** (one node, `scripts/hwflops/run.sh`). The MI355X counter
`SQ_INSTS_VALU_MFMA_MOPS_BF16` counts matrix-core work in units of 512 FLOPs. A bf16 GEMM of known size calibrates
this exactly (1.000). Counted over one warm-up and one timed step, per GPU:

| Workload | hardware FLOPs / `real` formula |
|---|---|
| bf16 GEMM 8192³, 1 GPU | 1.000 |
| `flash_attn_func` fwd+bwd, bi-directional, s=16K, 1 GPU | 1.000 |
| `flash_attn_func` fwd+bwd, causal, s=16K, 1 GPU | 1.008 |
| ringX3, ringX4, zigzag: causal, 1:16384, 8 GPUs | 1.001 |
| ringX1, ring: bi-directional, 8:8192, 8 GPUs | 1.000 |
| ringX2: causal, 1:16384, 8 GPUs | 0.84 (one rank; see below) |

- **The `real` count is what the hardware executes.** The FlashAttention backward is 2.5x the forward (it
  recomputes QKᵀ), as the formula's × 3.5 assumes. Causal masking costs only 0.1–0.8% extra, for the diagonal
  tiles. The context-parallel kernels add no recompute and no redundant blocks.
- **`utils` overstates causal by 1.7–1.8x.** Quote `real` (and its `%`) for Lux results.
- **ringX2 is not load-balanced.** Only one rank's counters were saved per run (all ranks wrote the same file
  name), and that rank did 0.84x the mean work. Without zigzag sharding, the causal work per rank is uneven, so the
  step waits on the busiest rank. This would explain its ~50% efficiency at every scale. It is inferred from one
  rank and not yet confirmed per rank.
- **Use `_rocm_sdk_core/bin/rocprofv3`** in the Primus image. The `rocprofv3` on `PATH` (`_rocm_sdk_devel`) loads a
  second `libLLVM` next to torch's, and the process aborts with "Option 'spirv-expand-step' registered more than
  once" and then hangs.

## Collective bandwidth

The ranking of ringX3 against ringX4, and of ringX1/2 against ring, follows from how fast broadcast/reduce run
against all-gather/reduce-scatter at your message size and GPU count (paper Fig. 7). Measure them with
[nccl-tests](https://github.com/NVIDIA/nccl-tests) (CUDA) or rccl-tests (ROCm). rccl-tests now lives in
[ROCm/rocm-systems](https://github.com/ROCm/rocm-systems), under `projects/rccl-tests`; the old ROCm/rccl-tests
repo is deprecated.

```bash
# build (MPI launcher); rccl-tests: HIP_HOME=<rocm> NCCL_HOME=<rccl>, nccl-tests: CUDA_HOME=... NCCL_HOME=...
make MPI=1 MPI_HOME=<mpi> HIP_HOME=<rocm> NCCL_HOME=<rccl>
# one process per GPU (-g 1), 1 MB to 2 GB, doubling; -n iterations (default 20), -w warm-up (default 1)
for op in broadcast reduce all_gather reduce_scatter all_reduce; do
  mpirun -np <gpus> -N 8 ./build/${op}_perf -b 1M -e 2G -f 2 -g 1
done
```

TODO: which MPI to use on Lux has not been determined.

Message sizes per kernel step, in bytes, with b = batch, s_l = tokens per GPU, H = heads, d = head_dim, and bf16
inputs (from the kernels' buffers):

| Kernel | Forward | Backward |
|---|---|---|
| ringX1/2 | broadcast Q: 2·b·s_l·H·d. reduce of the fp32 partial output: 4·b·s_l·H·d, plus the lse | broadcast KV: 4·b·s_l·H·d. reduce of fp32 dKV: 8·b·s_l·H·d |
| ringX3 | all-gather KV: 4·b·s_l·H·d per rank | reduce-scatter of dKV |
| ringX4 | broadcast KV: 4·b·s_l·H·d | reduce of fp32 dKV: 8·b·s_l·H·d |

Read **busbw**, not algbw, when comparing collectives. busbw scales algbw by 2(n−1)/n for all-reduce, (n−1)/n for
all-gather and reduce-scatter, and 1 for broadcast and reduce (nccl-tests `doc/PERFORMANCE.md`). Run the tests
within one node (8 GPUs) and at your multi-node CP size: the crossover between the two collective pairs is what
decides between ringX4 and ringX3.

## Reading the results

```bash
cd script            # or wherever the log.algo* files are
python ../benchmark/utils.py
# Batch size: 32, Sequence length: 8192
#   Function: ringX1_attn
#     GPUs: 64, Time: <t> sec, TFLOPS: <x>
```

- **Time** is the total for `num_iter` timed iterations (1 warm-up dropped). Per-iteration time = time / num_iter.
  The paper averaged 10 iterations after 1 warm-up for the kernel results.
- **TFLOPS/GPU** = 3 × F / (time / 5) / 10¹² / ngpus. F = 8·b·S·h² + 4·b·S²·h, with S = seq_length × ngpus and
  h = heads × head_dim. That is the paper's 4bsh²(s/h + 2). The 3 counts fwd+bwd; see issue 3 for the 5. The formula
  is the same for causal and non-causal runs, so compare TFLOPS only between runs of the same kind.
- **Compare kernels at the same point.** Use the same batch, s/N, heads and GPU count. The baselines in the same log
  (zigzag/stripe for causal, ring for bi-directional) are the reference.
- **Repeat.** Run each point at least twice, ideally in different allocations, because node placement and network
  load change collective times. Run with `--forward_only` too if inference matters.
- **Decide.** Pick the kernel with the lowest fwd+bwd time at the CP size and per-GPU length you will train with. If
  two are within noise, prefer the one with lower memory (ringX4 over ringX3, ringX2 over ringX3). Then confirm the
  choice end to end (`integration.md`): kernel gains shrink in a full training step.
- **Where the time goes.** Use `--profile` (with `--num_iter 15` or more) and look at the communication share. The
  paper measured ~80% for ring/stripe and ~50% for ringX1/ringX3.
