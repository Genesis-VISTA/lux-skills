# Benchmarking RingX kernels

Paths are in [jqyin/ringX-attention](https://github.com/jqyin/ringX-attention) (read at `d1e53d7`). Everything below
comes from a static reading of that code. None of it has been run on Lux.

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

Every item here comes from reading the code; none was confirmed by running it.

1. **Import path.** `submit_frontier.sh` sets `PYTHONPATH=../ringX_attn` and passes `--module ringX1_attn`. The
   modules start with `from .utils import get_default_args` (a relative import), so importing `ringX1_attn` as a
   top-level module should fail. Passing `--module ringX_attn.ringX1_attn` does not help either: the script then
   looks up `ringX_attn.ringX1_attn_func`. A workaround that leaves upstream untouched is one shim module per
   kernel (TODO: verify on a run):

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

### Lux (MI355X), adapted but not yet run

The values below come from this repo's `primus-skill` job (`hpc_jobs/primus-smoke/job.lux.slurm`), which runs RCCL
over Lux's fabric:

```bash
# inside an allocation: salloc -A <account> -p batch -N <nodes> --exclusive
IFACE=bond0
export MASTER_ADDR=$(srun -N1 -n1 -w "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)" \
  bash -c "ip -4 addr show $IFACE | grep -oP '(?<=inet )[0-9.]+'" | tail -1)
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export NCCL_SOCKET_IFNAME=$IFACE GLOO_SOCKET_IFNAME=$IFACE
export NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
export NCCL_IB_GID_INDEX=1 USING_AINIC=1 RCCL_USE_AMD_SMI_LIB=1
export PYTHONPATH=$PWD/shims                       # the import shim above
srun -N "$SLURM_NNODES" --ntasks-per-node=8 --cpus-per-task=16 bash -c '
  export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$SLURM_NTASKS HIP_VISIBLE_DEVICES=$SLURM_LOCALID
  python benchmark_ringX_func.py --module ringX4_attn --causal --batch_size 1 --seq_length 16384 \
    --num_heads 32 --head_dim 128 --num_iter 5' >> log.algo-ringX4_attn_n${SLURM_NNODES}_${SLURM_JOB_ID}
```

TODO before this runs:

- **A Python environment with torch (ROCm), flash_attn built for gfx950, ringX_attn and ring_flash_attn.**
  Candidates in this repo:
  - the Primus image `primus-v26.5.sif`. TODO: check whether it contains `flash_attn` by running
    `apptainer exec --rocm <sif> python -c "import flash_attn"`;
  - the vLLM image, which builds FlashAttention at `FA_HASH 145b1010` (see `vllm-skill`'s Dockerfile). It is a
    podman image, and nobody has checked whether its FlashAttention build provides the backward kernels.
  - Inside apptainer, pass the NCCL variables as `APPTAINERENV_*`, not with `--env`, which comma-splits
    `NCCL_IB_HCA` (a note in the primus job).
- **CPU binding.** `--cpus-per-task=16` (128 / 8 ranks) is an assumption. primus-skill notes that without a
  `--cpus-per-task` Slurm confines the ranks to one CPU.
- **NCCL_ALGO, GDR and the other Frontier tunings** have no tested Lux equivalent. Leave RCCL's defaults.

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
