---
name: parallel-attention-skill
description: Choose, benchmark and integrate RingX context-parallel attention (ringX1 to ringX4, jqyin/ringX-attention) for long-sequence training on GPU clusters such as OLCF Lux or Frontier. Use when asked which context-parallel attention algorithm fits a model and machine, how to benchmark RingX kernels against ring or zigzag ring attention, or how to replace scaled_dot_product_attention or flash_attn_func with RingX in a ViT or GPT training loop.
---

# Parallel attention with RingX

RingX ([jqyin/ringX-attention](https://github.com/jqyin/ringX-attention), package `ringX_attn`) is a set of
context-parallel (CP) attention kernels. Each GPU holds a shard of the sequence, and the kernels exchange Q or KV
with collectives (broadcast, reduce, all-gather, reduce-scatter) instead of ring attention's point-to-point
send/recv. They are built on FlashAttention 2 and replace `scaled_dot_product_attention` / `flash_attn_func`
directly: inputs are `(batch, seqlen_local, nheads, head_dim)` plus a CP process group.

This skill has three parts, each in its own reference file:

1. **Choose an algorithm** from the table below → [`references/algorithm-selection.md`](references/algorithm-selection.md)
2. **Benchmark the kernels** with your dimensions on your machine → [`references/benchmarking.md`](references/benchmarking.md)
3. **Integrate the winner** into a training script → [`references/integration.md`](references/integration.md)

RingX is not vendored: it is installed from upstream. The job scripts upstream are written for Frontier (MI250X,
Slingshot, `cxi`). On Lux (8x MI355X per node, ROCm/RCCL, Slurm partition `batch`), use the job
`parallel-attention-skill/ringx-bench` (below), which carries the adaptations in `references/benchmarking.md`. All
four kernels were run on Lux (2026-10-23, Primus image): correctness on one node, and the kernel benchmark from 1
to 16 nodes (128 GPUs). See "Reference results on Lux" below.

## Choosing an algorithm

```
bi-directional attention (ViT, BERT, encoders)?
 └─ yes → ringX1 (pipelined variant ringX1o)
causal attention (GPT, decoders)?
 ├─ one node, or a high-bandwidth, homogeneous network        → ringX4
 ├─ medium context (< ~100K tokens) that fits on a few nodes  → ringX2
 └─ many nodes / millions of tokens, and full KV fits per GPU → ringX3
then: benchmark the candidates (and ring/zigzag ring) at YOUR batch, sequence, heads and GPU count
```

| Algorithm | Attention | Communication | Load balancing | Context length | Best for |
|---|---|---|---|---|---|
| `ringX1` | bi-directional | broadcast(Q), all-reduce(lse), reduce(num, den) | n/a (pipelined) | unlimited | ViT/BERT, large Q/K/V tensors |
| `ringX2` | causal | same as ringX1 | unbalanced | unlimited | medium contexts (< ~100K) on a few nodes |
| `ringX3` | causal | all-gather(KV), reduce-scatter(dK, dV) | zigzag | bounded by per-GPU memory | many nodes, millions of tokens |
| `ringX4` | causal | broadcast(KV), reduce(dK, dV) | zigzag | unlimited | a single node or a high-bandwidth homogeneous network |

This is a heuristic, not a rule. Which kernel is fastest depends on collective bandwidth and message size, and those
change with the network, node count and tensor shape. The final choice comes from benchmarking with your own
dimensions (part 2).

Reported results (SC '25 paper, on Frontier):

- up to **3.4x** kernel speedup over ring attention;
- **~1.5x** end-to-end speedup for a ViT (climate) and for GPT (Llama3 8B);
- **38% MFU** for Llama3 8B at 1M tokens on 4,096 GPUs.

Evidence, the communication/memory table and the bandwidth crossover are in `references/algorithm-selection.md`.

## Quickstart

```bash
git clone https://github.com/jqyin/ringX-attention && cd ringX-attention
pip install -e .                                 # package ringX_attn; needs torch + flash_attn already installed
pip install ring-flash-attn                      # baselines the benchmark imports (ring_flash_attn); the paper used v0.1.1
# kernel benchmark (Frontier script; see references/benchmarking.md for Lux and other clusters)
cd script && sbatch submit_frontier.sh ringX1_attn
python ../benchmark/utils.py                     # parses log.algo* here into time and TFLOPS/GPU
```

In a model (bi-directional; for causal pass `causal=True` and zigzag-shard the sequence, see part 3):

```python
from ringX_attn import ringX1_attn_func
out = ringX1_attn_func(q, k, v, causal=False, group=cp_group)   # q,k,v: (b, s/N, h, d), same layout as flash_attn_func
```

## Reference results on Lux

Measured on 2026-10-23 (8x MI355X per node, bf16, fwd+bwd, 32 heads × 128, one run each). The numbers are real
attention TFLOPS/GPU, with causal attention halved. The full tables and the job IDs are in "Lux reference results"
in `references/benchmarking.md`.

| 64K tokens/GPU, batch 1 | 1 node | 4 nodes | 16 nodes (8M tokens) | vs best baseline at 16 nodes |
|---|---|---|---|---|
| causal ringX3 | **441** | **445** | out of memory | n/a |
| causal ringX4 | 407 | 410 | **398** | 1.44x (zigzag 278) |
| causal ringX2 | 235 | 225 | 217 | |
| bi-directional ringX1 | 446 | 441 | 432 | 1.29x (ring 336) |

- **RingX weak-scales almost flat to 128 GPUs,** while ring, zigzag and stripe fall off once a job spans nodes. At
  16K tokens/GPU, ringX3 is 2.8x the best baseline on 4–16 nodes, and ringX1 is 4.1–4.5x ring on 4–8 nodes.
- **On Lux, choose ringX3 while its gathered KV fits,** about 16·b·S·H·d bytes for global sequence S (263 GiB at 8
  nodes × 64K). Beyond that, use ringX4: 12.5 GiB at every scale and 87–92% of ringX3's speed at 64K/GPU. ringX2 is
  not load-balanced and stays at about half speed.
- **For the same tokens per GPU, a longer sequence with a smaller batch is faster.** On one node, causal ringX3
  goes from 163 (4K × 16) to 443 (64K × 1) TFLOPS/GPU. At 64K and above, ringX3 and ringX1 run at 95–104% of
  single-GPU `flash_attn_func`.
- **The `real` FLOP count matches the hardware.** rocprofv3 MFMA counters agree with it within 0.1% for ringX1,
  ringX3, ringX4, ring and zigzag. RingX's `utils` number overstates causal by 1.7–1.8x.

## Running through VISTA

`hpc_jobs/ringx-bench/` follows VISTA's HPC job format. In a VISTA project, the agent runs it as
`parallel-attention-skill/ringx-bench`, on Lux only. One job runs the correctness tests and then the kernel
benchmark at the points given; each point is `BATCH:TOKENS_PER_GPU`.

```text
# correctness + the default points (causal 1:16384, bi-directional 32:8192), ~8 min once running
submit_hpc_job(job="parallel-attention-skill/ringx-bench", cluster="lux")
# your model's shape: e.g. causal, 40 heads x 128, two splits of 64K tokens per GPU, 3 runs each
submit_hpc_job(job="parallel-attention-skill/ringx-bench", cluster="lux", duration="2:00:00",
               script_args="MODE=bench CAUSAL_POINTS=4:16384,16:4096 BIDIR_POINTS= NUM_HEADS=40 REPEATS=3")
get_hpc_job_outputs(job_id=<id>, cluster="lux", files=["summary.txt", "results.json"])
```

| Key | Default | Meaning |
|---|---|---|
| `MODE` | `all` | `correctness` or `bench` to run only one part |
| `TESTS` | `ringX1,ringX2,ringX3,ringX4` | correctness tests against single-GPU `flash_attn_func` |
| `CAUSAL_KERNELS`, `BIDIR_KERNELS` | `ringX2,ringX3,ringX4`, `ringX1` | RingX kernels to time (also `ringX2o`, `ringX4o`, `ringX1o`) |
| `CAUSAL_POINTS`, `BIDIR_POINTS` | `1:16384`, `32:8192` | `BATCH:TOKENS_PER_GPU,...`; empty skips that attention type |
| `BASELINES` | 1 | also time ring, zigzag ring and stripe (ring only for bi-directional) |
| `CEILING` | 1 | time single-GPU `flash_attn_func` at the same per-GPU shapes |
| `NUM_HEADS`, `HEAD_DIM` | 32, 128 | attention shape (Llama3-8B) |
| `NUM_ITER`, `REPEATS`, `RUN_TIMEOUT` | 5, 1, 900 s | timed iterations, runs per kernel and point, limit per run |

Defaults (`cluster_defaults.json`): 1 node, 8 GPUs, exclusive, partition `batch`, 1 hour, the Primus image as `SIF`.
Before every `sbatch`, VISTA runs `setup_lux.sh` on the login node. It clones RingX at the pinned commit `d1e53d7`
and installs `ring-flash-attn==0.1.8` into `$VISTA_JOB_DIR` (first submission only, seconds). By hand, the same job
is `scripts/setup.sh` then `scripts/bench.sh [KEY=VALUE ...]`, with outputs in `runs/ringx.<jobid>/`.

**Read `summary.txt`.** It ends with `=== BENCH OK` or `=== BENCH FAILED: <reason>`. Each test gets PASS or FAIL
with its max |diff|. Each kernel run gets two TFLOPS/GPU numbers. `utils` uses RingX's `benchmark/utils.py`
formula, as in the paper. `real` counts attention FLOPs only, and its `%` is relative to single-GPU
`flash_attn_func` at the same shape. Each point ends with its fastest kernel and the speedup over the best baseline.
A kernel that runs out of memory is listed as FAILED with its reason, and the job goes on. A healthy default run
passes every test, and ringX1 is about 2.7x ring at `32:8192`. The README of the job folder has the full list of
outputs and the reference numbers.

## Gotchas

- **The rank order is the sequence order.** The kernels decide which blocks to mask from `dist.get_rank(group)`. For
  causal kernels, CP rank r must hold the right chunks: chunk r for ringX2, chunks r and 2N−1−r (zigzag) for ringX3
  and ringX4. Data, labels, position ids and RoPE must all be sharded the same way.
- **Causal-only and non-causal-only kernels.** ringX2's backward, ringX3, ringX4 and ringX4o assert `causal=True`.
  ringX4's assert message wrongly reads "ringX1 is intended for causal=False". ringX3b asserts `causal=False`.
  `ringX1o_varlen` raises `NotImplementedError` for causal. All of them default to `causal=False`.
- **ringX3 memory.** ringX3 all-gathers the full KV onto every GPU (O(hbs) memory), so the context it supports is
  bounded by one GPU's memory. ringX4 has the same communication volume but O(hbs/N) memory.
- **ringX4 needs an explicit group.** It calls `dist.get_global_rank(group, i)`, which raises "Group None is not
  registered" for the default `group=None`. Pass `group=dist.group.WORLD` (or your CP group). ringX1/2/3 accept None.
- **No masks or ALiBi.** There is no attention-mask argument, and `alibi_slopes` must be `None`.
- **Private FlashAttention API.** The kernels call `flash_attn.flash_attn_interface._flash_attn_forward/_backward`.
  A FlashAttention build that changes those signatures will break them. FlashAttention 2.8.3 (ROCm, Primus image)
  wraps them as torch custom ops; RingX's `get_default_args` still reads the arguments, and the kernels work. The
  ring-flash-attention baselines (0.1.8) do not: they omit `rng_state`, and the ROCm backward then fails with "Cannot
  access data pointer of Tensor that doesn't have storage". The fix is in `references/benchmarking.md`.
- **The upstream benchmark as written.** `--module ringX1_attn` with `PYTHONPATH=../ringX_attn` likely fails,
  because the modules use relative imports. Without `--causal`, it stops at the zigzag baseline, which asserts
  causal, so plain ring is never timed. `benchmark/utils.py` assumes 5 iterations. Fixes are in
  `references/benchmarking.md`.

## Verify after any change

Compare the CP output and gradients against single-GPU `flash_attn_func` on the same (broadcast) inputs, as
RingX's `test/test_ringX*_attn_func.py` do. See "Numerical check" in `references/integration.md`.

## License and acknowledgments

The skill's own text is Apache-2.0 (`LICENSE`). RingX is MIT (Copyright 2025). The benchmark baselines come from
[ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention) (zhuzilin). The ViT and GPT examples link to
[climate-vit](https://github.com/jqyin/climate-vit) and [forge](https://github.com/at-aaims/forge) (`xforge` branch).
None of them is copied here, apart from small adapted files in `hpc_jobs/ringx-bench/`. `bench_ringx.py` follows
RingX's benchmark loop, and `tests/test_ringX4_attn_func.py` is adapted from RingX's ringX3 test (both MIT).
`shims/utils.py` reimplements ring-flash-attention's test helpers (MIT). The job clones RingX and installs
ring-flash-attention on Lux at submission.

If you use RingX, cite:

```bibtex
@inproceedings{yin2025ringx,
  title     = {RingX: Scalable Parallel Attention for Long-Context Learning on HPC},
  author    = {Yin, Junqi and Palash, Mijanur and Shankar, Mallikarjun and Wang, Feiyi},
  booktitle = {Proceedings of the International Conference for High Performance
               Computing, Networking, Storage and Analysis (SC '25)},
  year      = {2025},
  publisher = {ACM},
  doi       = {10.1145/3712285.3759859}
}
```
