# ringx-bench

Test and benchmark the RingX context-parallel attention kernels on one OLCF Lux node (8x MI355X) in the Primus
image. The job does three things:

1. It checks each kernel's output and gradients against single-GPU `flash_attn_func`, using RingX's own tests.
2. It times forward plus backward for the RingX kernels and the ring-flash-attention baselines (ring, zigzag ring,
   stripe) at the points you give.
3. It times single-GPU `flash_attn_func` at the same per-GPU shapes, as a compute reference.

The job is part of `parallel-attention-skill`, which explains how to choose a kernel from the results.

Supported clusters: Lux

Default nodes: 1 (exclusive, 8 GPUs). With more nodes, the context-parallel group is every GPU of the job; it was
run on 1–16 nodes. Multi-node points take longer: at 64K tokens/GPU, 16 nodes took 69 minutes, mostly in the ring
baselines (set `RUN_TIMEOUT` so slow baselines time out instead of using up the job).
Default time: 1:00:00. The default run takes about 8 minutes once it starts. With all the defaults, each kernel
run takes 10–60 s, so allow about 1 minute per kernel and point. Points with large batches are slower: the baselines
at `64:4096` take up to 35 s for 5 iterations. Raise `duration` for big sweeps or `REPEATS`.

Every submission first runs `setup_lux.sh` on the Lux login node. It clones
[jqyin/ringX-attention](https://github.com/jqyin/ringX-attention) at the pinned commit `d1e53d7` and installs
`ring-flash-attn==0.1.8` with `pip --target`. Both go into `$VISTA_JOB_DIR`, so only the first submission pays for
them (seconds). Neither is installed into the image: the job puts both on `PYTHONPATH`. The image is the shared,
read-only Primus image named by `SIF` in `cluster_defaults.json` (torch 2.12, ROCm 7.15, FlashAttention 2.8.3,
RCCL 2.30.4).

## Script args

`script_args` is a space-separated list of `KEY=VALUE` pairs. All are optional, and any other key is refused.
Lists are comma-separated, with no spaces.

- `MODE=all|correctness|bench` (default `all`): run both parts, or only one of them.
- `TESTS=<list>` (default `ringX1,ringX2,ringX3,ringX4`): the correctness tests to run.
- `CAUSAL_KERNELS=<list>` (default `ringX2,ringX3,ringX4`; also `ringX2o`, `ringX4o`): RingX kernels to time at the
  causal points.
- `BIDIR_KERNELS=<list>` (default `ringX1`; also `ringX1o`): RingX kernels to time at the bi-directional points.
- `CAUSAL_POINTS=<b>:<s>,...` (default `1:16384`) and `BIDIR_POINTS=<b>:<s>,...` (default `32:8192`): the points
  to time. Each point is a batch size and a number of **tokens per GPU**. The global sequence is 8 × s. s must be a
  multiple of 16 (2 × GPUs). Leave one empty (`BIDIR_POINTS=`) to skip that attention type.
- `BASELINES=0|1` (default 1): also time the ring-flash-attention baselines. Causal points get ring, zigzag and
  stripe. Bi-directional points get ring only, because zigzag and stripe assert causal.
- `CEILING=0|1` (default 1): time single-GPU `flash_attn_func` at each point's per-GPU shape.
- `NUM_HEADS=<n>` (default 32), `HEAD_DIM=<n>` (default 128, a multiple of 8): the attention shape. The defaults are
  Llama3-8B.
- `NUM_ITER=<n>` (default 5): timed iterations per run, after 1 warm-up iteration.
- `REPEATS=<n>` (default 1): runs per kernel and point. Use 3 or more for the baselines: at batch 64 they varied
  up to 1.7x between runs.
- `RUN_TIMEOUT=<s>` (default 900): seconds before one kernel run is stopped.

Examples:

- `script_args="MODE=correctness"`
- `script_args="CAUSAL_POINTS=1:65536,4:16384,16:4096 BIDIR_POINTS= BASELINES=0"`
  (the same 64K tokens per GPU, split differently between batch and sequence)
- `duration=7200, script_args="CAUSAL_POINTS=64:4096 BIDIR_POINTS=64:8192 REPEATS=3"`

Memory limits measured on Lux: causal batch 64 at 16384 tokens/GPU does not fit. ringX3 runs out of memory
all-gathering KV, and ringX2 and ringX4 fail with "CUDA driver error: 1". A failed kernel run is reported in the
summary as FAILED with its reason, and the job continues with the next run.

## Outputs

In the job's output directory:

- `summary.txt`: what to read first. It holds the software versions, then PASS or FAIL for each test with its max
  |diff|, then one line per kernel run. Each run line gives the time, TFLOPS/GPU, % of the single-GPU reference and
  peak memory. Each point ends with its fastest kernel and that kernel's speedup over the best baseline. The file
  ends with `=== BENCH OK` or `=== BENCH FAILED: <reason>`, including for failures before anything runs.
- `results.json`: the same data, machine-readable.
- `correctness/<test>.log` and `.rc`, and `bench/<mode>-b<b>-s<s>-<kernel>.<rep>.log`: the raw output of each run.
- `runs.tsv`: one line per kernel run (point, kernel, run, exit code, log).
- `ceiling.txt`: the single-GPU `flash_attn_func` reference. `utils_summary.txt`: RingX's own
  `benchmark/utils.py` table, for comparison with upstream numbers.
- `env.txt`: node, RingX commit, torch, FlashAttention and RCCL versions. `job.log`: the job's own log.

The summary gives two TFLOPS numbers. **`utils`** is the number RingX's `benchmark/utils.py` reports, as in the
paper. It counts projection FLOPs the kernel does not do, and it counts causal attention as if it were full.
**`real`** counts only attention FLOPs (4·b·S²·H·d × 3.5 for forward plus backward), halved for causal. Use `real`
and the `%` of the single-GPU reference to judge efficiency. Use `utils` to compare with published RingX numbers.

What a healthy run looks like, from one Lux node with the defaults:

- Every test passes. Typical max |diff| values are out ≤ 0.004, lse ≤ 4e-6, and dq/dk/dv ≤ 0.03.
- At bi-directional `32:8192`, ringX1 gets about 340 utils TFLOPS/GPU (about 77% of the single-GPU reference). It
  is about 2.7x faster than ring.
- At causal `1:16384`, ringX2 and ringX3 get about 360 utils TFLOPS/GPU, within ~10% of ring.
- At causal `64:4096`, ringX3 gets about 580 utils TFLOPS/GPU, about 3x ring.

These are single runs. The full tables, including multi-node scaling to 16 nodes and a sequence-length sweep, are
in `references/benchmarking.md`.
