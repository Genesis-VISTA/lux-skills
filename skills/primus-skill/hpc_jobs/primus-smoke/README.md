# primus-smoke

Smoke-test AMD Primus (Megatron backend) LLM pre-training on OLCF Lux (8x MI355X per node,
apptainer): mock data, no checkpoint, 20 iterations by default. Reports per-iteration loss,
step time and TFLOP/s/GPU. Part of the `primus-skill` skill, which explains how to read the
numbers.

Supported clusters: Lux only (Slurm over SSH; the user is asked to log in through the hub once
per session).

Default nodes: 1 (exclusive, 8 GPUs)
Default time: 0:30:00 (a 1-node run takes about 3 minutes once it starts; queue wait varies)

Every submission first checks the Primus checkout on the Lux login node (cloned and built on
the first submission, a few minutes; reused after that). The Primus apptainer image is the
shared one named by `SIF` in `cluster_defaults.json`, which is only read. If it is missing or
unreadable, the job uses the user's own copy in `images/` beside the job's uploaded sources, and
builds it first on its compute node if that is missing too. Building takes about 30 minutes, so submit with `duration="1:00:00"` or more.
A job with less time left fails within seconds and says so (exit 3); resubmit it longer. Later
jobs reuse the image.

## Script args

`script_args` is a space-separated list of `KEY=VALUE` pairs. All are optional, and any other
key is refused:

- `CONFIG_NAME=<file>.yaml`: experiment config (default `smoke-mistral_7B-bf16-mock.yaml`).
  Also available: `smoke-qwen2.5_3B-bf16-mock.yaml`.
- `GLOBAL_BATCH_SIZE=<n>`: global batch (default 256). To compare node counts, keep the per-GPU
  batch fixed: `GLOBAL_BATCH_SIZE` = 256 x nodes for Mistral-7B.
- `TRAIN_ITERS=<n>`: iterations (default 20).

Example: `node_count=2, script_args="GLOBAL_BATCH_SIZE=512"`.

## Outputs

In the job's output directory:

- `summary.txt`: the first, third and last iteration lines (loss, step ms, TFLOP/s/GPU), the
  exit line, and `FAILED` if training failed.
- `train.log`: the full training log.
- `primus/`: Primus's own workspace output.

A healthy run exits 0 with 20/20 iterations and falling loss. Expect about 900 TFLOP/s/GPU or
more on 1 node and about 800 on 2 nodes at gbs 512. Iterations 1-2 are slow (JIT compilation),
so ignore them.
