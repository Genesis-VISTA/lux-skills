# mistral-cpt

Continued pretraining of Mistral-Large-Instruct-2411 (123B) on OLCF Lux (8x MI355X per node, apptainer), from
tutorial 3 of the Lux AI-for-Science hackathon: FP8, TP1 x PP8 (each node holds one copy of the model, so data
parallelism equals the node count), 90% Sci-Base scientific text + 10% chat and tool-calling conversations,
starting from the base model's Megatron checkpoint. Reports per-iteration step time, TFLOP/s/GPU, loss and
validation loss. Part of the `primus-skill` skill, which explains how to read the numbers.

Supported clusters: Lux only (Slurm over SSH; the user is asked to log in through the hub once per session).

Default nodes: 4 (exclusive, 8 GPUs each). Tested at 4 and 256; below 4 each GPU holds more optimizer state.
Default time: 0:45:00. The job checks every node's network rails, loads the 448 GB checkpoint from Lustre
(~10 min), then trains: iterations 1-2 take ~140 s and ~90 s (compilation), later ones ~38 s. The default 4-step
run takes ~8 min of job time; 10 iterations ~13 min. VISTA allows at most 4:00:00, so a long run (the reference
one-epoch run is 283 iterations on 256 nodes, ~3 h of training) must fit in that.

The model, the data, the Primus image and the tutorial's scripts are read from the shared package `PKG` in
`cluster_defaults.json`. On the first submission, setup on the Lux login node copies the tutorial's work dir
(patched Primus, scripts, JIT caches; ~2 min) into the job's shared folder; later submissions reuse it.

## Script args

`script_args` is a space-separated list of `KEY=VALUE` pairs. All are optional, and any other key is refused:

- `TRAIN_ITERS=<n>`: training iterations (default 4, the tutorial's test run).
- `TRAIN_GBS=<n>`: global batch in sequences of 4,096 tokens (default 512). Must divide by the node count.
- `TRAIN_WARMUP=<n>`: learning-rate warm-up iterations (default 1).
- `TRAIN_EVAL_INTERVAL=<n>`: iterations between validations (default 2 for the 4-step run, else `TRAIN_ITERS`:
  validate at the end). `TRAIN_EVAL_ITERS=<n>`: global batches per validation (default 1).
- `CHAT_SHARE=<0..1>`: share of chat data (default 0.1). 0 trains on Sci-Base only, which breaks the model as a
  chat model (the science benchmark fell from 0.651 to 0.295); keep it above 0 for a model to serve.
- `SAVE_INTERVAL=<n>`: save a checkpoint every n iterations and at the end (default off). Each is 224 GB.
- `LUX_GPU_CHECK=0|1`: a 4-minute GPU arithmetic check on every node before training (default 0; use 1 for long runs).
- `RUN_NAME=<name>`: names the run's log and checkpoint folder (default `cpt-N<nodes>`).

Examples: `script_args="TRAIN_ITERS=10"`, `node_count=8, script_args="TRAIN_ITERS=20 TRAIN_GBS=1024 SAVE_INTERVAL=20"`.

## Outputs

In the job's output directory:

- `summary.txt`: the rails preflight, one line per iteration (step ms, TFLOP/s/GPU, loss, grad norm), the
  validation losses, the checkpoint folder if one was saved, the exit line and `VERDICT: OK` or `FAILED`.
- `train.log`: the full job log, all ranks.

Checkpoints are not in the output directory: they go to the job's shared folder, under
`work/output/amd/root/<run>/checkpoints/` (the path is in `summary.txt`).

A healthy 4-node run: `4 nodes RAILS OK`, loss ~1.23 at iteration 1 (the checkpoint loaded; a random model starts
near 10), ~1,300-1,340 TFLOP/s/GPU from iteration 3 on, validation loss ~1.42, exit 0. Training loss moves up and
down between iterations because batches carry different amounts of chat; read the validation loss. A grad-norm
guard stops a run whose gradient norm explodes (exit 3). A node that fails the rails check stops the job and is
named; resubmit (VISTA cannot exclude nodes, so tell the user which node failed).
