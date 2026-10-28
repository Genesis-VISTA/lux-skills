# grpo-train

Run tutorial 4's Mistral-Large-Instruct-2411 (123B) GRPO recipe on the HPC side of OLCF Lux. verl coordinates vLLM rollouts and a Megatron-Core policy update. The job uses 8 MI355X GPUs per node, apptainer, the package's prepared science parquets and its custom verifiable reward. `setup_lux.sh` copies the small tutorial work tree to Orion Lustre once; the model, datasets and container are read from `PKG` in `cluster_defaults.json`. Their exact world-shared paths and build instructions are in [../../references/build-your-own.md](../../references/build-your-own.md).

Supported cluster: Lux. Use 8 or 32 nodes, the configurations documented by the tutorial. Default: 8 exclusive nodes, 8 GPUs each, 2:30:00 wall limit, 3 smoke steps, no checkpoint or validation. The tutorial reports about 25 minutes when model loading is healthy. The longer wall limit allows for the documented Lustre-related weight-loading stall. Full-run mode needs a longer duration per segment (around 3 hours at 8 nodes). VISTA caps each job at 4 hours; the tutorial estimates 7–9 hours for all 125 steps on 8 nodes.

## Script args

`script_args` is a space-separated list of `KEY=VALUE` pairs. All keys are optional; unknown keys are refused.

- `MODE=smoke|control|pubonly` (default `smoke`): prepared `rl_control` smoke or checkpointed `control`/public-only training.
- `TARGET_STEP=<n>`: for smoke, 3 (default) or 10; for the full modes, 25, 50, 75, 100 or 125. This is the cumulative training step, not the number of new steps. A full-mode job advances at most 25 steps from its latest checkpoint.
- `RUN_NAME=<name>`: checkpoint and log name, starting with a letter or digit, then letters, digits, `.`, `_` and `-`. Defaults to `smoke_<jobid>`, `rl_control` or `rl_pubonly`. Keep it fixed across resume segments and distinct between arms.
- `GPU_CHECK=0|1` (default 1): preflight GPU arithmetic check on every node. Ionic rail checks always run.
- `DIST_TIMEOUT_MIN=<n>` (default 90): timeout for Megatron and torch distributed groups during model loading. Keep the job duration longer than this timeout so slow loading has time to complete.

Examples:

```text
submit_hpc_job(job="verl-grpo-skill/grpo-train", cluster="lux")
submit_hpc_job(job="verl-grpo-skill/grpo-train", cluster="lux", script_args="MODE=smoke TARGET_STEP=10")
submit_hpc_job(job="verl-grpo-skill/grpo-train", cluster="lux", duration="3:00:00", script_args="MODE=control TARGET_STEP=25")
submit_hpc_job(job="verl-grpo-skill/grpo-train", cluster="lux", duration="3:00:00", script_args="MODE=control TARGET_STEP=50")
```

## Outputs

In the per-job `$VISTA_OUT` directory:

- `summary.txt`: step metrics, last validation accuracy, checkpoint path, and `VERDICT: OK` or `FAILED`.
- `train.log`: full launcher and verl output, including preflight failures and Ray diagnostics.
- `job.log`: mode, run name, target, resources and exit codes.

Checkpoints stay in the job's shared `$VISTA_JOB_DIR/work/work/ckpt/<run>/global_step_<n>/`, not in VISTA's output folder. A checkpoint is about 1.6 TB with optimizer state. The tutorial keeps two by default and may require up to about 5 TB during a save. The Ray logs live under `$VISTA_JOB_DIR/work/logs/verl/ray_<jobid>/`.

A healthy smoke log has `8 nodes RAILS OK`, the GPU preflight, steps 1–3, `driver exit 0`, and `VERDICT: OK`. The measured 8-node, three-step and ten-step baselines are in [../../results/lux-8node-2026-10-27.md](../../results/lux-8node-2026-10-27.md). For a full mode, `VERDICT: OK` also requires the target checkpoint directory. If preflight names a bad node, choose another allocation; VISTA cannot submit `--exclude` for this job. The tutorial's `resume_mode=auto` should continue from the latest checkpoint when the next cumulative target is submitted; confirm the 25→50 transition on Lux before relying on a full VISTA chain.
