---
name: verl-grpo-skill
description: Run and interpret Mistral-Large 123B GRPO reinforcement learning with verl on OLCF Lux. Use for the Lux AI-for-Science tutorial's three-step smoke test, checkpointed control or public-only training, science reward debugging, and evaluation of RL checkpoints.
---

# verl GRPO on OLCF Lux

This skill runs the HPC side of Lux: Slurm, 8 MI355X GPUs per node, apptainer, Orion Lustre and node-local NVMe. The tutorial trains Mistral-Large-Instruct-2411 with **verl GRPO**: vLLM generates eight solutions per prompt, a verifiable science reward scores the final answer, and Megatron-Core updates the policy on the same GPUs. The base image uses Primus v26.8, but the training driver is verl. Use `primus-skill` for Primus continued pretraining and `vllm-skill` for a general serving endpoint.

The shared package defaults to `/lustre/orion/world-shared/stf218/junqi/Genesis-Hackathon`, the same package root used by `primus-skill` and `vllm-skill`. Its GRPO inputs are:

| Input | Path relative to `PKG` | Purpose |
| --- | --- | --- |
| verl image | `containers/verl_v26.8A.sif` | 12 GB Primus v26.8 + verl/vLLM image |
| Starting model | `models/Mistral-Large-Instruct-2411/` | 123B HF model, about 224 GiB |
| Training and dev data | `datasets/sci_rl/{rl_control,rl_pubonly,sci_dev}.parquet` | 41,091 / 41,091 / 1,200 rows |
| Tutorial code | `tutorials/04_mistral_rl/` | GRPO driver, reward, Ray/Slurm and data scripts |

The skill copies only the small tutorial work tree to your Lustre directory; jobs read these large inputs in place. The starting model and Zenith inference image were already in Genesis-Hackathon; the GRPO files were copied from the Lux AI-for-Science tutorial package. `PKG` can point to another complete copy. To build your own image, model or data, follow [references/build-your-own.md](references/build-your-own.md); it also explains the environment overrides. The older tutorial README names a ven136 path; use a path you can actually read on Lux.

## Start on a Lux login node

```bash
cd <this skill directory>
source env.lux.sh                    # ACCOUNT, PKG, WORKDIR; override any before sourcing
scripts/setup.sh                     # one-time work tree; no large model or image copy
scripts/train.sh smoke               # 8 nodes, 3 GRPO steps, about 25 minutes in the job
```

The 25-minute timing assumes model loading proceeds normally. If the log stays in Megatron initialization and then reports a `SCATTER` or Gloo timeout, the tutorial documents a Lustre-related weight-loading stall. Retry with its longer distributed timeout and enough Slurm time for loading plus training:

```bash
RL_DIST_TIMEOUT_MIN=90 RL_TIME=02:30:00 scripts/train.sh smoke
```

Check the new job's `logs/verl/slurm-rl-smoke_<jobid>.out` for `dist_timeout:` and the first GRPO step. A longer timeout only permits slow initialization to finish; repeated timeouts need investigation of model reads and distributed communication.

`ACCOUNT` must be an account you may charge. By default, `WORKDIR` is `/lustre/orion/$ACCOUNT/scratch/$USER/verl-grpo-skill/work`; set it to another Orion path before sourcing if your project has a different scratch layout. `scripts/train.sh` accepts `smoke`, `control`, or `pubonly`, then optional `sbatch` arguments. The tutorial's `RL_*` environment variables pass through; for example:

```bash
scripts/train.sh control             # 125 steps, checkpoint + validation every 25
RL_SAVE_FREQ=125 RL_KEEP_CKPT=1 scripts/train.sh control
scripts/train.sh pubonly             # public-data ablation
```

To run ten training steps as a hand-run check without writing a large checkpoint, use a distinct run name:

```bash
RL_STEPS=10 RL_RUN=smoke10 RL_SAVE_FREQ=-1 RL_TEST_FREQ=-1 \
  RL_VAL_BEFORE=False RL_DIST_TIMEOUT_MIN=90 RL_TIME=02:30:00 scripts/train.sh control
```

The full run is estimated at 7–9 hours on 8 nodes from the tutorial's smoke timing. A checkpoint with optimizer state is about 1.6 TB; the default keeps two and may temporarily need about 5 TB while writing another. Reuse the same `RL_RUN` to resume from the latest checkpoint. `control` trains on `rl_control.parquet`; `pubonly` replaces its Sci-Base problems with public ones. The package does not include the tutorial's `main` arm start model and pool.

## Run through VISTA

Import this skill and submit `verl-grpo-skill/grpo-train` on `lux`. The job follows the repository's `hpc_jobs/` format. Its default is the tested tutorial recipe's 8-node smoke configuration, not a full study run:

```text
submit_hpc_job(job="verl-grpo-skill/grpo-train", cluster="lux")
submit_hpc_job(job="verl-grpo-skill/grpo-train", cluster="lux",
               script_args="MODE=smoke TARGET_STEP=10")
get_hpc_job_outputs(job_id=<id>, cluster="lux", files=["summary.txt"])
```

For a checkpointed `control` or `pubonly` run, set `MODE` and a **cumulative** `TARGET_STEP`. Submit targets 25, 50, 75, 100 and 125 in order, retaining the same `RUN_NAME` and VISTA job shared folder. Each submission advances at most 25 steps so it fits VISTA's four-hour limit; set a duration around three hours for each 8-node segment. Check `summary.txt` for `VERDICT: OK` and a checkpoint before submitting the next segment. Resuming at a higher cumulative target is supported by the tutorial's `resume_mode=auto`; validate that behavior with the first 25→50 transition before relying on a multi-segment study run.

```text
submit_hpc_job(job="verl-grpo-skill/grpo-train", cluster="lux", duration="3:00:00",
               script_args="MODE=control TARGET_STEP=25")
submit_hpc_job(job="verl-grpo-skill/grpo-train", cluster="lux", duration="3:00:00",
               script_args="MODE=control TARGET_STEP=50")
```

The job accepts only `MODE=smoke|control|pubonly`, `TARGET_STEP=3` or `10` for smoke or a multiple of 25 through 125 for the other modes, `RUN_NAME=<name>` (starts with a letter or digit; then letters, digits, `.`, `_` or `-`), `GPU_CHECK=0|1` (default 1), and `DIST_TIMEOUT_MIN=<n>` (default 90). Use 8 or 32 nodes, the configurations documented by the tutorial. The default smoke wall limit is 2.5 hours to allow for slow model loading. The default full-run names are `rl_control` and `rl_pubonly`. Do not reuse one run name for two arms: verl resumes any existing checkpoint under that name. The job's shared work directory is `$VISTA_JOB_DIR/work`; its path appears in `summary.txt`.

The job's `setup_lux.sh` checks the package and builds that directory once on the login node. The job runs tutorial 4's `ray_slurm/run_verl.sbatch` *inside* VISTA's allocation; it does not call `sbatch` again. VISTA cannot pass the tutorial's bad-node exclusion. The launcher checks every node's ionic rails and optionally the GPUs, names failures in `train.log`, and stops if preflight fails.

## Read the result

Read `summary.txt` first: `VERDICT: OK`, the last GRPO step, mean reward, mean response length, step time, validation accuracy when enabled, and the checkpoint path. `train.log` contains the full Ray/verl log. The measured Lux three-step and ten-step results, including per-step reward and time, are in [results/lux-8node-2026-10-27.md](results/lux-8node-2026-10-27.md). Reward need not rise over three steps. The tutorial's reference 32-node, 125-step `control` run improved the science-suite mean from 0.666 to 0.739; an 8-node full run is estimated from smoke timing, not yet reported as a measured end-to-end result in this skill.

The custom reward is in the package's `tutorials/04_mistral_rl/rl/verl_rl/reward_science.py`. It accepts a clear final `answer is` statement near the end, scores exact multiple-choice sets or numeric answers within the specified tolerance, rejects conflicting answers and truncation, and penalizes repetitive output. The source's `test_reward_science.py` checks its parsing. If rewards look wrong, inspect examples in the work directory and that test before changing the GRPO settings. The training pool was filtered for prompts solved 1–7 times out of eight by the starting model; always-incorrect and always-correct groups give no GRPO advantage.

## Score a checkpoint

For a hand-run checkpoint, on a Lux login node:

```bash
scripts/eval.sh rl_control 125
```

For a VISTA checkpoint, set `WORKDIR` to the path in `summary.txt` before invoking `scripts/eval.sh`. The wrapper uses the tutorial's `hf_eval_dir.sh` to create a vLLM-readable HF directory using hard links to the exported weights, then submits tutorial 2's one-node Mistral benchmark. It writes science scores under the extracted Zenith package's `bench/mistral/runs/` directory. The benchmark includes GPQA questions: share aggregate `science.txt`, not the question-bearing JSON output.

For rebuilding or changing the training pool, read [references/build-your-own.md](references/build-your-own.md), then the package's `tutorials/04_mistral_rl/README.md`, section 6, and its `rl/data/` scripts. The default skill job reads the prepared parquets and does not need the optional data-pipeline venv or an internet connection.

## License and provenance

This skill's wrappers, VISTA job and documentation are Apache-2.0 (`LICENSE`). The tutorial code remains in the shared package and retains its upstream terms. Mistral-Large-Instruct-2411 and models trained from it use the Mistral Research Licence (research and non-commercial use). The RL pools include sources with separate terms, including non-commercial datasets and Qwen-generated OpenScience rows; see tutorial 4's `README.md` and the original package's `README.hackathon`. Do not put model weights, datasets, benchmark questions, container images or run logs in this repository.
