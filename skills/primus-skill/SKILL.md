---
name: primus-skill
description: Run AMD Primus (Megatron backend) LLM training on OLCF Lux (MI355X, Slurm + apptainer). Use when asked to set up, smoke-test, scale, or debug Primus/Megatron training jobs on Lux (a Mistral-7B smoke test, or full Mistral-Large-Instruct-2411 123B continued pretraining on Sci-Base and its export to Hugging Face format), add a new model config, or interpret TFLOP/s/GPU results.
---

# Primus on OLCF Lux

Lux: `login1.lux.olcf.ornl.gov`, nodes `lux001..504`, 8x MI355X (gfx950, 288 GB) per node, 128 CPU cores,
8x ionic 400G RoCE NICs, Slurm partition `batch`, container runtime **apptainer** (no podman/docker),
internet **only via proxy** (`proxy.ccs.ornl.gov:3128`). Bulk data goes on Lustre (`/lustre/orion/<proj>/...`);
`$HOME` is small NFS.

Everything here was verified on Lux (see "Verified results"). Read "Gotchas" before changing anything.

## Quickstart

```bash
cd <this repo>
source env.lux.sh                  # detects ACCOUNT; sets PRIMUS_DIR, SIF, proxy, PRIVATE_IFACE
scripts/setup.sh                   # once: clone Primus release/v26.5, pin Megatron, build helpers_cpp in-image
scripts/run_smoke.sh 1             # 1 node, Mistral-7B-v0.3, mock data, 20 iters (~3 min incl. startup)
scripts/run_smoke.sh 2 GLOBAL_BATCH_SIZE=512     # 2 nodes, SAME per-GPU batch (gbs = 256 x nodes)
cat runs/<jobid>/summary.txt      # first/3rd/last iteration + exit status (written by the job)
```

Full 123B model (Mistral-Large-Instruct-2411, FP8, TP1 x PP8, 90% Sci-Base + 10% chat mix, from the shared package):

```bash
scripts/setup_cpt.sh               # once: work dir $CPT_WORKDIR (patched Primus, tutorial scripts, JIT caches)
scripts/run_cpt.sh 4               # 4 nodes, 4 steps from the base checkpoint (~10 min once running)
scripts/export_cpt.sh <checkpoints dir> "<iters>"   # to HF safetensors for vLLM; see "Mistral-Large-123B"
```

`run_smoke.sh` prints the job id, log path and output folder. Queue wait varies from seconds to many hours depending on
load (`sbatch --test-only` shows the estimate); check `squeue -u $USER`. Wait for the job to leave the queue,
then read `summary.txt` (or `hpc_jobs/primus-smoke/summarize.sh <log>` on any log).

**Requirements:** a Slurm account (`sacctmgr -nP show assoc user=$USER format=Account`), and the image
`$SIF` (default `$PRIMUS_ROOT/images/primus-v26.5.sif`; override with `export SIF=...`).
No HF token is needed for the smoke test (Mistral-7B-v0.3 tokenizer is ungated, fetched via proxy).

## Layout

| Path | Purpose |
|---|---|
| `hpc_jobs/mistral-cpt/` | The 123B continued pretraining as a VISTA HPC job: `job.lux.slurm` runs tutorial 3's `lux/train.sbatch` in VISTA's allocation, `setup_lux.sh` makes the work dir once, `summarize_cpt.sh` writes `summary.txt`. |
| `hpc_jobs/primus-smoke/` | **The smoke test**, as a VISTA HPC job (see "Running through VISTA"). The hand-run scripts below use the same files. |
| `  setup_lux.sh` | Login node, idempotent: clone Primus `release/v26.5`, pin Megatron (`d3528a2`), build `helpers_cpp` in the image (Python 3.12 ABI). Refuses to touch an existing non-git directory. |
| `  job.lux.slurm` | Launcher: apptainer, torchrun, `bond0` master IP, node-local JIT caches; node count from `-N`. Takes `KEY=VALUE` args, writes `train.log` and `summary.txt` to `$VISTA_OUT`. No `#SBATCH` lines. |
| `  cluster_defaults.json` | Resources (1 node, 8 GPUs, exclusive, `batch`, 30 min) and env overrides such as `SIF`. |
| `  README.md` | The job's description for the agent: args and outputs. |
| `  summarize.sh` | Extracts per-iteration metrics from the LAST rank's log lines. |
| `  build_image.sh`, `build_helpers.sh` | Build the image (compute node only) and Megatron's `helpers_cpp`; used by setup and the job. |
| `  *.yaml` | Experiment configs (top-level `modules:`) go to `$PRIMUS_DIR/examples/megatron/configs/MI355X/`, all others are model configs and go to `$PRIMUS_DIR/primus/configs/models/megatron/`, at job start. Kept flat because the job copies them with a `*.yaml` glob (VISTA uploads subfolders too). |
| `env.lux.sh` | Site settings for running by hand. All values overridable via env before sourcing. |
| `scripts/setup.sh`, `scripts/run_smoke.sh` | Hand-run wrappers: source env, run `setup_lux.sh` / submit `job.lux.slurm` with `-A $ACCOUNT` and the same resources. |
| `scripts/setup_cpt.sh`, `run_cpt.sh`, `export_cpt.sh` | The 123B path by hand: wrappers around tutorial 3's tested scripts in `$PKG/tutorials/03_mistral_cpt/` (see "Mistral-Large-123B"). `run_cpt.sh` accepts any `train.sbatch` knob and excludes bad nodes; the VISTA job takes a fixed list. |
| `LICENSE` | Apache-2.0, for this repo's own code (see "License and acknowledgments"). |
| `lux-primus/` | The earlier upstream Mistral-Large package (runbook, scaling logs; git-ignored). Superseded by the 123B path. |
| `runs/` | Job logs (git-ignored). |

## Verified results (Mistral-7B-v0.3, BF16, mock data, seq 2048, mbs 4, Turbo off)

| Nodes | gbs | microbatches/rank | steady TFLOP/s/GPU | step |
|---|---|---|---|---|
| 1 | 256 | 8 | ~944 | 3.25 s |
| 2 | 256 | 4 | ~686 | — |
| 2 | 512 | 8 | ~808-811 (2 runs) | 3.78 s |

Qwen2.5-3B (mbs 10, gbs 640): 1 node ~804, 2 nodes ~767 TFLOP/s/GPU. Healthy runs: exit 0, 20/20 iterations,
loss falling (mock data is memorized, so loss going near 0 is expected and says nothing about quality),
iteration 1-2 slow (JIT, ignore). **Compare scaling at equal per-GPU batch**: halving it cost ~27% per GPU,
a pure batch-size effect; the true 1->2 node cost at equal batch is ~14%.

## Adding a model or experiment

1. Model: add `hpc_jobs/primus-smoke/<name>.yaml` (`extends: llama_base.yaml` for Llama/Mistral-style decoders; set
   `hidden_size`, `ffn_hidden_size`, `num_layers`, `num_attention_heads`, `num_query_groups`, `kv_channels`,
   `rotary_base`, `norm_epsilon`, tokenizer; copy dims from the HF `config.json`). Reference: `hpc_jobs/primus-smoke/mistral_7B_v0.3.yaml`.
2. Experiment: copy `hpc_jobs/primus-smoke/smoke-mistral_7B-bf16-mock.yaml` (keep its top-level `modules:`), point `model:` at it.
3. Run: `scripts/run_smoke.sh 1` with `CONFIG_NAME=<file>.yaml` (e.g. `scripts/run_smoke.sh 1 CONFIG_NAME=x.yaml`).
4. A `${VAR:default}` knob in a YAML only works if the launcher forwards `VAR` into the container. `job.lux.slurm`
   accepts and forwards `CONFIG_NAME`, `GLOBAL_BATCH_SIZE` and `TRAIN_ITERS` (and sets `PRIMUS_WORKSPACE`); to add a
   knob, add it to the `case` that accepts arguments AND as `APPTAINERENV_<VAR>`, then list it in the job's README.
   An unforwarded variable is **silently ignored** (the default is used).

## Running through VISTA

`hpc_jobs/primus-smoke/` follows VISTA's HPC job format (VISTA docs, "Self-contained HPC skills": https://genesis-vista.github.io/docs/modsim-skill).
When this skill is loaded in a VISTA project, the agent runs it as `primus-skill/primus-smoke`: the skill's name, then
the job's. Nothing needs copying into VISTA's own `hpc_jobs/`. The skill has two jobs: `primus-smoke` (Mistral-7B,
mock data) and `mistral-cpt` (Mistral-Large-123B continued pretraining; see "Mistral-Large-123B"). Exporting a
checkpoint to Hugging Face format is not a VISTA job yet: `scripts/export_cpt.sh` runs by hand.

```text
submit_hpc_job(job="primus-skill/primus-smoke", cluster="lux", node_count=1)
submit_hpc_job(job="primus-skill/primus-smoke", cluster="lux", node_count=2, script_args="GLOBAL_BATCH_SIZE=512")
get_hpc_job_status(job_id=<id>, cluster="lux")
get_hpc_job_outputs(job_id=<id>, cluster="lux", files=["summary.txt"])

submit_hpc_job(job="primus-skill/mistral-cpt", cluster="lux")                                   # 4 nodes, the tutorial's 4 steps
submit_hpc_job(job="primus-skill/mistral-cpt", cluster="lux", script_args="TRAIN_ITERS=10")
submit_hpc_job(job="primus-skill/mistral-cpt", cluster="lux", node_count=8, duration="1:30:00",
               script_args="TRAIN_ITERS=20 TRAIN_GBS=1024 SAVE_INTERVAL=20 RUN_NAME=my-cpt")
```

VISTA uploads the job folder's files, except its README, `cluster_defaults.json`, `job.lux.slurm` and `setup_lux.sh`
(so the YAMLs and the build and summarize scripts), to `$RUN_DIR_Lux`, and runs `setup_lux.sh` on
the login node before every `sbatch`. A missing image or a failed clone is therefore an error from the submit call,
not a failed job. The Primus clone lives in VISTA's per-job shared folder (`$VISTA_JOB_DIR/Primus`), so only the
first submission clones and builds. `cluster_defaults.json` sets `SIF` to the shared image; if it is unreadable, setup
checks only that the user's own image folder is writable, and the job builds the user's own image (see "Image"). Only the `KEY=VALUE` args in the job's README are accepted.

Read the smoke test's `summary.txt` as in "Verified results": exit 0, 20/20 iterations, loss falling.

`mistral-cpt` accepts `TRAIN_ITERS`, `TRAIN_GBS`, `TRAIN_WARMUP`, `TRAIN_EVAL_INTERVAL`, `TRAIN_EVAL_ITERS`,
`CHAT_SHARE`, `SAVE_INTERVAL`, `LUX_GPU_CHECK` and `RUN_NAME` (its README has the defaults). Defaults: 4 nodes, 45
min (VISTA allows at most 4 h, so size `TRAIN_ITERS` to fit: ~10 min of loading, then ~38 s per step at gbs 512 on 4
nodes). Its setup copies the tutorial's work dir into `$VISTA_JOB_DIR/work` once (~2 min on the first submission);
every run shares it, and checkpoints (224 GB each) are written there, at the path `summary.txt` gives. Its
`summary.txt` lists the rails preflight, every iteration (step time, TFLOP/s/GPU, loss, grad norm), the
validation losses and `VERDICT: OK`; compare with the reference table under "Mistral-Large-123B". Unlike by hand,
VISTA cannot exclude nodes: a node that fails the rails check is named in `summary.txt`; tell the user and resubmit.

## Gotchas (each cost real time)

- **Primus-Turbo is broken in this image.** Stock Primus configs set `enable_primus_turbo: true`; every rank then fails with
  `Primus-Turbo requires amd-aiter==0.1.14.post1`. Set `enable_primus_turbo`, `use_turbo_attention`,
  `use_turbo_grouped_gemm` to `false` (our configs do).
- **Harmless log noise** (do not chase): apex `fused_weight_gradient_mlp_cuda` JIT failure
  (`cuda_runtime_api.h` not found; only an MoE patch is skipped); `libxalt_init.so` LD_PRELOAD error;
  "NCCL WARN dlvsym loaded successfully for ionic_dv_*" (driver load message); `destroy_process_group` warning at exit.
- **Apptainer needs flags podman didn't:** `--rocm` (else no GPU) and an explicit `--bind /lustre/orion` (else the
  directory is silently empty). Pass env with `APPTAINERENV_*`, never `--env` (it comma-splits values such as
  `NCCL_IB_HCA=ionic_0,ionic_1,...` and RCCL quietly uses one NIC).
- **`--cpus-per-task=128` is mandatory** when one task runs a multi-process launcher: without it Slurm confines all
  ranks to one CPU and throughput drops ~5x (reference measured 261 vs 1273 TFLOP/s).
- **JIT caches must be node-local** (`/tmp/$USER/primus`: Triton, inductor, `TORCH_EXTENSIONS_DIR`); `/tmp` is wiped
  between jobs, so each job recompiles. Small models start in ~1 min; the 123B model needs ~20 min.
- **Master address must be an IP on `bond0`**, not a hostname (can resolve to 127.0.1.1 inside the container).
- **Slurm on Lux requires `-A`, `-N`, `-t`** and rejects the `worker` partition; use `batch`. Ask for the shortest
  `-t` you need; short jobs backfill. `--mem=0` is not needed.
- **Never unpack/build many small files on Lustre** (apptainer image builds: use a compute node with node-local tmp).
- **Reading logs:** per-iteration metrics are printed by the last rank (`rank-<n-1>/<n>`), each line twice. Filters
  on "error"/"exit" match Primus's config dump and give false alarms; match `Training execution failed` instead.
- **Rare unexplained hang (seen once, job 84470, 2 nodes).** Iteration 1 printed, then no progress: all 16 GPUs at 100%,
  Python at ~360% CPU, no IB/NCCL error, and no watchdog kill after 14 min. The identical resubmit passed (20/20, same
  master node). Treat as transient: if a job shows no new iteration for ~5 min after iteration 1 (steady steps take 2-4 s),
  `scancel` it and resubmit; keep `-t` short (the sbatch default is 30 min) so a hang costs little. For large jobs add a
  first-iteration watchdog like the reference `submit_train.sbatch` (`FIRST_ITER_TIMEOUT`, `scancel` on silence).
  If it repeats on the same nodes, intersect node lists across failures and `--exclude` them.
- **GitHub through the proxy resets connections sometimes**; `setup.sh` retries the submodule fetch.
- A pre-existing checkout at some path is not yours to overwrite; use a different `PRIMUS_ROOT`.

## Image

The smoke test uses the shared image `/lustre/orion/world-shared/stf218/junqi/primus/primus-v26.5.sif` (docker
`rocm/primus:v26.5`; the reference package's `env.lux.sh` has its pinned digest). It is only read. If it is missing or unreadable, the job
uses the user's own copy, `USER_SIF` (under VISTA, `images/` beside the job's uploaded sources in the user's own
`<remote dir>.<user>.jobs/primus-skill/primus-smoke/`; by hand, `$PRIMUS_ROOT/images/`), and if that is missing too, builds it
(`hpc_jobs/primus-smoke/build_image.sh`) on its first compute node before training: ~128 GB RAM, node-local
`APPTAINER_TMPDIR`, ~26 min (a login-node build with tmpdir on Lustre did not finish in an hour). Give such a job at
least 1 hour; with under 45 min left it stops at once with exit 3. The image is copied next to `$USER_SIF` and renamed into
place, so a concurrent reader never sees half an image. Set `PRIMUS_IMAGE=rocm/primus@sha256:<digest>` to pin the source.

## Mistral-Large-123B continued pretraining

This is tutorial 3 of the Lux AI-for-Science hackathon, run from the shared read-only package
`$PKG=/lustre/orion/world-shared/stf218/junqi/Genesis-Hackathon` (`env.lux.sh`). The package holds everything, so
no HF token, download or checkpoint conversion is needed:

| `$PKG/...` | |
|---|---|
| `containers/primus_v26.5.sif` | Primus v26.5, ROCm 7.15 (`CPT_SIF`). **Not** the smoke test's `SIF`: the 123B numbers were measured on this one. |
| `models/mistral_large_megatron_tp1pp8/` | the base model as a Megatron TP1/PP8 checkpoint, where training starts (448 GB) |
| `models/Mistral-Large-Instruct-2411/` | the base model in HF format: tokenizer, export reference (224 GB) |
| `datasets/scibase_sciall_train/`, `scibase_sciallval_s200-208/` | Sci-Base in Megatron format: 34.07B training tokens, 1.34B held-out |
| `datasets/chatmix_v7/` | 1.98M chat and tool-calling conversations in Mistral's v7 format (1.75B tokens) |
| `tutorials/03_mistral_cpt/` | the tutorial's scripts (`lux/`), patched Primus (`primus/Primus.tar`), JIT caches, bad-node snapshot; its `README.md` is the full guide |

`MANIFEST.sha256` in `$PKG` checksums all of it.

**Setup** (`scripts/setup_cpt.sh [dir]`, once, ~2 min): makes `$CPT_WORKDIR` (default
`/lustre/orion/<account>/scratch/$USER/mistral-cpt`, must be on `/lustre/orion`; containers mount it as `/work`).
It holds the Primus tree, logs, Primus output and checkpoints, which are 224 GB each.

**Train** (`scripts/run_cpt.sh [nodes] [KEY=VAL ...]`): sources the tutorial's `hackathon_env.sh` and
`lux/lux_env.sh`, then submits `lux/train.sbatch` through `lux_sbatch` (account, `batch`, bad nodes excluded).
The defaults are the tutorial's 4-node, 4-step run. The wrapper's own keys are `TIME` (`-t`, default `00:45:00`),
`CHAT_SHARE` (default 0.1; 0 trains on Sci-Base only), `SAVE_INTERVAL` (adds `--save`) and `EXTRA_PRIMUS_ARGS`.
Every other pair goes to `train.sbatch`, for example `TRAIN_ITERS`, `TRAIN_GBS` (512), `TRAIN_WARMUP`, `EPOCHS`,
`RUN_NAME`, `LUX_GPU_CHECK=1`, `LUX_BW=1`, `GRAD_GUARD=0` or `LUX_EXCLUDE_EXTRA=luxNNN`.

```bash
scripts/run_cpt.sh 4                                             # tutorial smoke, 4 steps
scripts/run_cpt.sh 4 TRAIN_ITERS=20 SAVE_INTERVAL=20 TIME=01:30:00 RUN_NAME=my-cpt   # keep a checkpoint to export
scripts/run_cpt.sh 256 TRAIN_GBS=32768 TRAIN_ITERS=283 TRAIN_WARMUP=50 TRAIN_EVAL_INTERVAL=32 \
  SAVE_INTERVAL=71 LUX_GPU_CHECK=1 TIME=04:45:00 RUN_NAME=chatmix-1ep       # reference: one epoch, ~3 h training
```

TP1 x PP8 puts one model copy on each node, so data parallelism equals the node count and micro-batch is 1. The
global batch must divide by the node count, and the wrapper refuses one that doesn't. Runs have been measured at
4 and 256 nodes; below 4 the wrapper warns, because each GPU holds a larger share of the optimizer state.

`train.sbatch` first checks every node's 8 ionic rails, and with `LUX_GPU_CHECK=1` runs a 4-minute GPU arithmetic
check. A bad node stops the job and is named in the log; resubmit with `LUX_EXCLUDE_EXTRA=<node>`. It then loads
the 448 GB checkpoint from Lustre and trains. A grad-norm guard cancels a run whose gradient norm explodes (exit 3).

Logs: `$CPT_WORKDIR/logs/train/slurm-<RUN_NAME>_<jobid>.out` holds the preflight and the job. The ranks' output is
in `<RUN_NAME>_n<N>_j<jobid>_<date>.log` (`grep 'iteration .*/'` and `grep 'validation loss'`; it ends `=== exit
<rc>`). Checkpoints go to `$CPT_WORKDIR/output/amd/root/<that log's name>/checkpoints/iter_NNNNNNN/`.

Reference numbers (tutorial 3 README):

| Run | Nodes | gbs | step | TFLOP/s/GPU |
|---|---|---|---|---|
| 4-step smoke (job 265721) | 4 | 512 | ~37 s | ~1,300 (1,342.8 / 1,299.8); val loss 1.422 at step 4; 7 min container start to exit |
| Sci-Base only (job 227845, 1,206 steps) | 256 | 32,768 | 38.84 s | 1,283 (51% MFU); held-out loss best 1.1596 |

Training loss jumps between steps because batches carry different amounts of chat. Read the validation loss
instead.

**Why the chat mix:** training on Sci-Base alone broke the instruct model as a chat model. Answers ran on, tool
calls stopped, and the science suite fell from 0.651 to 0.295. Keep `CHAT_SHARE` above 0 for anything you mean to
serve. The tutorial README, section 1, has the details.

**Export and score:** `scripts/export_cpt.sh <checkpoints dir> "<iters>" [out]` runs `lux/export_hf.sbatch` on
1 node (~1 h). It checks a bitwise round trip of the base model, shapes and dtypes, and held-out loss, and it
fails if any check fails. Each `iter_NNNNNNN/` it writes can be served by vLLM: the `vllm-skill` skill's
`bench/mistral/mistral_bench.slurm.sh` (in its `inkling-serve/lux-inference/` engine) with `ZENITH_MODELPATH=<out> ML_MODEL=iter_NNNNNNN` runs the
correctness probe, tool check, GSM8K and the science suite. The base model scores 0.662 there (+-0.027), and a
4-step run will not move that.

**Verified from this repo** (job 308974, 2026-10-13, `scripts/run_cpt.sh 4 RUN_NAME=cpt-test`, lux297/300/301/303):
4 nodes RAILS OK; exit 0 in 7 min 41 s of job time; steps 3 and 4 took 37.2 s and 38.4 s at 1,339.8 and 1,297.6
TFLOP/s/GPU (reference 1,342.8 / 1,299.8). Loss 1.234 at step 1 shows the checkpoint loaded (a random model starts
near 10). Final validation loss was 1.420 (reference 1.422). Steps 1 and 2 are slow (142 s and 90 s, JIT and warmup);
ignore them. `export_cpt.sh` has only been dry-run.

**Verified through VISTA's submission path** (job 316703, 2026-10-13, `primus-skill/mistral-cpt` with
`script_args="TRAIN_ITERS=10"`, VISTA's `#SBATCH` header on stdin, uploaded sources only, lux417/433/438/447): setup
made the work dir in 3 min on the first submission; 4 nodes RAILS OK; exit 0 in 11 min 14 s of job time. Iterations
3-10: 38.2-38.4 s, 1,296.7-1,303.6 TFLOP/s/GPU (mean ~1,301); loss 1.212 at step 1; validation loss 1.271 at step 10.

## Verify after any change

Run `scripts/run_smoke.sh 1`, then read `runs/<jobid>/summary.txt`; expect exit 0, 20/20 iterations, ~900+ TFLOP/s/GPU.

## License and acknowledgments

This skill is adapted from AMD's work. **Primus** (AMD-AGI/Primus) and the `rocm/primus` image are AMD's. The
Mistral-Large-123B recipe is AMD's Mistral-Large MFU replication package, ported to Lux in the hackathon's
tutorial 3: the FP8 config, TP1 x PP8 layout, Primus patch, launch, preflight and export scripts and every
reference number. The Lux adaptation (this repo's job folder, wrappers, `env.lux.sh` and docs) was done at the
Oak Ridge Leadership Computing Facility. Errors introduced by that adaptation are ours, not AMD's.

This repo's own code is licensed under the Apache License 2.0 (`LICENSE`). Everything it runs or reads keeps its
own terms:

| Component | Licence |
|---|---|
| Primus (cloned by `setup_lux.sh`; `primus/Primus.tar` in `$PKG`) | MIT, Copyright (c) 2025 Advanced Micro Devices, Inc. (`Primus/LICENSE`) |
| Megatron-LM (Primus submodule) | BSD-3-Clause style, Copyright (c) NVIDIA Corporation (`third_party/Megatron-LM/LICENSE`) |
| Tutorial 3 scripts and replication package (`$PKG/tutorials/03_mistral_cpt/`) | AMD's terms for that package; copyright stays with AMD and its contributors |
| `rocm/primus` image (ROCm, PyTorch, Transformer Engine, apex, aiter) | their upstream licences, inside the image |
| `mistral_common` and its dependencies (`$PKG/tutorials/03_mistral_cpt/pylib/`) | Apache-2.0; pycountry LGPL-2.1; pydantic-extra-types MIT |
| Mistral-7B-v0.3 tokenizer (smoke test) | Apache-2.0 |
| Mistral-Large-Instruct-2411, its Megatron conversion and **any model trained from it** | Mistral Research Licence: research and non-commercial use only; derivatives carry the same licence |
| Sci-Base (opendatalab/Sci-Base) | CC BY 4.0 |
| chatmix_v7 sources | tulu-3-sft-mixture ODC-BY, smoltalk Apache-2.0, hermes-function-calling-v1 Apache-2.0 |

Scoring an export uses GPQA Diamond, whose authors ask that the questions not be posted online. Do not publish
`science*.json` results; aggregate scores are fine.
