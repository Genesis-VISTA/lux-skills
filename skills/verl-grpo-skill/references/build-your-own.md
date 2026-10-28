# Shared inputs and building your own

The skill's `env.lux.sh` sets `PKG=/lustre/orion/world-shared/stf218/junqi/Genesis-Hackathon` by default, matching `primus-skill` and `vllm-skill`. The Mistral starting model and Zenith inference image already existed there and match the originals' manifest hashes. The GRPO assets were copied from `/lustre/orion/world-shared/stf218/junqi/lux-primus/lux_hackathon_AI4Sci_tutorials`; that path remains the provenance source. The skill repository contains wrappers and documentation, not large inputs. Start from the skill directory with `source env.lux.sh; scripts/setup.sh`; the examples below assume that shell has `PKG` and `WORKDIR` set.

| Input | Shared path under `$PKG` | Default variable |
| --- | --- | --- |
| Training image, 12 GB | `containers/verl_v26.8A.sif` | `VERL_SIF` |
| Starting HF model, about 224 GiB | `models/Mistral-Large-Instruct-2411/` | `RL_HFMODELS` (parent directory), or `RL_MODEL` (exact model) |
| Zenith inference image | `containers/inference_image_gfx950_v8mi250_20261011.tar` | Used by checkpoint evaluation and data generation |
| Training pools, 41,091 rows each | `datasets/sci_rl/rl_control.parquet`, `rl_pubonly.parquet` | `RL_DATA` (directory) |
| Held-out dev set, 1,200 rows | `datasets/sci_rl/sci_dev.parquet` | `RL_DATA` (directory) |
| Candidate pools and pass-rate scores | `datasets/sci_rl/{rl_pool,rl_add1,scibase_scale1}.parquet`, `passrate/` | Used only to rebuild data |
| Raw data sources | `datasets/rl_sources/`, `datasets/Sci-Base/` | Used only for the full data pipeline |
| Reference trained model | `models/Mistral-Large-2411-SciRL-s125/` | Optional comparison; not the training start model |
| Tutorial code and Zenith bundle | `tutorials/04_mistral_rl/`, `tutorials/02_zenith_inference/` | Work tree setup, build and training scripts |

The work directory is separate: `$WORKDIR` holds the copied tutorial scripts, logs, generated data and checkpoints. Run `scripts/setup.sh` from this skill to create it. Training reads the shared inputs through Orion Lustre; nothing large is copied by setup. The tutorial code lives at `$PKG/tutorials/04_mistral_rl/` and is required even when all three large inputs are your own.

To use your own assets, create them on Orion Lustre, then set the overrides **before** running `scripts/train.sh`. `scripts/setup.sh` may run earlier against the shared defaults while you build the replacements; if you set overrides before setup, its input check requires those files to exist already.

```bash
export VERL_SIF=/lustre/orion/<account>/scratch/$USER/my-rl/containers/verl_v26.8A.sif
export RL_HFMODELS=/lustre/orion/<account>/scratch/$USER/my-rl/models
export RL_DATA=/lustre/orion/<account>/scratch/$USER/my-rl/datasets/sci_rl
# Set RL_MODEL only when its path differs from $RL_HFMODELS/Mistral-Large-Instruct-2411.
scripts/train.sh smoke
```

`RL_DATA` needs `rl_control.parquet`, `rl_pubonly.parquet` and `sci_dev.parquet` for the skill's setup check, even if the first run uses only control. For checkpoint evaluation, also set `HFMODELS=$RL_HFMODELS`. In VISTA, pass the same overrides in the job environment for both setup and execution; `cluster_defaults.json` supplies only `PKG` by default. These overrides do not publish your files to world-shared: use a project location with permissions appropriate for your collaborators.

## Rebuild the image

The shipped image is variant A of `$WORKDIR/images/build_verl.sbatch`: Primus v26.8 base digest `sha256:79629b5e5ba4fcfef1d49cfd9dc5247681243437e7fd1db815279870f41c1b3c`, verl commit `7b19203c52503760f92e0368067c7eb744236274`, and vLLM commit `18d87a87dc39c1a50f452b218fdcc3aeec48d18e`. The builder pins other Python dependencies in its header. Genesis-Hackathon's existing `primus_v26.5.sif` and Zenith vLLM image serve other skills; neither replaces the verl image. Building variant A needs registry/package access through the OLCF proxy, node-local NVMe, and a one-node GPU allocation; allow up to six hours for the build.

After `scripts/setup.sh`, on a Lux login node:

```bash
source "$PKG/tutorials/04_mistral_rl/hackathon_env.sh"
cd "$WORKDIR"
source lux/env.sh
export VERL_SIF="$WORKDIR/containers/verl_v26.8A.sif"
export OUT="$VERL_SIF" VARIANT=A
mkdir -p containers logs/images
lux_sbatch -N 1 -t 02:00:00 --wrap='cd "$WORKDIR"; bash images/pull_sif.sh "$PRIMUS_V268_REF" "$PRIMUS_V268_SIF"'
# Wait for the pull job to finish successfully before building:
lux_sbatch -t 06:00:00 images/build_verl.sbatch
```

The pull runs on a compute node to avoid the login node's shared `/tmp` limit. The build log is `logs/images/build_A_<jobid>.log`; the last step runs `images/inspect.sh` against the new SIF. Keep `VERL_SIF` set for later training jobs. Read the current headers of `images/pull_sif.sh` and `images/build_verl.sbatch` before changing versions or variants.

## Download the starting model

The shared model is the Hugging Face repository `mistralai/Mistral-Large-Instruct-2411`. Its packaged download metadata records revision `ba78820945ae22361b0274cf0ae6d696c967c1a4`. Accept its Mistral Research Licence and configure Hugging Face credentials if the repository requests them. The native `consolidated*` weights duplicate the HF safetensors; exclude them to avoid doubling the download.

```bash
source "$PKG/tutorials/04_mistral_rl/hackathon_env.sh"
cd "$WORKDIR"
source lux/env.sh
lux_proxy_on
python3 -m venv "$WORKDIR/hfvenv"
"$WORKDIR/hfvenv/bin/pip" install 'huggingface_hub[cli]'
export RL_HFMODELS="$WORKDIR/models"
"$WORKDIR/hfvenv/bin/hf" download mistralai/Mistral-Large-Instruct-2411 \
  --revision ba78820945ae22361b0274cf0ae6d696c967c1a4 \
  --exclude 'consolidated*' \
  --local-dir "$RL_HFMODELS/Mistral-Large-Instruct-2411"
```

Use an adequately sized Lustre location for `RL_HFMODELS`; `$WORKDIR/models` is only an example. Confirm `config.json`, `tokenizer.json`, the safetensor index, and all referenced shards exist before training. A different starting model changes the experiment and may need different model conversion settings.

## Reassemble pools from the shared candidates

This route uses the package's candidate parquets and precomputed scores, so it does **not** regenerate questions or model rollouts. The tutorial says the command below reproduces `rl_control.parquet` exactly. It is the fast way to change pool selection and inspect the assembly code. Create the tutorial's optional Python 3.9 data environment in your existing work directory:

```bash
source "$PKG/tutorials/04_mistral_rl/hackathon_env.sh"
cd "$WORKDIR"
source lux/env.sh
lux_proxy_on
/usr/bin/python3.9 -m venv datavenv
datavenv/bin/pip install -r data/requirements.txt
mkdir -p work/data/verl
S="$PKG/datasets/sci_rl"
lux_sbatch -t 00:30:00 data/job.sbatch data/assemble_pool.py \
  --pool "$S/rl_pool.parquet" "$S/rl_add1.parquet" "$S/scibase_scale1.parquet" \
  --vanilla "$S/passrate/passrate_base.rows.jsonl" \
            "$S/passrate/passrate_base_add1.rows.jsonl" \
            "$S/passrate/passrate_base_scibase1.rows.jsonl" \
  --out-dir work/data/verl
```

Wait for assembly before starting the public-only ablation. Preserve the order of score files: later files win when a row has multiple scores.

```bash
S="$PKG/datasets/sci_rl"
lux_sbatch -t 00:30:00 data/job.sbatch data/pub_pool_public_only.py \
  --control work/data/verl/rl_control.parquet \
  --pool "$S/rl_pool.parquet" "$S/rl_add1.parquet" "$S/scibase_scale1.parquet" \
  --vanilla "$S/passrate/passrate_base.rows.jsonl" \
            "$S/passrate/passrate_base_add1.rows.jsonl" \
            "$S/passrate/passrate_base_scibase1.rows.jsonl" \
  --out work/data/verl/rl_pubonly.parquet
# Wait for this job to finish before training.
[ -e work/data/verl/sci_dev.parquet ] || ln -s "$S/sci_dev.parquet" work/data/verl/sci_dev.parquet
export RL_DATA="$WORKDIR/work/data/verl"
```

Check that each training pool has 41,091 rows and the dev set 1,200 before using them. `scripts/setup.sh` must have completed before these commands; after reassembly, set `RL_DATA` for each training submission.

## Generate the data from raw sources

For a fully new pool, follow `$PKG/tutorials/04_mistral_rl/README.md` section 6 and the usage docstring in each `rl/data/` script. The ordered stages are: fetch public and evaluation sources with `data/fetch_sources.sh all`; normalize records; classify OpenScience domains; decontaminate against the evaluation sets; split by source document; render verl parquets; generate Sci-Base questions; score each candidate with eight rollouts from the starting model; assemble the control and public-only pools. Run CPU stages through `data/job.sbatch` and generation through `gen/vllm_batch.sbatch` as the tutorial documents. The data venv above is required.

This longer route also needs the shared Sci-Base documents, tutorial 2's Zenith inference image/server, and tutorial 3's Sci-Base CPT model for the question signal filter. The tutorial's `models/Mistral-Large-2411-SciBase-chatmix/` is only a placeholder in the source package, so users must build or supply that CPT model for the same signal-filter stage. Some evaluation sources require separate access, including gated GPQA. The shared `datasets/rl_sources` can save most public-source downloads; using it is a partial rebuild rather than a fully fresh fetch. The tutorial's section 6 records the exact source mix, filtering rules, generator settings, estimated node-hours, and licence/decontamination notes. Do not treat a newly generated pool as byte-identical to the package without comparing its counts, IDs, and scores.
