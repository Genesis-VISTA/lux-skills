# Muse Spark Open on Lux (reference)

**Target: Lux HPC side** (Slurm, rootless podman, ROCm image, Orion Lustre, node-local NVMe). No Kubernetes.

This is the detail behind `SKILL.md` § "Muse Spark Open". Its source is AMD's Zenith **v0.1.9**:
- repository `github.com/jbelofamd/zenith`, git tag `v0.1.9`;
- commit `70e5f0bf136e9b236251da24373b0241c6b40d4e` (`docker/VERSION`; `README.md` L3).

That tree is called `zenith/` below. Every path, flag and version is cited as `zenith/<file>:L<line>`. Lux facts
come from the [Lux User Guide](https://docs.olcf.ornl.gov/systems/lux_user_guide.html) ("guide"). Model facts not
in zenith come from Meta's Hugging Face [model card](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B) ("card"). The shared Lux paths and the OLCF proxy come from the
skill's maintainer ("maintainer"). Anything none of these settles is marked `TODO:`.

**Status: ported.** The VISTA job `vllm-skill/muse-serve` (`hpc_jobs/muse-serve/`) serves Muse. Its engine
(`hpc_jobs/muse-serve/lux-inference/`) is zenith v0.1.9's `docker/` and the bench clients the job runs, renamed
Zenith -> Lux as the v8 engine was. Muse's serving recipe is a tuned-defaults case keyed on the model name, as for
Inkling (§5). The Inkling job `vllm-skill/inkling-serve` and its v8 engine are unchanged and still refuse `MODEL=Muse-*`.

## 1. The model

| item | value | source |
|---|---|---|
| name | Muse-Spark-Open-816B-A42B: 816B total, 42B active MoE, 1,048,576-token context, 62 layers (16 full attention, 46 sliding window) | `zenith/README.md:L1274-1275` |
| MXFP4 checkpoint | `amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4`, 79 GB per MI355X GPU; needs gfx950 (MI355X), as gfx90a has no FP4 | `zenith/README.md:L1279`, `L1298-1299` |
| bf16 checkpoint | `meta-models/Muse-Spark-Open-816B-A42B`, 1.64 TB, 205 GB of weights per GPU on one MI355X node | `zenith/README.md:L1280`, `L1297-1298` |
| draft (speculative decoding) | `Inferact/Muse-Spark-Open-DSpark` at revision `ffc3eb50c8ae7a5a391ddedc43085d2e5249bc35`, local dir `Muse-Spark-Open-DSpark`, method `dspark` | `zenith/bench/muse/spec_drafts_download.sh:L10`, `zenith/bench/muse/muse_ladder_spec.sbatch:L36` |
| model implementation | vLLM's `muse-spark-support` branch (vLLM PR #61879), not in a release. Code under `vllm/models/muse_spark/amd/` (`model.py`, `muse_moe_runner.py`) | `zenith/README.md:L1303`, `zenith/docker/patches/dflash2_fp32_aux_cast.py:L6-7`, `zenith/docker/patches/muse_mxfp4_emulation.py:L7` |
| architecture | MoE Transformer with a 1.8B Perception Encoder (text, image, video and audio in; text out). The language model has hidden size 8192, 128 query / 16 KV heads (GQA, head dim 64), gated attention with QK-norm, 256 experts with 8 active per token, and a 202,048-token vocabulary. The job serves text only (`--language-model-only`) | card |
| registry / `architectures` name of the target model | `TODO:` not in zenith or on the card (the card loads it with `AutoModelForMultimodalLM`) (the only class name found is the draft's, `MuseSparkDSparkForCausalLM`, in `zenith/results/muse/hyperloom_bf16_2026-10-19/curves/hl_patch_dspark_tp_argmax_357284/tree.diff:L95`) | none |
| config / tokenizer / chat template | they ship with the checkpoint; the card describes no chat template. Reasoning effort is set with `reasoning_effort` (or `chat_template_kwargs={"reasoning_effort": ...}`): `low`, `medium`, `high` or `xhigh` (the default); any other value is an error. The job's gsm8k step uses `low`, as zenith's does | card |
| sampling defaults | temperature 1.0, top_p 1.0, top_k off | card |
| reasoning and tool parsers | the `muse` parsers, picked by the zenith launcher for `Muse-Spark*` model names | `zenith/docker/zenith_container_launcher.sh:L812-819` |
| licence | the **Muse Model License** (version "Muse Spark Open", 2026-10-15). It requires "Built with Muse Spark Open" to be shown in products built on the model, and a separate agreement with Meta for Model-as-a-Service offerings above $20M gross revenue. `meta-models/...` (bf16) is **gated**: Meta's access form asks for full legal name, date of birth and organization, and cannot be edited after it is submitted. AMD's MXFP4 repo carries the same licence and shows no gate | card; `zenith/README.md:L1288` |

## 2. Dependency pins

| component | Muse image (`musemain2`) | shipped v8 image (Inkling) | source |
|---|---|---|---|
| vLLM | `0e939af8ad477f52230534fca91053d96e7da59a`: vLLM main `642c18d64` (2026-10-15) merged into `muse-spark-support` `6cc692b5a`. Shipped as the 22 KB git bundle `docker/vllm_src/vllm_musemain_0e939af8.bundle` (head `refs/heads/musemain`) | `bc0f47cd03d6ae99f9f217f096684423bb4ebc2b` | `zenith/README.md:L1303-1309`, `L1510`; `zenith/docker/vllm_src/README.md` (table) |
| AITER | `a3158e4800b6080e3adb25eb58bc48686bc1859c` | `b4d9154d125e09efbe098d986e40fea3549c1244` | `zenith/README.md:L1309`, `L1511`; `zenith/docker/build_image.sh:L59-61` |
| base image | `rocm/pytorch:rocm7.14_ubuntu26.04_py3.14_pytorch_release_2.12.0` (PyTorch 2.12.0, Python 3.14), the same for both | same | `zenith/README.md:L1508` |
| image tag / tar | `ZENITH_IMAGE_TAG=musemain2`, `inference_image_gfx950_musemain2.tar` | `inference_image_gfx950_v8mi250_20261011.tar` | `zenith/README.md:L1309-1310` |
| fix-ups | every patch in `zenith/docker/patches/` applies on this tree: leave `ZENITH_BUILD_SKIP_PATCHES` empty | the v8 set in `lux-inference/docker/patches/` | `zenith/README.md:L1313`; `zenith/docker/Dockerfile.zenith.inference:L983-990` |

A prebuilt `musemain2` tar is shared on Lux at
`/lustre/orion/world-shared/stf218/junqi/Genesis-Hackathon/containers/inference_image_gfx950_musemain2.tar` (maintainer).
`TODO:` its sha256 (`VERIFY=1 bash hpc_jobs/muse-serve/setup_lux.sh` prints it on Lux).

## 3. Setup on Lux

**Nothing to set up for the default.** `hpc_jobs/muse-serve/cluster_defaults.json` reads the shared Genesis-Hackathon
package (maintainer):
- `LUX_MODELPATH=/lustre/orion/world-shared/stf218/junqi/Genesis-Hackathon/models`, which holds
  `Muse-Spark-Open-816B-A42B-Quark-MXFP4/`;
- `LUX_IMAGE_TAR=.../containers/inference_image_gfx950_musemain2.tar`.

`setup_lux.sh` checks both before every submission. Hand runs read the same paths (`env.lux.sh`: `MUSE_MODELPATH`,
`MUSE_IMAGE_TAR`). The DSpark draft must sit in the same directory as the model, because the engine mounts that one
directory as `/app/models`. `SPEC=auto` (the default) uses the draft when `Muse-Spark-Open-DSpark/.verified` is
there, and otherwise serves without it, which is slower, and says so in `summary.txt`.
`TODO:` the DSpark draft is not known to be in the shared directory. Its owner adds it with
`scripts/muse.sh download <that dir> drafts`.

**Walkthrough: your own copy** (bf16, or a private MXFP4). When the agent is asked for the bf16 checkpoint, or
setup reports a model MISSING, it walks the user through these steps. They run once, on a Lux login node, as the user:

1. **bf16 only: get access.** Open https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B, log in and submit
   Meta's access form (full legal name, date of birth, organization; it cannot be edited later). This accepts the Muse
   Model License. Then make a read token at https://huggingface.co/settings/tokens. MXFP4 and the drafts need neither.
2. **Network and CLI.** Lux reaches Hugging Face through the OLCF proxy (maintainer; the guide lists the same proxy for
   its Kubernetes side, L2865-2869):
   ```bash
   export https_proxy=http://proxy.ccs.ornl.gov:3128 http_proxy=http://proxy.ccs.ornl.gov:3128
   python3 -m pip install --user 'huggingface_hub[cli]==1.33.0'   # the pin in zenith bench/eval_requirements.txt:L30
   hf auth login                                                  # bf16 only: paste the token (stored in ~/.cache/huggingface)
   ```
3. **Pick a directory** on Orion with room: bf16 is 1.64 TB (`zenith/README.md:L1280`). Use
   `/lustre/orion/<projid>/scratch/<userid>/hfmodels` (the guide's Orion scratch layout, L229). Remember that
   scratch is purged.
4. **Download.** `scripts/muse.sh download <dir> bf16` (or `mxfp4`) sets the proxy and runs zenith's commands
   (`zenith/README.md:L1288-1294`). They are resumable, so rerun after an interruption:
   ```bash
   hf download meta-models/Muse-Spark-Open-816B-A42B     --local-dir <dir>/Muse-Spark-Open-816B-A42B
   hf download amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4 --local-dir <dir>/Muse-Spark-Open-816B-A42B-Quark-MXFP4
   bash hpc_jobs/muse-serve/lux-inference/bench/muse/spec_drafts_download.sh <dir>   # public; curl; sha256 -> <draft>/.verified
   ```
   The draft script also fetches two DFlash2 drafts (`spec_drafts_download.sh:L11-12`); the job never uses them (§6).
5. **Point the job at it** with `MODELPATH=<dir>`: `script_args="MODEL=Muse-Spark-Open-816B-A42B MODELPATH=<dir>"`
   in VISTA, or `scripts/muse.sh MODEL=... MODELPATH=<dir>` by hand. The job accepts only `/lustre/orion/...` paths.
   It checks `<dir>/<MODEL>/config.json` and the draft's `.verified`.

Optional: `bench/muse/mxfp4_r63_mirror.sh <dir>` builds the `-Quark-MXFP4-r63` copy the validated run used, which the
job accepts as `MODEL=...-Quark-MXFP4-r63`. It re-downloads shard `model-00063-of-00065` away from a Lustre OST that
hung, and the files are identical (`zenith/README.md:L1296-1297`; `zenith/bench/muse/mxfp4_r63_mirror.sh:L2-4`).

**Image fallback.** If the shared tar is unreadable, `BUILD_IMAGE=1`, with a 4 h duration, builds one on the job's
GPU node. It runs the engine's `build_image.sh gfx950` with zenith's Muse pins (`zenith/README.md:L1306-1310`):
- `VLLM_HASH=0e939af8...` and `VLLM_BUNDLE=vllm_musemain_0e939af8.bundle`;
- `AITER_HASH=a3158e48...` and `ZENITH_IMAGE_TAG=musemain2`;
- `ZENITH_BUILD_SKIP_PATCHES` empty (L1313).

The build sets the proxy above for the download of the base image and packages. It writes
`$VISTA_JOB_DIR/images/inference_image_gfx950_musemain2.tar` under a `flock` and is used only if it printed
`=== zenith image build checks passed ===` (L1533-1535). A build takes ~36 min (allow ~1 h), and it must run on a GPU
node (L1523-1528).

## 4. Launch

The job's defaults reproduce zenith's release recipes. They live in the engine's `lux_tuned_defaults.sh`
(the Muse case), not in a site sbatch. All Muse serves share these settings: `VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE=bf16`, `VLLM_USE_BREAKABLE_CUDAGRAPH=1`,
`--language-model-only --dtype bfloat16 --no-enable-prefix-caching`, no MTP (`ZENITH_MTP=0`), `ZENITH_HL_OVERLAY=0`,
and attention backend `ROCM_AITER_FA` on MI355X (`zenith/README.md:L1316-1318`). The sbatch sets them
(`zenith/bench/muse/muse_ladder_spec.sbatch:L52`, `L54`, `L64-66`). It adds the following:

| setting | value | source (`muse_ladder_spec.sbatch`) |
|---|---|---|
| expert parallel | `--enable-expert-parallel` unless `MUSE_EP=0` | L67 |
| quick-reduce | `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT8` by default (`MUSE_QR=` turns it off) | L63-64 |
| Model Runner V2 | `VLLM_USE_V2_MODEL_RUNNER=1` whenever a draft is used (required for DSpark) | L64; `zenith/results/muse/hyperloom_bf16_2026-10-18/README.md:L75` |
| draft | `--speculative-config {"method":"dspark","model":"/app/models/Muse-Spark-Open-DSpark","num_speculative_tokens":7...}` | L68 |
| graphs | `--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","max_cudagraph_capture_size":$MUSE_CG_MAX}` | L66 |
| container storage | `/mnt/bb/$SLURM_JOB_ID/containers/{storage,tmp}` (the guide's NVMe slice, guide "NVMe Usage") | L47-48 |
| context / KV / seqs | `ZENITH_MAX_MODEL_LEN=1048576`, `ZENITH_TARGET_KV_GB` (default 170), `ZENITH_MAX_NUM_SEQS` (default 2048) | L53-56 |

The Muse case sets the same values for each checkpoint. **MXFP4, one MI355X node, the validated best**
(`zenith/README.md:L1326-1335`). In VISTA it is `submit_hpc_job(job="vllm-skill/muse-serve", cluster="lux")`;
upstream it is:

```bash
ZENITH_TARGET_KV_GB=180 MUSE_SPEC=dspark MUSE_CG_MAX=1024 MUSE_MNBT=8192 \
    MUSE_EXTRA_ENV=ZENITH_MUSE_FLYDSL_TUNED=1 \
    MUSE_SPEC_EXTRA=',"num_speculative_tokens_per_batch_size":[[1,4,7],[5,16,4],[17,256,3],[257,2048,0]]' \
    sbatch bench/muse/muse_ladder_spec.sbatch
```

Release re-run: job 360800, score 71.96, probe 20/20, gsm8k 0.976 (250). Its full launch environment is in
`zenith/results/muse/release_validation_2026-10-20/lux/360800/config.txt` (`release_validation_2026-10-20/README.md`,
table). `MUSE_QR=INT6` scores a little higher but changes the math (`zenith/README.md:L1335`, `L1386-1390`).

**bf16, one node per replica** (`zenith/README.md:L1337-1345`). In VISTA it is
`script_args="MODEL=Muse-Spark-Open-816B-A42B"`; upstream it is:

```bash
MUSE_MODEL=Muse-Spark-Open-816B-A42B MUSE_SPEC=dspark MUSE_EP=0 MUSE_QR= MUSE_MNBT=4096 MUSE_CG_MAX=512 \
    MUSE_SPEC_EXTRA=',"num_speculative_tokens_per_batch_size":[[1,8,7],[9,256,3],[257,2048,0]]' \
    MUSE_EXTRA_ENV=VLLM_TUNED_CONFIG_FOLDER=/opt/zenith_patches/muse_bf16_moe_configs \
    ZENITH_TP=8 ZENITH_PP=1 ZENITH_TARGET_KV_GB=60 ZENITH_MAX_NUM_SEQS=512 \
    MUSE_CONCS="1 2 4 8 16 32 64 128 256" sbatch bench/muse/muse_ladder_spec.sbatch
```

Release re-run: job 361594, score 56.87, gsm8k 0.984 (`release_validation_2026-10-20/README.md`, table).

For MXFP4, `lux_tuned_defaults.sh` reproduces 360800's `ZENITH_VLLM_EXTRA_ARGS` exactly, and the same set of
`ZENITH_VLLM_EXTRA_ENV` variables. The job serves an OpenAI-compatible endpoint at `http://<head>:8000/v1`, like
zenith's (`muse_ladder_spec.sbatch:L70`). `BENCH=gsm8k` runs zenith's release check, 250 questions at effort low and
4,096 tokens (`muse_ladder_spec.sbatch` gsm8k step). The 8k/1k throughput ladder is not ported. Single-node Muse does
not need the RCCL-ANP plugin (`zenith/README.md:L1545-1546`), so the job runs on 1 node only.

## 5. The port: `muse-serve` vs zenith v0.1.9 and vs `inkling-serve`

`hpc_jobs/muse-serve/lux-inference/README.md` lists every file kept, renamed and added. In short:

- **Kept from zenith:**
  - `docker/` byte-for-byte, including `patches/` (`diff -r` is empty) and `vllm_src/`, the `musemain` bundle;
  - `lux_container_launcher.sh`'s `Muse-Spark*) echo muse ;;` (zenith `docker/zenith_container_launcher.sh:L819`);
  - Muse's patches (the list below);
  - the bench clients;
  - `bench/muse/spec_drafts_download.sh` and `mxfp4_r63_mirror.sh`.
- **Renamed Zenith -> Lux,** as in the v8 engine:
  - `submit_zenith.slurm.sh`, `zenith_container_launcher.sh`, `zenith_tuned_defaults.sh`, `zenith_podman_env.sh`,
    `zenith_ray_util.py` and `Dockerfile.zenith.inference` become `submit_lux.slurm.sh`, `lux_container_launcher.sh`
    and so on;
  - the job and log names and the `zenith:` log prefix become `lux`.

  The `ZENITH_*` knobs, `/opt/zenith_patches` and the patch files keep their names, because the image and the
  patched vLLM read them.
- **Added for Lux:**
  - the Muse case in `lux_tuned_defaults.sh`, with the new knob `ZENITH_MUSE_DRAFT`;
  - the v8 engine's `serve_lux.slurm.sh` wrapper, with a `gsm8k` step;
  - `ZENITH_STATE_DIR`;
  - `bash`-started scripts;
  - the VISTA job (`job.lux.slurm`, `setup_lux.sh`, `cluster_defaults.json`, `summarize.sh`);
  - `scripts/muse.sh`.
- **Dropped:**
  - zenith's `bench/muse/*.sbatch`, which hard-code account `ven136` and `/lustre/orion/ven136/...`
    (`muse_ladder_spec.sbatch:L2`, `L10`, `L29`, `L44`);
  - the throughput ladder;
  - every non-Lux launcher.

What the `inkling-serve` (v8) engine lacks for Muse, and why Muse has its own job and engine:
- **the build path:** no `docker/vllm_src/`; no `VLLM_BUNDLE` or `AITER_HASH` (zenith `docker/build_image.sh:L59-66`);
  no bundle fetch (zenith `Dockerfile.zenith.inference:L756-766`) and no `ZENITH_BUILD_SKIP_PATCHES` (L983-990);
- **the patches:** `muse_flydsl_tuned_fmoe.py`, `muse_bf16_moe_configs/`, `muse_mxfp4_emulation.py`,
  `rocm_mxfp4_emulation_routed.py`, `dflash_k0_skip_draft.py`, `dflash2_fp32_aux_cast.py`,
  `spec_decode_pp_greedy_warmup.py`, `vllm_custom_ar_capture_copyin.py` and `hl_overlay_0e939af8/`. Eleven shared
  patch files also differ in content, among them `zz_hl_overlay.py` and `pp_broadcast_deferred_recv.py`;
- **the `muse` parser case.**

VISTA uploads each job folder on its own (`RUN_DIR_Lux`), so `muse-serve` carries its own engine (2.3 MB) rather
than sharing the v8 one.

## 6. Limitations

- **One node per endpoint.** Up to about 256 concurrent 8k/1k requests per node; above that, add 1-node replicas
  behind a router. If one endpoint must take c up to 1024, use 2 nodes, TP8 x PP2 (`zenith/README.md:L1347-1351`;
  `zenith/results/muse/hyperloom_bf16_2026-10-18/README.md:L88-91`).
- **PP2 needs `ZENITH_PP_DEFERRED_RECV=0`.** Otherwise it hangs in warmup
  (`zenith/results/muse/lux_2node_pp2_mxfp4_1m_2026-10-16/README.md:L15-16`).
- **No DP2, no EP16:** `muse_moe_runner.py` requires DP1 on ROCm (same README `L18-19`). TP16 across 2 nodes is
  refused by the launcher (`zenith/results/muse/lux_bf16_nodes_2026-10-18/README.md:L35-37`).
- **Do not use the DFlash2 drafts.** Both corrupt the target's output: NVIDIA's from the first request, RadixArk's
  once about 300 requests run at once (`zenith/results/muse/spec_decode_mxfp4_2026-10-17/README.md:L25-26`). Use DSpark.
- **bf16 with the draft:** do not raise KV above 60 GB with graphs to 512; KV 63 ran out of memory in warmup
  (`hyperloom_bf16_2026-10-18/README.md:L185`).
- **MXFP4 needs gfx950.** MI355X has it, so every Lux node qualifies (`zenith/README.md:L1298-1299`).
- **No MTP and no prefix caching** in any Muse serve (`zenith/README.md:L1316-1317`).
- These did not help (`zenith/README.md:L1396-1399`, `L1411-1413`):
  - an fp8 KV cache;
  - TP MoE for MXFP4;
  - EP8 for bf16;
  - the unified attention backend;
  - drafting above 256 requests.

## 7. Troubleshooting

- **Workers die while loading with an NCCL "remote process exited":** the job got the default memory, not the node's.
  Keep `--exclusive`, or add `--mem=0` (`zenith/README.md:L174-178`). The guide also documents `--exclusive`
  (guide, "Node sharing").
- **HIP out-of-memory in the DSpark speculator at c = 128 (bf16), then HTTP 500s:** seen once (job 360801) and not
  reproduced. Resubmit (`zenith/results/muse/release_validation_2026-10-20/README.md`, "lux bf16, first attempt").
- **`draft ... not downloaded/verified`:** rerun `spec_drafts_download.sh`; it resumes and skips verified drafts
  (`spec_drafts_download.sh:L5`).
- **Warmup hangs on 2 nodes (PP2):** set `ZENITH_PP_DEFERRED_RECV=0` (§6).
- **`no image at ...; resubmit with BUILD_IMAGE=1`:** the first run has to build it (§3 step 3).
- **`=== BUILD FAILED`:** read `build.log` in the job's output. The partial tar is removed, so a resubmit starts clean.
- **Cleanup removes every container of yours on the node** (`${PODMAN} rm -af`, as in zenith's sbatch,
  `muse_ladder_spec.sbatch:L143`, and the v8 wrapper). The job's nodes are exclusive.

## 8. CUDA / NVIDIA assumptions (ROCm check)

Nothing in the Muse path needs NVIDIA hardware or CUDA. The image is ROCm 7.14 (`zenith/README.md:L1508`) and
MXFP4 runs only on gfx950. The NVIDIA-looking names are:

- `VLLM_USE_BREAKABLE_CUDAGRAPH`, `cudagraph_mode` and `max_cudagraph_capture_size` are vLLM's names; on ROCm they
  mean HIP graphs (`muse_ladder_spec.sbatch:L64-66`).
- `CUDA_VISIBLE_DEVICES` is set alongside `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES`
  (`zenith/docker/run_container.sh:L649-651`).
- `NCCL_*` variables are read by RCCL.
- `nvidia/Muse-Spark-Open-816B-A42B-DFlash2` is a draft model published by NVIDIA, not a CUDA dependency. It is
  excluded for correctness, not for its platform (§6).
- `--gres=gpu:8` (`muse_ladder_spec.sbatch:L6`; `zenith/README.md:L1528`) is not in the guide, which documents
  `--gpus`, `--gpus-per-node` and `--gpus-per-task`. Both engines' srun still asks for `--gres=gpu:8`, so both
  `job.lux.slurm` files unset `SLURM_GPUS_PER_NODE`, because Slurm refuses the two together.
