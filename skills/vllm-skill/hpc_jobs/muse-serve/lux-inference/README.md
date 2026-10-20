# lux-inference (Muse): the vLLM serving engine for Muse Spark Open on Lux

A curated, Lux-only copy of AMD's Zenith inference package **v0.1.9** (github.com/jbelofamd/zenith, tag `v0.1.9`,
commit `70e5f0bf136e9b236251da24373b0241c6b40d4e`), the first release that serves Muse Spark Open. It is the engine of
the VISTA job `vllm-skill/muse-serve` (`../job.lux.slurm`). It runs on the image
`inference_image_gfx950_musemain2.tar` (vLLM `0e939af8` from `docker/vllm_src/`, AITER `a3158e48`; zenith
`README.md` §1d.2). A copy is shared in the Genesis-Hackathon package, and the job can rebuild it from this directory
with `BUILD_IMAGE=1`. The Inkling engine of
`vllm-skill/inkling-serve` (`../../inkling-serve/lux-inference/`, Zenith v8) is separate and unchanged. This file records
what changed from upstream, in the same way as that engine's README.

## What was kept

| Path | Purpose |
|---|---|
| `docker/serve_lux.slurm.sh` | Entry point: Lux site setup, fabric check (multi-node only), correctness probe, then serve (or benchmark, `LUX_BENCH`). |
| `docker/submit_lux.slurm.sh` | All serving defaults; one `lux_container_launcher.sh` per node via srun. |
| `docker/lux_container_launcher.sh`, `run_container.sh`, `run_node.sh` | Per-node podman container, Ray cluster, RCCL health check, KV sizing, `vllm serve`. The launcher picks the `muse` reasoning and tool parsers for `Muse-Spark*` models. |
| `docker/lux_tuned_defaults.sh` | Tuned defaults: Inkling-MXFP4 (1 / 2 nodes), Mistral-Large-2411 (1 node) and, added here, Muse Spark Open (MXFP4 and bf16, 1 node). |
| `docker/lux_podman_env.sh` | Rootless podman storage (node-local NVMe on Lux). |
| `docker/lux_ray_util.py`, `ray_health_*.py`, `kv_cache_sizing.py`, `stage_host_fabric.sh` | Helpers copied into the container / run on the host. |
| `docker/patches/` | vLLM source patches, including Muse's (`muse_flydsl_tuned_fmoe.py`, `muse_bf16_moe_configs/`, `dflash_k0_skip_draft.py`, ...), the gfx950 kernel overlays (`hl_overlay/`, `hl_overlay_0e939af8/`) and the RCCL-ANP README and build scripts (`anp/`). **Byte-identical to zenith v0.1.9.** |
| `docker/vllm_src/` | The vLLM `musemain` git bundle (`vllm_musemain_0e939af8.bundle`: vLLM main `642c18d64` merged into `muse-spark-support` `6cc692b5a`, PR #61879) and its README. |
| `docker/Dockerfile.lux.inference`, `build_image.sh`, `requirements.txt` | How the image is built (`BUILD_IMAGE=1` runs it). |
| `bench/inkling/` | The correctness probe, gsm8k (zenith's Muse release check) and the MMLU-Pro subset. |
| `bench/science/` | Physics / chemistry / materials-science suite (GPQA Diamond, MaScQA, MMLU-Pro). |
| `bench/muse/` | `spec_drafts_download.sh` (the DSpark draft and two DFlash2 drafts, sha256-checked) and `mxfp4_r63_mirror.sh`. |

Dropped: the Flux / Tioga / MI300A / MI250X launchers, `diag/`, the NIXL / MoRI images, disaggregated serving, the
RCCL tuner, `bench/` clients for other models, the Muse ladder and tuning sbatches (they hard-code another site's
account and paths), `docs/`, `recipes/`, `results/` and `vllm_upstream/`. The RCCL-ANP binaries are not shipped,
upstream or here. Muse serves on one node, where RCCL-ANP is not needed (zenith `README.md` L1545-1546).

## What was renamed

The same renames as the v8 engine:

| Upstream | Here |
|---|---|
| `submit_zenith.slurm.sh` | `submit_lux.slurm.sh` (job name `lux.inference`, log `lux.<jobid>.out`) |
| `zenith_container_launcher.sh` | `lux_container_launcher.sh` |
| `zenith_tuned_defaults.sh` | `lux_tuned_defaults.sh` |
| `zenith_podman_env.sh` | `lux_podman_env.sh` |
| `zenith_ray_util.py` | `lux_ray_util.py` (imports updated) |
| `Dockerfile.zenith.inference` | `Dockerfile.lux.inference` |
| log prefix `zenith:` | `lux:` |

**Not renamed, on purpose:**
- the `ZENITH_*` knobs, `/opt/zenith_patches`, `.zenith_state/` and `/tmp/zenith-xdg-*`;
- the files in `docker/patches/` and `docker/vllm_src/`;
- the build's `=== zenith image build checks passed ===` line.

The image bakes the patches in at `/opt/zenith_patches`, and the patched vLLM reads `ZENITH_*` variables (for example
`ZENITH_MUSE_FLYDSL_TUNED`, `ZENITH_PP_DEFERRED_RECV`). Renaming them would silently turn those switches off.
`serve_lux.slurm.sh` copies every `LUX_<KNOB>` onto `ZENITH_<KNOB>` instead, so `LUX_MUSE_DRAFT=` and
`ZENITH_MUSE_DRAFT=` mean the same thing. The scripts were renamed with
`sed` over the docker/ scripts only; `diff -r` against zenith's `docker/patches/` is empty.

## Added

- `serve_lux.slurm.sh`: the Lux wrapper of the v8 engine (the `LUX_*` alias loop, `ZENITH_RUNS_DIR`, `LUX_BENCH`), whose
  upstream is the same file in zenith v0.1.9. `LUX_BENCH` steps here are `science`, `mmlu` and `gsm8k`. `gsm8k` uses
  250 questions, effort low and 4,096 tokens, the settings of zenith's `bench/muse/muse_ladder_spec.sbatch`.
- `lux_tuned_defaults.sh`: the Muse defaults, one case per checkpoint, keyed on `ZENITH_MODEL` like Inkling's. They are
  the values zenith's `muse_ladder_spec.sbatch` and `README.md` §1d.3 pass by hand. The MXFP4 result matches the
  `ZENITH_VLLM_EXTRA_ARGS` of the validated job 360800 exactly
  (`results/muse/release_validation_2026-10-20/lux/360800/config.txt`). `ZENITH_MUSE_DRAFT` (default
  `Muse-Spark-Open-DSpark`; empty = no draft) is the one new knob.
- `lux_container_launcher.sh`: `ZENITH_STATE_DIR`, as in the v8 engine, so the job keeps its sources read-only.
- `submit_lux.slurm.sh`, `run_node.sh`: the launcher is started through `bash`, as in the v8 engine. A skill upload
  need not keep execute bits.

## Acknowledgments

This engine, the Muse serving recipes and every Muse number the skill quotes are AMD's work (Zenith v0.1.9). That
work covers:
- the image, the vLLM `musemain` merge, the patches and the tuned MoE tables;
- the DSpark K schedules and the benchmark clients.

The Lux adaptation (the renames, the `LUX_*` aliases, the Muse tuned-defaults case, the VISTA job and the skill's
`scripts/`) was done at the Oak Ridge Leadership Computing Facility. Errors introduced by that adaptation are ours,
not AMD's.

## Licensing

| Component | Where | Licence |
|---|---|---|
| Zenith v0.1.9 files (`docker/`, `bench/`) | this directory | MIT, `LICENSE-zenith` (Copyright (c) 2026 The Zenith authors) |
| The Lux changes listed under "What was renamed" and "Added" | this directory | Apache-2.0, the skill's `../../../LICENSE` |
| vLLM (`docker/vllm_src/` bundle, in the image); `docker/patches/` modifies it | `docker/vllm_src/`, `docker/patches/` | vLLM is Apache-2.0 (copyright the vLLM contributors); patch files carrying an `SPDX-License-Identifier` header follow it |
| amd-anp, libionic (built by `docker/patches/anp/build/`, not shipped) | `docker/patches/anp/` | `LICENSE-amd-anp.txt` (BSD-style), `LICENSE-libionic-*.txt` |
| ROCm, PyTorch, AITER and the rest of the image | `inference_image_gfx950_musemain2.tar` (built, not shipped) | their upstream licences |
| Muse Spark Open checkpoints and drafts | `$LUX_MODELPATH` | the Muse Model License (version "Muse Spark Open"; Meta's model card). bf16 is gated by Meta's access form |
