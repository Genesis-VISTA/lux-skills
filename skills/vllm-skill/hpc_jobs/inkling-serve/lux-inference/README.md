# lux-inference: the vLLM serving engine for Lux

A curated, Lux-only copy of the Zenith v8 inference package (`zenith_inference_v8_lux`, final release 2026-10-12,
sha256 `d3a0fc86...7f648f7e` in the hackathon package's `tutorials/02_zenith_inference/CHECKSUMS.txt`). It runs on the
image `inference_image_gfx950_v8mi250_20261011.tar` (sha256 `e03d1693...8f40713`). It is the engine of the VISTA job
`vllm-skill/inkling-serve` (`../job.lux.slurm`), which the skill's `scripts/` also submit by hand (see
`../../../SKILL.md`); this file records what changed from upstream.

## What was kept

| Path | Purpose |
|---|---|
| `docker/serve_lux.slurm.sh` | Entry point: Lux site setup, fabric check, correctness probe, then serve (or benchmark, `LUX_BENCH`). |
| `docker/submit_lux.slurm.sh` | All serving defaults; one `lux_container_launcher.sh` per node via srun. |
| `docker/lux_container_launcher.sh`, `run_container.sh`, `run_node.sh` | Per-node podman container, Ray cluster, RCCL health check, KV sizing, `vllm serve`. |
| `docker/lux_tuned_defaults.sh` | Tuned defaults for Inkling-MXFP4 on 1 / 2 nodes and Mistral-Large-2411 on 1 node. |
| `docker/lux_podman_env.sh` | Rootless podman storage (burst buffer on Lux). |
| `docker/lux_ray_util.py`, `ray_health_*.py`, `kv_cache_sizing.py` | Helpers copied into the container / run on the host. |
| `docker/patches/` | vLLM source patches, RCCL-ANP plugin (`anp/`), tuned gfx950 kernels (`hl_overlay/`). **Byte-identical to upstream.** |
| `docker/Dockerfile.lux.inference`, `build_image.sh` | How the image was built (reference; rebuilding is not needed). |
| `bench/science/` | Physics / chemistry / materials-science suite (GPQA Diamond, MaScQA, MMLU-Pro). |
| `bench/inkling/` | Correctness probe, MMLU-Pro 280-q subset, 1M-token needle check, throughput client. |
| `bench/mistral/` | Serve Mistral-Large-2411 (or a fine-tune) and score it in one job. |
| `docs/LUX_HACKATHON.md`, `docs/LUX_MISTRAL_LARGE.md` | The upstream tutorial guides, with every measured number. |
| `../../../results/` | The Lux shipping-gate results (Inkling 2026-10-12, Mistral 2026-10-11), scores only. Kept outside the job folder, so VISTA does not upload them with every job. |

Dropped: Flux / Tioga / MI300A / MI250X launchers and docs, disaggregated-serving demos, `diag/`, the
upstream-contribution notes, and results from other systems.

## What was renamed

| Upstream | Here |
|---|---|
| `submit_zenith.slurm.sh` | `submit_lux.slurm.sh` (job name `lux.inference`, log `lux.<jobid>.out`) |
| `zenith_container_launcher.sh` | `lux_container_launcher.sh` |
| `zenith_tuned_defaults.sh` | `lux_tuned_defaults.sh` (log line `lux: tuned defaults for ...`) |
| `zenith_podman_env.sh` | `lux_podman_env.sh` |
| `zenith_ray_util.py` | `lux_ray_util.py` (imports updated) |
| `Dockerfile.zenith.inference` | `Dockerfile.lux.inference` |

**Not renamed, on purpose:** the `ZENITH_*` environment knobs, `/opt/zenith_patches`, `.zenith_state/` and
`/tmp/zenith-xdg-*`. The image was built with its patches applied at `/opt/zenith_patches`, and the patched vLLM
inside it reads `ZENITH_*` variables (`ZENITH_PP_DEFERRED_RECV`, `ZENITH_HL_OVERLAY`, ...). Renaming them would
silently turn those switches off. Instead, `serve_lux.slurm.sh` and `bench/mistral/mistral_bench.slurm.sh` copy
every `LUX_<KNOB>` onto `ZENITH_<KNOB>` at the start (the `LUX_` value wins), so `LUX_MTP=1` and `ZENITH_MTP=1`
mean the same thing. The docs in `docs/` still use the `ZENITH_` names.

## Added

- `serve_lux.slurm.sh`: the `LUX_*` alias loop; `ZENITH_RUNS_DIR` (output under `<dir>/serve.<jobid>/` instead of
  `docker/serve.<jobid>/`); `LUX_BENCH="science mmlu needle"` to benchmark and exit instead of serving.
- `mistral_bench.slurm.sh`: the `LUX_*` alias loop.
- `lux_container_launcher.sh` and `serve_lux.slurm.sh`: `ZENITH_STATE_DIR`, where the per-job `.zenith_state/<jobid>/`
  lives (default beside the scripts, as upstream), so the VISTA job can keep its sources read-only.
- `patches/anp/`: the two libionic symlinks are not shipped (a VISTA skill import refuses symlinks); `job.lux.slurm`
  recreates them in its per-job copy of the engine. See `docker/patches/anp/README.md`.

## Acknowledgments

This engine, and the serving recipe the skill is built on, is adapted from AMD's **Zenith** inference package for AMD
Instinct GPUs (`zenith_inference_v8_lux`, release 2026-10-12). AMD built the image, the vLLM patches, the RCCL-ANP
fabric fix, the tuned gfx950 kernels, the tuned defaults and the benchmark clients, and measured every number in
`docs/` and `results/`. The Lux adaptation (renames, the `LUX_*` aliases, `ZENITH_RUNS_DIR`, `LUX_BENCH` and the
skill's `scripts/`) was done at the Oak Ridge Leadership Computing Facility. Errors introduced by that adaptation
are ours, not AMD's.

## Licensing

The skill's own code (`../scripts/`, `../env.lux.sh`, `../SKILL.md` and the Lux changes listed under "What was
renamed" and "Added") is licensed under the Apache License 2.0, in `../LICENSE`. Everything else keeps its own
terms:

| Component | Where | Licence |
|---|---|---|
| Zenith engine files adapted from AMD (`docker/`, `bench/`, `docs/`, `results/`) | this directory | AMD's terms for the Zenith package; copyright stays with AMD and its contributors |
| vLLM (in the image); `docker/patches/` modifies it | the image, `docker/patches/` | vLLM is Apache-2.0 (copyright the vLLM contributors); patch files carrying an `SPDX-License-Identifier` header follow it, the rest are AMD Zenith files (row above) |
| amd-anp RCCL network plugin (`librccl-anp.so`) | `docker/patches/anp/` | BSD-style RCCL licence, `docker/patches/anp/LICENSE-amd-anp.txt` (NVIDIA, AMD, Microsoft) |
| libionic | `docker/patches/anp/` | GPL-2.0 or Linux-OpenIB; redistributed under Linux-OpenIB, `LICENSE-libionic-Linux-OpenIB.txt` (GPL text also included) |
| ROCm, PyTorch, AITER, flash-attention, RCCL in the image | `inference_image_gfx950_v8mi250_20261011.tar` | their upstream licences, inside the image |
| Inkling-MXFP4 (and Inkling-bf16) | `$LUX_MODELPATH` | Apache-2.0 (model card in the model directory) |
| Mistral-Large-Instruct-2411 and anything fine-tuned from it (`bench/mistral/`) | `$LUX_MODELPATH` | Mistral Research Licence: research and non-commercial use only; derivatives carry the same licence |
| GPQA Diamond | `$LUX_DATASETSPATH/science/` | CC BY 4.0, [idavidrein/gpqa](https://github.com/idavidrein/gpqa) |
| MaScQA | `$LUX_DATASETSPATH/science/` | CC BY-NC-SA 4.0 (non-commercial), [M3RG-IITD/MaScQA](https://github.com/M3RG-IITD/MaScQA) |
| MMLU-Pro | `$LUX_DATASETSPATH/science/`, `bench/inkling/` | MIT, [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) |

**GPQA:** the authors ask that the questions not be posted online, to keep them out of training crawls. Do not
copy `gpqa_diamond.jsonl` or per-question result files (`science*.json`, ignored by `../.gitignore`) to the web or
to a public git repository. Aggregate scores are fine.
