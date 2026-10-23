---
name: autogen-skill
description: Run AutoGen 0.2 multi-agent workflows headless on OLCF Lux against a vLLM endpoint. It runs the hackathon's four-agent ICF (inertial confinement fusion) capsule optimizer, one campaign or a laser-energy sweep, with the best design's implosion movie. The LLM is a vllm-skill Inkling server or gpt-oss-20b served in the job. Use when asked to run, sweep or debug AutoGen or the ICF agents on Lux, or to interpret their yields, transcripts and movies.
---

# AutoGen agents on OLCF Lux

This is the hackathon's tutorial 1, a multi-agent scientific workflow, run as batch jobs. Four agents share a
group chat:

- the **Principal Investigator** plans the campaign and decides when it is done;
- the **Design Explorer** proposes capsule designs (shell mass, radius and implosion velocity, at a fixed laser
  energy);
- the **Defect Adversary** attacks each design with drive asymmetry and surface roughness, within a budget;
- an **execution proxy** runs every proposal through `icf_core.py`, a fast 0D implosion model served over MCP.

The result is the design whose *worst-case* yield over the adversary's probes is highest (the "robust yield").
The tutorial's Gradio chat UI is not part of this skill. A job asks the app's own request handler for a campaign
or a sweep, a movie and a debrief, then exits.

Lux: 8x MI355X per node, Slurm partition `batch`. The agents need an OpenAI-compatible LLM:

- **`BACKEND=endpoint` (default):** a running `vllm-skill/inkling-serve` server (Inkling-MXFP4 on 2 nodes, the
  `vllm-skill` skill). The campaign job runs on one whole node (the defaults). A campaign takes ~7 min;
  a sweep takes ~11 min per laser energy.
- **`BACKEND=self`:** the job serves gpt-oss-20b (`gptoss-20b-hedp`) with vLLM on GPU 0 of its own exclusive
  node, from the same v8 image. It is up ~4 min after the job starts, and no other job is needed.

## Quickstart

```bash
cd <this repo>
source env.lux.sh            # ACCOUNT, PKG (hackathon package), ICF_VENV_CACHE, ICF_RUNS_DIR, ICF_GPTOSS_MODEL
scripts/setup.sh             # preflight, seconds: account, the app's files, the BACKEND=self model and image
scripts/campaign.sh                       # 3 MJ campaign against your running inkling-serve / inkling.serve job
scripts/campaign.sh SWEEP_MJ=2:4:1        # sweep: one campaign per laser energy
scripts/campaign.sh BACKEND=self          # no server needed: gpt-oss-20b on the job's own node
cat runs/icf.<jobid>/summary.txt
```

To start an Inkling server, use the `vllm-skill` skill (its `serve.sh` script, job `inkling.serve`). A
campaign submitted while the server is still starting waits for it, up to 40 min.

**Requirements:**

- a Slurm account;
- nothing to install: the first job builds the app's venv on its compute node (python3.11, PyPI through
  `proxy.ccs.ornl.gov:3128`; `ICF_PIP_PROXY` overrides it);
- for `BACKEND=self`, the gpt-oss-20b model and the v8 image in the hackathon package at `$PKG`, or your own
  `ICF_GPTOSS_MODEL` / `ICF_VLLM_IMAGE_TAR`.

No API key: the app's Anthropic backend is switched off.

## Running through VISTA

`hpc_jobs/icf-campaign/` follows VISTA's HPC job format. In a VISTA project, the agent runs it as
`autogen-skill/icf-campaign`, on Lux only.

```text
# 1. an Inkling server (`vllm-skill`), if none is running
submit_hpc_job(job="vllm-skill/inkling-serve", cluster="lux", duration="2:00:00")
# 2. campaigns against it: they find your newest running inkling-serve job, or take SERVE_JOB=<its id>
submit_hpc_job(job="autogen-skill/icf-campaign", cluster="lux", script_args="ENERGY_MJ=3")
submit_hpc_job(job="autogen-skill/icf-campaign", cluster="lux", script_args="SWEEP_MJ=2:4:1 SERVE_JOB=<id>")
# or with no server: gpt-oss-20b inside the job
submit_hpc_job(job="autogen-skill/icf-campaign", cluster="lux", duration="2:00:00", script_args="BACKEND=self")
get_hpc_job_outputs(job_id=<id>, cluster="lux", files=["summary.txt", "debrief.md", "implosion.gif"])
```

| Key | Default | Meaning |
|---|---|---|
| `BACKEND` | `endpoint` | `self`: serve gpt-oss-20b in the job (needs the whole node, as in the defaults) |
| `SERVE_JOB` / `ENDPOINT` | newest running inkling-serve job | the server to use, by job id or OpenAI base URL |
| `ENERGY_MJ` | 3 | laser energy of a single campaign |
| `SWEEP_MJ` | unset | `LO:HI:STEP` in MJ: one campaign per energy instead |
| `MOVIE`, `DEBRIEF` | 1, 1 | implosion movie + time histories; the PI's written analysis |
| `REASONING_EFFORT`, `MAX_TOKENS`, `TEMPERATURE` | medium, 8192, 1.0 (Inkling) | the agents' LLM settings |
| `MAX_DELTA`, `MAX_ROUGHNESS` | 0.3, 3e-6 m | the Defect Adversary's budget |

Defaults (`cluster_defaults.json`): 1 node, 8 GPUs, exclusive, partition `batch`, 1 hour. Before every `sbatch`,
VISTA may run `setup_lux.sh`, which only checks that files are readable. The job builds everything it needs
itself. The first job builds the app's venv on node-local disk (~20 s): the node's NVMe if the job has one,
otherwise `/tmp`, since VISTA's S3M submissions cannot ask for NVMe. It saves the venv as one tar in
`$VISTA_JOB_DIR`; later jobs unpack it in ~1 s. Each job then runs the app's offline tests. No duration needs to
allow for the build.

**Show the movie with `implosion.gif`.** It is the same movie as `implosion.mp4` (480 px, 12 fps), for viewers
that show images but not video.

## Reading the results

`summary.txt` holds:

- the model and settings;
- the **robust yield**: the worst case over the adversary's defect probes of the best design. This is the
  number the campaign optimizes;
- the best single (defect-free or lightly probed) run;
- the best design's parameters;
- for a sweep, a table of laser energy, defect-free yield and robust yield;
- timings.

It ends with `=== CAMPAIGN OK` or `=== CAMPAIGN FAILED: <reason>`. Every early failure writes it too: a bad
argument, no server found, or vLLM not coming up.

- **`debrief.md`** is the PI's analysis. **`transcript.md`** shows how the agents got there: the Explorer's
  proposals with its kinetic-energy check, the Adversary's rationale, and the PI's decisions.
- **`campaign.json`** holds every simulation (design, defect vector, yield) for your own plots.
- **The surrogate is a 0D model** for teaching and agent research, not a design code. Its yields are in MJ of
  fusion energy, and ignition-like jumps are steep. Report a design's robust yield, not its best single run: the
  adversary's job is to show where a design falls off the cliff.
- **Few probes make a weak robust yield.** The summary gives the number of probes behind it. With 1-2 probes, the
  "worst case" is optimistic.

## Reference results

These are real runs on Lux, October 2026. `results/` holds their `summary.txt`, `debrief.md` and plots. The
agents are stochastic (temperature 1.0), so designs and yields vary from run to run. The numbers below give
the scale, not targets.

| Run | Robust yield | Best single run | Sims | Time |
|---|---|---|---|---|
| Inkling-MXFP4, 3 MJ (job 338305) | 7.19 MJ (3 probes) | 8.69 MJ | 25 | campaign 430 s, movie 20 s, debrief 14 s |
| gpt-oss-20b `BACKEND=self`, 3 MJ (job 338312) | 6.20 MJ (4 probes) | 8.18 MJ | 24 | vLLM up in ~4 min; campaign 167 s |
| Inkling-MXFP4, sweep 2:4:1 (job 338441) | 4.34 / 7.32 / 10.88 MJ | defect-free 5.13 / 8.48 / 12.58 MJ | 111 | 1985 s (~11 min per point) |

From the sweep, the robust yield grows roughly linearly with laser energy. The defects cost 13-15 % at every
point. The movie is 4 s: 48 frames at 12 fps, 720x720 (`implosion.mp4`, ~350 kB). The GIF is ~650 kB.

## Layout

```
SKILL.md, env.lux.sh, LICENSE          this file; site layer; Apache-2.0
scripts/setup.sh, scripts/campaign.sh  hand-run preflight and submit (the same job VISTA runs)
hpc_jobs/icf-campaign/                 the VISTA job folder
  job.lux.slurm        the job: finds or starts the LLM, waits for it, runs run_campaign.py
  run_campaign.py      headless driver over the app's handle_request (keyword routing), writes the outputs
  setup_lux.sh         optional preflight: the app's files, the self-serve model and image (builds nothing)
  cluster_defaults.json, README.md
  icf_4agent/          the tutorial app (AMD, MIT): app.py, icf_core.py, icf_mcp_server.py, icf_viz.py,
                       start_vllm_lux.sh (the gpt-oss server), tiktoken/, requirements + constraints, tests/
results/               reference runs
```

## Gotchas

- **`BACKEND=self` needs the whole node.** `start_vllm_lux.sh` numbers GPUs as the node does, so in a shared
  1-GPU job it could land on someone else's GPU. The job refuses unless it has all 8 (`SLURM_GPUS_ON_NODE`).
  Note that `ROCR_VISIBLE_DEVICES` is `0` in any 1-GPU job, so it says nothing about which GPU.
- **The v8 image's tag** is `localhost/inference_image_gfx950:v8mi250_20261011`, not `:latest`, which
  `start_vllm_lux.sh` assumes. The job sets `ICF_VLLM_IMAGE_REF`.
- **gpt-oss needs `tiktoken/o200k_base.tiktoken`** offline. It is in `icf_4agent/tiktoken/`, and
  `start_vllm_lux.sh` mounts it.
- **Routing is fixed to keywords** (`ICF_ROUTER=keywords`). With LLM routing, a request could be answered as chat
  instead of running a campaign. The driver checks the route before it starts.
- **ffmpeg** comes from the `imageio-ffmpeg` wheel in the venv (the job links it as `ffmpeg` on PATH). The
  tutorial's own 160 MB `bin/ffmpeg` is not shipped.
- **The uploaded files are read-only, and the app writes `media_output/` beside itself.** The job runs from a copy
  in `$VISTA_OUT/app`.
- **Upload drops the exec bit,** so every script is run through `bash`.
- **Everyone on one Inkling server shares its throughput.** Many campaigns at once against a server that is also
  running the science benchmark will be slow, but not wrong.
- **The venv is never built on Lustre.** From a compute node, Lustre takes ~35 ms per small file written,
  against ~1.4 ms from a login node (same client settings and layout), and slows further when two jobs write at
  once. A venv is ~22,000 files: a build or unpack on Lustre took over 90 min, where node-local disk takes 20 s.
  The job builds on the NVMe (`/mnt/bb/<jobid>`) if it has one, else on `/tmp`, which is a private, memory-backed
  overlay per job (the ~650 MB venv is removed at exit). `ICF_NVME=0` forces the `/tmp` path for testing. It
  saves one tar, `venv-<hash of the pins>.tar`, in `$VISTA_JOB_DIR` (by hand, in `work/`). Changing
  `requirements.txt` or `constraints.txt` changes the hash, and the next job rebuilds. A broken tar is fixed by
  deleting it.
- **One node only.** The whole job runs on its first node (no `srun`), with the venv on that node's local disk.
  With `node_count` above 1, the extra nodes sit idle and `job.log` says so.
- **The venv moves between jobs** (its directory has the job id in it), so the job runs it only as
  `$VENV/bin/python`.
  Its console scripts (`pip`, `gradio`, ...) have stale paths, so call them as `python -m`.
- **PyPI is reachable from compute nodes only through `proxy.ccs.ornl.gov:3128`.** The job sets it for pip alone.
  Exported to the whole job, it would also send the agents' requests to the vLLM server through the proxy.
- **`BACKEND=self` is reproducible run to run.** vLLM serves with `seed=0`, so a fresh server given the same
  prompts samples the same replies. Two such jobs gave identical campaigns. Against a shared Inkling server,
  other users' requests change the batching, so runs vary.

## Verify after any change

Run `scripts/setup.sh`: it ends with `setup OK`. To rebuild the venv from scratch, delete `work/venv-*.tar`. Then run:

- `scripts/campaign.sh ENERGY_MJ=3` against a running server;
- `scripts/campaign.sh BACKEND=self`.

Both should end with `=== CAMPAIGN OK` and produce `implosion.mp4`, `implosion.gif` and `time_histories.png`.

## License and acknowledgments

The skill's own code is Apache-2.0 (`LICENSE`). `hpc_jobs/icf-campaign/icf_4agent/` is AMD's tutorial app under
its MIT `LICENSE`. gpt-oss-20b and Inkling keep their own licenses.
