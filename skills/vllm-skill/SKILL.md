---
name: vllm-skill
description: Serve LLMs with vLLM on OLCF Lux (MI355X, Slurm + rootless podman) as an OpenAI-compatible endpoint, by default Inkling-MXFP4 on 2 nodes (TP8 x PP2) at a 1,048,576-token context, or Muse Spark Open (816B MoE) on 1 node, and score models on a physics / chemistry / materials-science suite (GPQA Diamond, MaScQA, MMLU-Pro). Use when asked to serve Inkling, Mistral or Muse Spark on Lux, debug or tune the server, point agents at it, run the science benchmark, or interpret tok/s and accuracy results.
---

# vLLM on OLCF Lux

Lux: `login1.lux.olcf.ornl.gov`, nodes `lux001..504`, 8x MI355X (gfx950, 288 GB) per node, 8x Pollara 400G RoCE
NICs, Slurm partition `batch`. Inference runs in **rootless podman** (the Primus skill uses apptainer; this image is
a 57 GB podman archive). Compute nodes reach the internet only through the OLCF proxy (`http(s)_proxy=http://proxy.ccs.ornl.gov:3128`); serving needs none.

The default is the hackathon's tutorial 2: **Inkling-MXFP4 on 2 nodes, 16 GPUs, TP8 x PP2, full 1M-token
context, reasoning and tool calls on**. The same endpoint is what tutorial 1's four ICF agents use, so a server
started here serves both parts (see "Two-part tutorial").

The engine (`hpc_jobs/inkling-serve/lux-inference/`) is the Zenith v8 package, which was validated on Lux, renamed for
Lux. Everything that runs on Lux is in the VISTA job folder `hpc_jobs/inkling-serve/` (see "Running through VISTA");
the hand-run `scripts/` submit the same job. **Muse Spark Open** has its own job, `hpc_jobs/muse-serve/`, on a port of
AMD's Zenith v0.1.9 (see "Muse Spark Open"). Read "Gotchas" before changing anything.

## Quickstart

```bash
cd <this repo>
source env.lux.sh                   # ACCOUNT, PKG (hackathon package), LUX_MODELPATH / LUX_IMAGE_TAR / LUX_DATASETSPATH
scripts/setup.sh                    # preflight: account, image, model, science data, RCCL plugin, python (seconds)
scripts/serve.sh                    # 2 nodes, Inkling-MXFP4, 1M context. Job "inkling.serve", up in ~11-14 min
export INKLING_URL=$(scripts/endpoint.sh | tail -1)    # waits until it answers -> http://<head>:8000/v1
scripts/summarize.sh <jobid>        # UP / FABRIC OK / KV pool / correctness 10/10 / SERVING
scripts/science.sh "$INKLING_URL" --n 20               # 60-question science smoke test against it
scripts/bench.sh                    # separate job: serve + full science suite + exit (job "inkling.bench")
```

`serve.sh` and `bench.sh` print the job id and log path (`runs/<name>.<jobid>.out`). The job's outputs are in
`runs/serve.<jobid>/`: `summary.txt` and `endpoint.txt` (refreshed every minute), `job.log`, the benchmark reports,
and vLLM's `server.log`. Queue wait varies (`sbatch --test-only` estimates it; `squeue -u $USER`). The server runs
until its time limit (8 h by hand) or `scancel`.

**Requirements:** a Slurm account, and the hackathon package at `$PKG` (default
`/lustre/orion/world-shared/stf218/junqi/Genesis-Hackathon`) or your own `LUX_MODELPATH` (dir holding
`Inkling-MXFP4/`, 528 GB), `LUX_IMAGE_TAR` (`inference_image_gfx950_v8mi250_20261011.tar`) and
`LUX_DATASETSPATH` (holding `science/{gpqa_diamond,mascqa,mmlu_pro_sci}.jsonl`). No HF token, no proxy.

## Running through VISTA

`hpc_jobs/inkling-serve/` follows VISTA's HPC job format (VISTA docs, "Self-contained HPC skills": https://genesis-vista.github.io/docs/modsim-skill).
When this skill is loaded in a VISTA project, the agent runs it as `vllm-skill/inkling-serve`: the skill's name, then
the job's. It serves Inkling and Mistral; for Muse Spark use `vllm-skill/muse-serve` ("Muse Spark Open"). Both run on Lux only.

```text
submit_hpc_job(job="vllm-skill/inkling-serve", cluster="lux", script_args="BENCH=science")            # 2 nodes: serve, full suite, exit
submit_hpc_job(job="vllm-skill/inkling-serve", cluster="lux", script_args="BENCH=science,mmlu SCIENCE_N=20")  # ~25 min smoke test
submit_hpc_job(job="vllm-skill/inkling-serve", cluster="lux", node_count=1, script_args="MODEL=Mistral-Large-Instruct-2411 BENCH=science")
submit_hpc_job(job="vllm-skill/inkling-serve", cluster="lux", duration="4:00:00")                    # serve Inkling-MXFP4 for agents
get_hpc_job_status(job_id=<id>, cluster="lux")
get_hpc_job_outputs(job_id=<id>, cluster="lux", files=["summary.txt", "endpoint.txt"])
```

`script_args` is a space-separated list of `KEY=VALUE` pairs; any other key is refused:

| Key | Default | Meaning |
|---|---|---|
| `MODEL` | `Inkling-MXFP4` | also `Mistral-Large-Instruct-2411` (1 node only) and `Inkling-bf16` (2 nodes; not in the shared package) |
| `BENCH` | unset (serve) | comma-separated `science`, `mmlu`, `needle`: benchmark, then exit |
| `SCIENCE_N` | all | first n questions of each science benchmark; implies `BENCH=science` |
| `MTP` | 0 | 1: MTP speculative decoding (fast for 1-2 users, slower from 4 up) |
| `MAX_MODEL_LEN` | 1,048,576 (2 nodes), 100,000 (1 node) | context length |

Defaults (`cluster_defaults.json`): 2 nodes, 8 GPUs each, exclusive, partition `batch`, 2 hours (VISTA allows at
most 4), and the shared package's model, image and benchmark data. Before every `sbatch`, VISTA runs
`setup_lux.sh` on the login node: it checks that the image, the model and the RCCL-ANP plugin are readable, so a
missing file is an error from the submit call, not a failed job.

Read `summary.txt` (refreshed every minute while the job runs): `=== UP` after 11-14 min (up to ~25 when
Lustre is busy), `=== FABRIC OK`,
the KV cache, correctness 10/10, `=== SERVING at http://<head>:8000/v1`, then any `=== BENCH` steps and scores, and
failure signatures. Compare with "Reference results". Then:

- **Serve mode ends at the time limit,** so Slurm reports the job as TIMEOUT. That is its normal end; a benchmark
  job exits 0 by itself. Confirm the duration with the user: a serve job holds 2 nodes for all of it.
- **The endpoint (`endpoint.txt`) is reachable only from inside Lux**, by agents or clients on its login or compute
  nodes, not from VISTA itself. For tutorial 1's ICF agents, pass it as `INKLING_URL`: VISTA names the job itself, so
  `run_icf_with_inkling.sh` cannot find it as `inkling.serve`.
- `science.json` holds every GPQA question: never fetch it into a shared place or quote it. Share `science.txt`.

## Use the endpoint

Any OpenAI client, model `Inkling-MXFP4`:

```python
from openai import OpenAI
c = OpenAI(base_url="http://<head>:8000/v1", api_key="unused")
r = c.chat.completions.create(model="Inkling-MXFP4", max_tokens=8000,
        messages=[{"role": "user", "content": "Estimate the Debye temperature of copper."}])
print(r.choices[0].message.reasoning)   # the thinking
print(r.choices[0].message.content)     # the answer
```

- The thinking is returned in `reasoning` (streamed as `delta.reasoning`), the answer in `content`. Default effort is 0.9;
  `"chat_template_kwargs": {"reasoning_effort": 0.5}` changes it per request.
- **Leave room for thinking:** ~2,300 tokens on average for MMLU-Pro; a reply cut off mid-thought has `content: null`.
- Tools: `"tools": [...]`, `"tool_choice": "auto"` -> `tool_calls`, `finish_reason: "tool_calls"`.
- Context 1,048,576 tokens (prompt + output). KV pool 10.3M tokens: ~10 full-length requests, or many agent sessions.
- From a login node, `unset http_proxy https_proxy` first if your shell sets them, or curl goes to the proxy.

## Two-part tutorial (agents on this server)

```bash
scripts/serve.sh                                         # part 1: this skill
cp -r $PKG/tutorials/01_icf_4agent/icf_4agent ~/icf_4agent && chmod -R u+w ~/icf_4agent
(cd ~/icf_4agent && ./create_aivenv.sh)                  # once, on login1, ~5 min
$PKG/tutorials/01_icf_4agent/run_icf_with_inkling.sh -A $ACCOUNT ~/icf_4agent   # part 2: the 4-agent ICF app
```

`run_icf_with_inkling.sh` finds your RUNNING job **named `inkling.serve`** (so do not rename it; `bench.sh` uses
`inkling.bench` so the app never latches onto a job that is about to exit), or takes `INKLING_URL`. Agents get
8,192-token replies at medium effort. Everyone on one server shares its throughput; for one or two agents on long
contexts, serve with `LUX_MTP=1` (below).

## Science benchmark suite

| benchmark | questions | tests | licence |
|---|---|---|---|
| GPQA Diamond, physics + chemistry | 179 (86 / 93) | graduate, written to resist look-up, 4 options | CC BY 4.0 |
| MaScQA | 646 | materials science, GATE exams, 14 topics; MC, matching, numeric ranges | CC BY-NC-SA 4.0 |
| MMLU-Pro physics / chemistry / engineering | 3,400 | broad undergraduate, 10 options | MIT |

Zero-shot chain of thought at temperature 0, extraction from "The answer is ...". A reply cut off while thinking is scored
from its `reasoning`. The **suite score** is the unweighted mean of the 3 accuracies. The report also gives Wilson 95%
intervals, per-field accuracy (physics, chemistry, materials, engineering) and counts of truncated, errored and
unanswered replies. Within noise:
<2 points on MMLU-Pro, <4 on MaScQA, <7 on GPQA, about +-2.7 on the suite score.

- `scripts/bench.sh [nodes] [KEY=VAL]`: serve, run `LUX_BENCH` (default `science`; also `mmlu`, the 280-q MMLU-Pro
  subset, and `needle`, 15 cells to 1,040k tokens), stop. `-t` defaults to 2 h (`BENCH_TIME=03:00:00` for more).
  Results go to `runs/serve.<jobid>/{science,mmlu_pro,long_context}.txt`.
- `scripts/science.sh <head|URL> [args]`: same suite from the login node against a live server. Passes through
  `--n N` (first N per benchmark), `--benchmarks gpqa,mascqa`, `--concurrency`, `--effort`, `--max-tokens`.
- Defaults for reasoning models: concurrency 256, `max_tokens` 16384 (upstream's 128 / 8192 were tuned for non-reasoning Mistral).
- Another model: `scripts/bench.sh 1 LUX_MODEL=Mistral-Large-Instruct-2411` (1 node, 1M context via YaRN), or a fine-tune
  through `hpc_jobs/inkling-serve/lux-inference/bench/mistral/mistral_bench.slurm.sh` (see
  `hpc_jobs/inkling-serve/lux-inference/docs/LUX_MISTRAL_LARGE.md` §7).
- **GPQA must stay private:** `science*.json` holds every question and reply (written mode 600, git-ignored). Share
  `science.txt` only.

## Layout

| Path | Purpose |
|---|---|
| `hpc_jobs/inkling-serve/` | **Everything that runs on Lux**, as a VISTA HPC job (see "Running through VISTA"). The hand-run scripts below submit the same files. |
| `  job.lux.slurm` | Launcher: `KEY=VALUE` allow-list, runs the engine from a per-job copy under `$VISTA_OUT`, refreshes `summary.txt` / `endpoint.txt`. No `#SBATCH` lines. |
| `  cluster_defaults.json` | Resources (2 nodes, 8 GPUs, exclusive, `batch`, 2 h) and the shared package's paths. |
| `  setup_lux.sh` | Login-node preflight before every submission; changes nothing. |
| `  README.md`, `summarize.sh` | The job's description for the agent; the summary it writes. |
| `  lux-inference/` | The engine: launchers, patches, RCCL-ANP plugin, tuned kernels, bench clients, upstream docs. Its `README.md` lists what was renamed from Zenith. |
| `hpc_jobs/muse-serve/` | The Muse Spark Open job (same file set; 1 node, `BUILD_IMAGE=1`); its `lux-inference/` is the Zenith v0.1.9 engine. |
| `results/` | The upstream shipping-gate results (scores only), kept out of the job folder so VISTA does not upload them. |
| `env.lux.sh` | Site settings for running by hand. All values overridable via env before sourcing. |
| `scripts/setup.sh` | Preflight only (the job's `setup_lux.sh`, plus account and python); `--verify` adds the image sha256 (~5 min). |
| `scripts/serve.sh` | Submit a server: `[nodes] [KEY=VAL ...]` (any `LUX_*` knob); `SBATCH_ARGS` passes extra sbatch flags. |
| `scripts/bench.sh` | Submit serve + benchmark + exit. |
| `scripts/endpoint.sh` | Find your running `inkling.serve` (or a job id), wait for it, print the base URL. |
| `scripts/science.sh` | Science suite against a live endpoint. |
| `scripts/muse.sh` | Muse: `download <dir>` the weights and drafts (login node), or submit `muse-serve` with its `KEY=VAL` args. |
| `scripts/summarize.sh` | Startup milestones, fabric, KV, correctness, scores and failure signatures for a job. |
| `runs/` | Job logs and results (git-ignored). |

## Knobs

By hand, set them as `KEY=VAL` arguments to `serve.sh` / `bench.sh`, or export them before running. `LUX_<KNOB>` is an
alias for the engine's `ZENITH_<KNOB>`, so every knob in `hpc_jobs/inkling-serve/lux-inference/docs/LUX_HACKATHON.md`
works under either name. Through VISTA only the job's short `script_args` are accepted ("Running through VISTA").

| Knob | Default (2 nodes) | Meaning |
|---|---|---|
| nodes (1st arg) | 2 | 2: TP8 x PP2, 1M context. 1: TP8, 100k context, MTP-3, the tuned max-throughput node |
| `LUX_MODEL` | `Inkling-MXFP4` | `Inkling-bf16` (2 nodes only, 100k), `Mistral-Large-Instruct-2411` (1 node, 1M) |
| `LUX_MTP` | 0 on 2 nodes | 1: MTP speculative decoding, 3 draft tokens (see performance) |
| `LUX_MAX_MODEL_LEN` | 1048576 | 1 node: `LUX_MAX_MODEL_LEN=1048576` gives 1M too (KV pool 4.5M tokens) |
| `LUX_TARGET_KV_GB` | 200 | KV cache per GPU |
| `LUX_REASONING_PARSER` / `LUX_TOOL_CALL_PARSER` | inkling | `none`: the thinking comes back inline in `content` |
| `LUX_TUNED_DEFAULTS` | 1 | 0: generic defaults instead of `docker/lux_tuned_defaults.sh` |
| `LUX_BENCH`, `LUX_SCIENCE_ARGS` | unset | Benchmark mode (see above) |

The launcher prints the values in force: `lux: tuned defaults for Inkling-MXFP4 on 2 nodes (TP8 x PP2): ...`
(`summarize.sh` shows it). Check it before trusting a run that set knobs.

## Reference results

Two sources: the Zenith v8 shipping gate on Lux (upstream package before the Lux renames, same image and engine
code, 2026-10-12, jobs 237537/237538/239568; raw data in `results/`), and this repo's own
`scripts/bench.sh` run (job 306934, see "Verified from this repo"). The throughput table is upstream only.

**Healthy 2-node start** (`summarize.sh`): `UP after 11-14 min`, `FABRIC OK: 44-49 cross-node channels, all
RCCL-ANP with GPUDirect RDMA`, `GPU KV cache size: 10,301,326 tokens`, correctness `10/10`, `SERVING at ...`.
Health check: TP all-reduce ~2,855 Gbps bus BW, PP handoff ~2,930 Gbps aggregate (~91% of line rate).

| check (2 nodes, 1M ctx) | result |
|---|---|
| MMLU-Pro 280-q subset | 0.864 (0.861, 0.882 in two more runs), 381 s; this repo 0.875, 377 s |
| needle, 32k-1,040k at depths 10/50/90% | 15/15 (both); 1,040k prompt: TTFT 64 s, prefill 16k tok/s, decode 17 tok/s |
| GSM8K | 0.968-0.976 |
| science suite, Inkling-MXFP4 (this repo) | **0.887** (GPQA 0.832, MaScQA 0.947, MMLU-Pro 0.882), 38 min at c=256 |
| science suite, Mistral-Large-2411 (1 node) | 0.662 (GPQA 0.531, MaScQA 0.779, MMLU-Pro 0.676); 0.656 / 0.661 on this image |

**Verified from this repo** (job 306934, 2026-10-13, `scripts/bench.sh` running science, mmlu and needle,
lux449-450): UP after 14 min; FABRIC OK, 45 channels, TP all-reduce 2,853 Gbps, PP handoff 2,919 Gbps; KV cache
10,297,506 tokens; correctness 10/10. Science suite 0.887: GPQA 149/179 = 0.832 (physics 0.954, chemistry 0.720),
MaScQA 612/646 = 0.947, MMLU-Pro 2999/3400 = 0.882; by field physics 0.911, chemistry 0.895, materials 0.947,
engineering 0.814; 0 errors. Truncated at 16,384 tokens: 33 GPQA, 10 MaScQA, 262 MMLU-Pro (scored from `reasoning`).
Needle 15/15 to 1,040k. `=== BENCH finished`; 65 min of job time.

**Verified through VISTA's submission path** (job 314500, 2026-10-13, `vllm-skill/inkling-serve` with
`script_args="BENCH=mmlu"`, VISTA's `#SBATCH` header on stdin, uploaded sources only, clean environment, lux430-431):
UP after 25 min (weight loading from a busy Lustre); FABRIC OK, 44 channels, TP all-reduce 2,854 Gbps, PP handoff
2,931 Gbps; correctness 10/10; MMLU-Pro subset 249/280 = 0.889; exit 0 in 32 min. The sources were not written.

**Throughput** (upstream gate), 8k-token prompts / 1k out, reasoning on (output tok/s):

| concurrency | 2 nodes tok/s/user | 2 nodes total | 2 nodes + `LUX_MTP=1` tok/s/user | 1 node (tuned) tok/s/user | 1 node total |
|---|---|---|---|---|---|
| 1 | 84.9 | 83 | **147.0** | 157.5 | 158 |
| 16 | 61.2 | 889 | 56.9 | 104.9 | 1,296 |
| 128 | 37.3 | 3,756 | 27.3 | 39.1 | 3,104 |
| 512 | 17.1 | **6,036** | 11.6 | 14.2 | 3,904 |

What to choose:
- **2 nodes** (default) for the 1M context with a 10.3M-token KV pool, and the most total tokens from 128 users up.
- **`LUX_MTP=1`** when one agent, or a few, need fast replies on long contexts: 1.7x at c=1, 2.2x at 119k-token prompts,
  3.5x decode at 1M. From c=4 up it is slower, so leave it off for shared servers and for the science suite.
- **1 node** for the best per-GPU efficiency (curve score 94.9 vs 54.5 per GPU), with a 100k context, or 1M with a 4.5M-token pool.

## Gotchas (each cost real time upstream)

- **Do not rename the `ZENITH_*` names inside the engine.** The image's patched vLLM reads them; use the `LUX_`
  aliases. `docker/patches/` is byte-identical to the validated set; keep it so (it is mounted over the image's copy).
- **`patches/anp/libionic.so.1` and `libionic-rdmav59.so` must be symlinks to one file**, so ANP and libibverbs share
  one loaded library. A skill import refuses symlinks and copies would load it twice, so the repo ships only
  `libionic.so.1.1.61`; `job.lux.slurm` copies the engine to `$VISTA_OUT/engine` and makes both links there. Run
  multi-node serving through the job (or `scripts/serve.sh`), not from the source tree.
- **The job never writes its sources.** Outputs go to `$VISTA_OUT` (`LUX_SERVE_OUT`) and the per-node container state
  to `$VISTA_OUT/state` (`ZENITH_STATE_DIR`, a Lux addition to the engine; default beside the scripts).
- **A knob must reach srun.** `serve.sh` exports `KEY=VAL` args, and `submit_lux.slurm.sh` exports every knob it knows.
  A variable set but not exported is silently ignored: the job runs with the default and the log looks normal.
- **Podman storage must be node-local** (`/mnt/bb/<jobid>`, set by `serve_lux.slurm.sh`); on NFS, podman hangs on a lock.
- **Cross-node RCCL must be RCCL-ANP + GDRDMA.** RCCL silently falls back to TCP sockets (1/12 the PP bandwidth). The
  engine pins `NCCL_NET=RCCL-ANP`, the health check fails below 1,500 Gbps PP, and `serve_lux` stops on any
  non-`RCCL-ANP/<n>/GDRDMA` channel. On `FABRIC CHECK FAILED` / `RCCL health check FAILED`, resubmit with
  `SBATCH_ARGS=--exclude=<nodes>`. `ZENITH_ALLOW_SOCKET_FALLBACK=1` is for debugging only.
- **Nodes with fewer than 8 visible GPUs** give TP7 and a mismatched PP group; exclude them (see
  `$PKG/tutorials/03_mistral_cpt/badnodes_snapshot.txt` for known bad Lux nodes).
- **A running job's time limit cannot be extended** by users on Lux. Ask for enough `-t` up front
  (`SBATCH_ARGS="-t 04:00:00"`). Inkling-bf16 reads 1.9 TB at start-up: allow up to an hour.
- **Inkling-bf16 on 1 node is infeasible** (`kv_cache_sizing.py` prints INFEASIBLE and exits). Use 2 nodes.
- **MTP on 2 nodes** also sets `ZENITH_PP_DEFERRED_RECV=0`: with the deferred PP recv on, vLLM's warmup hangs (an unmatched RCCL
  send; the watchdog kills it after 10 min). The two stages then take turns, hence the loss with many users.
- **Do not use DP2 x TP4 on one node** (the MoE runs in lockstep: much slower). `kv_cache_sizing.py` overestimates KV per token for
  Inkling; size from vLLM's own `GPU KV cache size` line.
- **Bench clients need Python >= 3.9** (dict `|`). Upstream ran them with the Lux hosts' `python3`; `science.sh` and
  `setup.sh` pick the newest of `python3.12..3.9` / `python3`, or `$PYTHON`.
- **Running the science suite against a shared server** at concurrency 256 slows everyone else on it. Use `bench.sh`
  (its own nodes) or `--concurrency 32`.
- **Reading logs:** `runs/<name>.<jobid>.out` has the milestones (`=== UP/FABRIC/SERVING/BENCH`); `server.log` is huge and
  full of RCCL INFO lines (logging is turned up on purpose so the fabric check can see the transports). Search for the
  strings in `summarize.sh`, not for "error".

## Muse Spark Open (parallel section)

**Lux HPC side** (Slurm + rootless podman), no Kubernetes. Job `vllm-skill/muse-serve`, engine
`hpc_jobs/muse-serve/lux-inference/`: AMD Zenith v0.1.9 renamed for Lux like the v8 one, with Muse's tuned defaults in
`lux_tuned_defaults.sh` as for Inkling. Muse-Spark-Open-816B-A42B is an 816B-total / 42B-active MoE, 1M context,
served on **1 node** with the lossless DSpark draft; more users take more 1-node replicas.

```text
submit_hpc_job(job="vllm-skill/muse-serve", cluster="lux")                                          # serve the shared MXFP4, 2 h
submit_hpc_job(job="vllm-skill/muse-serve", cluster="lux", script_args="BENCH=gsm8k")               # serve, score, exit
submit_hpc_job(job="vllm-skill/muse-serve", cluster="lux", script_args="MODEL=Muse-Spark-Open-816B-A42B MODELPATH=<dir>")  # bf16
```

| Key | Default | Meaning |
|---|---|---|
| `MODEL` | `Muse-Spark-Open-816B-A42B-Quark-MXFP4` | or `Muse-Spark-Open-816B-A42B` (bf16, 1.64 TB; gated, not shared: needs `MODELPATH`) |
| `SPEC` | `auto` | DSpark draft if it is beside the model; `dspark` requires it, `none` skips it (DFlash2 is refused) |
| `BENCH` | unset (serve) | comma-separated `gsm8k`, `science`, `mmlu` |
| `MODELPATH` | the shared models dir | your own `/lustre/orion/...` download |
| `GSM8K_N`, `SCIENCE_N`, `MAX_MODEL_LEN` | 250, all, 1,048,576 | as for inkling-serve |
| `BUILD_IMAGE` | 0 | 1: only if the shared `musemain2` image is unreadable; builds one (~1 h) into `$VISTA_JOB_DIR` |

The MXFP4 checkpoint and the image are shared, so the default needs no setup. **For bf16 or a private copy, walk the
user through [MUSE_SPARK.md](MUSE_SPARK.md) §3** (Meta's access form, the proxy, `hf auth login`, `scripts/muse.sh
download <dir> bf16`), then submit with `MODELPATH=<dir>`. Requests take `reasoning_effort` `low|medium|high|xhigh`
(default `xhigh`). Healthy: correctness 10/10, gsm8k ~0.976 (MXFP4) / ~0.984 (bf16). Limits and ROCm notes are in
MUSE_SPARK.md.

## Verify after any change

Check the VISTA format with `hpc-skill-author`'s validator on a copy of the repo in a folder named `vllm-skill`
(it requires the folder name to equal `name:`), then run `scripts/setup.sh`, then
`scripts/bench.sh 2 LUX_BENCH="science mmlu" LUX_SCIENCE_ARGS="--n 20"` and `scripts/summarize.sh <jobid>`. Expect FABRIC OK, correctness 10/10, MMLU-Pro subset ~0.86 (+-0.04), 0 errors in the
science smoke test, and `=== BENCH finished`.

## License and acknowledgments

Adapted from AMD's Zenith inference package for AMD Instinct GPUs (`zenith_inference_v8_lux`): the image, vLLM patches,
RCCL-ANP fabric fix, tuned kernels and benchmark clients are AMD's work. The skill's own code is Apache-2.0
(`LICENSE`). The AMD engine files, the models (Inkling Apache-2.0; Mistral-Large under the Mistral Research Licence)
and the benchmarks (GPQA CC BY 4.0, MaScQA CC BY-NC-SA 4.0, MMLU-Pro MIT) keep their own terms; see
`hpc_jobs/inkling-serve/lux-inference/README.md` § Licensing. Do not publish GPQA questions or per-question `science*.json` results.
