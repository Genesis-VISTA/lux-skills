# muse-serve

Serve Muse Spark Open with vLLM on one OLCF Lux node (8x MI355X, rootless podman) as an OpenAI-compatible endpoint,
or serve it, run benchmarks against it and exit. It is an 816B-total, 42B-active MoE with a 1,048,576-token context;
reasoning and tool calls are on (the `muse` parsers). The default is the shared MXFP4 checkpoint with the DSpark speculative
draft, zenith v0.1.9's validated recipe (score 71.96, probe 20/20, gsm8k 0.976). Part of the `vllm-skill` skill,
which explains how to read the results and use the endpoint. Inkling and Mistral are served by `vllm-skill/inkling-serve`.

Supported clusters: Lux only (Slurm over SSH; the user is asked to log in through the hub once per session).

Nodes: 1 (exclusive, 8 GPUs). Muse serves one endpoint per node; more users take more 1-node replicas.
Default time: 2:00:00. Start-up takes about 10-15 min after the job starts: 12 min in the Lux run (job 361678,
including loading the 58 GB image). In zenith's 1-node MXFP4 run, vLLM went from launch to serving in ~8.5 min, 4 of
them reading weights. A serve job runs until its time limit (at most
4:00:00 through VISTA), so Slurm reports TIMEOUT: that is the normal end of a serve job.

The checkpoint and the image are read in place from the shared Genesis-Hackathon package (`cluster_defaults.json`):
`LUX_MODELPATH=/lustre/orion/world-shared/stf218/junqi/Genesis-Hackathon/models`, which holds
`Muse-Spark-Open-816B-A42B-Quark-MXFP4/`, and `LUX_IMAGE_TAR=.../containers/inference_image_gfx950_musemain2.tar`.
Every submission first checks them on the Lux login node, so the default MXFP4 serve needs no setup. The DSpark draft
is used when `Muse-Spark-Open-DSpark/` is verified in the same directory (`SPEC=auto`); otherwise the job serves
without it, which is slower, and says so in `summary.txt`.

**bf16, or your own copy:** the weights go into your own directory first, once, on a login node. The skill's
`MUSE_SPARK.md` §3 walks through it: Meta's access form for bf16, the OLCF proxy, `hf auth login`, then
`scripts/muse.sh download <dir> bf16`. Then pass `MODELPATH=<dir>`.

## Script args

`script_args` is a space-separated list of `KEY=VALUE` pairs. All are optional, and any other key is refused:

- `MODEL=<name>`:
  - `Muse-Spark-Open-816B-A42B-Quark-MXFP4` (default, or its `-r63` mirror): EP8 MoE, KV 180 GB, graphs to 1024.
  - `Muse-Spark-Open-816B-A42B` (bf16, 1.64 TB): TP MoE with tuned bf16 tables, KV 60 GB, 512 sequences, score 56.87.
- `SPEC=auto|dspark|none`:
  - `auto` (default) uses the DSpark draft if it is in the model directory;
  - `dspark` requires the draft;
  - `none` never uses it.

  DSpark is lossless and faster at every concurrency up to 256. The DFlash2 drafts are refused: they corrupt Muse's
  output.
- `BENCH=<steps>`: benchmark and exit instead of serving. Comma-separated steps:
  - `gsm8k`: 250 questions, zenith's release check.
  - `science`: GPQA Diamond + MaScQA + MMLU-Pro.
  - `mmlu`: the 280-question MMLU-Pro subset.

  Unset: serve until the time limit.
- `GSM8K_N=<n>`, `SCIENCE_N=<n>`: fewer questions; each implies its step if `BENCH` is unset.
- `MAX_MODEL_LEN=<tokens>`: context length (default 1,048,576).
- `MODELPATH=/lustre/orion/...`: a models directory of your own (for bf16; see above).
- `BUILD_IMAGE=1`: only if the shared image is unreadable. Builds one on the node into `$VISTA_JOB_DIR/images/` (~1 h;
  ask for 4:00:00), then serves.

Examples:
- `script_args=""`: serve MXFP4.
- `script_args="BENCH=gsm8k"`: serve, score gsm8k (250), exit.
- `script_args="MODEL=Muse-Spark-Open-816B-A42B MODELPATH=/lustre/orion/<proj>/scratch/<user>/hfmodels BENCH=gsm8k,science"`.

## Outputs

In the job's output directory, refreshed every minute while the job runs:

- `summary.txt`:
  - the milestones (`=== BUILD`, `=== UP`, `=== SERVING at http://<head>:8000/v1`);
  - the KV cache size and the 10-question correctness probe;
  - any benchmark scores and failure signatures.
- `endpoint.txt`: the OpenAI base URL, once serving. It is reachable only from inside Lux (login and compute nodes).
  The model name is the checkpoint's directory name (for example `Muse-Spark-Open-816B-A42B-Quark-MXFP4`). Set
  `reasoning_effort` per request to `low`, `medium`, `high` or `xhigh` (the default; any other value is an error).
  The model card's sampling defaults are temperature 1.0 and top_p 1.0.
- `job.log`: the job's own log. `build.log`: the image build, if `BUILD_IMAGE=1` ran one.
- `gsm8k.txt`, `science.txt`, `mmlu_pro.txt`: benchmark reports.
- `science.json`: every science question and reply. It holds GPQA questions, which must stay private: do not fetch
  it into a shared place or quote it.
- `server.log`: vLLM's log (`lux: tuned defaults for Muse-...` gives the settings used); read `summary.txt` instead.
- `correctness.txt`, `state/`: probe output, the container logs.

A healthy MXFP4 start: `UP`, correctness 10/10, then `SERVING`; gsm8k ~0.976 (zenith job 360800; 0.972 on
Lux, job 361678). Expect a GPU KV cache of about 12.5M tokens, one user at ~185 tok/s on 8k prompts (~280 on chat
prompts) and ~2,850 total tok/s/GPU at 256 users (`../../results/muse_lux_mxfp4_1node_2026-10-20/`). On a HIP
out-of-memory error in the DSpark speculator followed by HTTP 500s (seen once with bf16 at c = 128), resubmit.
