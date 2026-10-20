# inkling-serve

Serve an LLM with vLLM on OLCF Lux (8x MI355X per node, rootless podman) as an OpenAI-compatible endpoint, or
serve it, run benchmarks against it and exit. By default: Inkling-MXFP4 on 2 nodes (TP8 x PP2) with a
1,048,576-token context, reasoning and tool calls on. Part of the `vllm-skill` skill, which explains how to read
the results and use the endpoint. For Muse Spark Open use the job `vllm-skill/muse-serve`.

Supported clusters: Lux only (Slurm over SSH; the user is asked to log in through the hub once per session).

Default nodes: 2 (exclusive, 8 GPUs each). Use 1 for `MODEL=Mistral-Large-Instruct-2411`, or for Inkling-MXFP4's
single-node configuration (100k context, tuned for throughput per GPU).
Default time: 2:00:00. Start-up takes 11-14 min after the job starts (up to ~25 when Lustre is
busy). A serve job runs until its time limit (at
most 4:00:00 through VISTA), so Slurm reports it as TIMEOUT: that is the normal end of a serve job. A benchmark job
exits by itself: the full science suite takes ~40 min after start-up, `mmlu` ~6 min, `needle` ~5 min.

The model, the 57 GB image and the benchmark data are read in place from the shared package named in
`cluster_defaults.json` (`LUX_MODELPATH`, `LUX_IMAGE_TAR`, `LUX_DATASETSPATH`). Every submission first checks they
are readable, on the Lux login node, so a missing file fails the submission rather than the job.

## Script args

`script_args` is a space-separated list of `KEY=VALUE` pairs. All are optional, and any other key is refused:

- `MODEL=<name>`: `Inkling-MXFP4` (default, 1 or 2 nodes), `Mistral-Large-Instruct-2411` (1 node, 1M context),
  `Inkling-bf16` (2 nodes, 100k context; not in the shared package, so only with your own `LUX_MODELPATH`).
- `BENCH=<steps>`: benchmark and exit instead of serving. Comma-separated steps: `science` (GPQA Diamond + MaScQA +
  MMLU-Pro, 4,225 questions), `mmlu` (the 280-question MMLU-Pro subset), `needle` (15 long-context cells to
  1,040k tokens). Unset: serve until the time limit.
- `SCIENCE_N=<n>`: only the first n questions of each science benchmark (implies `BENCH=science` if `BENCH` is
  unset). `SCIENCE_N=20` is a 60-question smoke test.
- `MTP=0|1`: MTP speculative decoding (3 draft tokens). 1.7x faster for one user, slower from 4 users up.
- `MAX_MODEL_LEN=<tokens>`: context length (default 1,048,576 on 2 nodes, 100,000 on 1 node).

Examples: `script_args="BENCH=science"` (full suite, 2 nodes), `script_args="BENCH=science,mmlu SCIENCE_N=20"`,
`node_count=1, script_args="MODEL=Mistral-Large-Instruct-2411 BENCH=science"`, `script_args="MTP=1"` (serve).

## Outputs

In the job's output directory, refreshed every minute while the job runs:

- `summary.txt`: start-up milestones (`=== UP`, `=== FABRIC OK`, `=== SERVING at http://<head>:8000/v1`), the KV
  cache size, the 10-question correctness probe, network bandwidth, any benchmark scores, and failure signatures.
- `endpoint.txt`: the OpenAI base URL, once serving. It is reachable only from inside Lux (login and compute nodes).
- `job.log`: the job's own log.
- `science.txt`, `mmlu_pro.txt`, `long_context.txt`: benchmark reports (scores only).
- `science.json`: every science question and reply. It holds GPQA questions, which must stay private: do not fetch
  it into a shared place or quote it.
- `server.log`: vLLM's log. Very large and full of RCCL INFO lines on purpose; read `summary.txt` instead.
- `correctness.txt`, `net_transports.txt`, `state/`: probe output, cross-node transports, per-node container logs.

A healthy 2-node Inkling-MXFP4 start: `UP after 11-14 min`, `FABRIC OK` with all channels RCCL-ANP with GPUDirect
RDMA, KV cache ~10.3M tokens, correctness 10/10, then `SERVING`. The full science suite scores ~0.887 (+-0.027).
