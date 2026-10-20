# Inkling-MXFP4 on Lux (MI355X): the hackathon tutorial

The tutorial serves **Inkling-MXFP4 on two Lux nodes (16 x MI355X, TP8 x PP2) with the model's full
1,048,576-token context and reasoning on**, as an OpenAI-compatible endpoint for agents. It is what
`docker/submit_lux.slurm.sh` does with no settings. `docker/serve_lux.slurm.sh` wraps it with the Lux site
setup and checks the fabric and correctness before it announces the endpoint.

`--nodes=1` gives the second configuration this package carries: one node, tuned for total token rate, with a
100k context ([section 7](#7-one-node-tuned-for-throughput)), or the full 1M with `ZENITH_MAX_MODEL_LEN=1048576`
([section 7.1](#71-1m-context-on-one-node)).

A second model, **Mistral-Large-Instruct-2411 on one node at 1M tokens**, and a physics / chemistry /
materials-science benchmark suite to score it (or a fine-tune of it) are in
[section 8](#8-second-model-mistral-large-2411-and-the-science-suite) and
[LUX_MISTRAL_LARGE.md](LUX_MISTRAL_LARGE.md).

## 1. Serve it

```bash
cd <this package>
ZENITH_MODELPATH=<dir holding Inkling-MXFP4/> \
ZENITH_IMAGE_TAR=<path to inference_image_gfx950_v8mi250_20261011.tar> \
  sbatch -A <account> -p batch docker/serve_lux.slurm.sh
```

The image is `inference_image_gfx950_v8mi250_20261011.tar` (sha256 `e03d16939f5e1b4b3cc7aa4e783246f666ca1350b3c5687faa3d6c9ce8f40713`),
built from this package's `docker/` with every fix-up baked in (README §3, §8). The results below were measured
on it on 2026-10-12, in the shipping gate: fresh copies of this tree, no `ZENITH_*` knobs set beyond the model
and image paths. Earlier numbers on the previous image (`inference_image_gfx950.tar`, zenith_v6) are noted where
they differ.

The job log (`inkling.serve.<jobid>.out`) reaches these lines in about 11-13 minutes from a cold start:

```
=== UP at http://lux326:8000/v1 after 12 min
     15 via NET/RCCL-ANP/0/GDRDMA
     ...
=== FABRIC OK: 45 cross-node channels, all RCCL-ANP with GPUDirect RDMA
GPU KV cache size: 10,301,326 tokens, Maximum concurrency for 1,048,576 tokens per request: 9.82x
  overall: 10/10 = 100.0%
=== SERVING at http://lux326:8000/v1 (model Inkling-MXFP4; ...)
```

The server runs until the job's time limit (8 h in the script) or `scancel`. The full server log is in
`docker/serve.<jobid>/server.log`. `serve_lux.slurm.sh` sets per-job podman storage on the node's burst buffer
(`/mnt/bb/<jobid>`); without that, podman hangs on an NFS lock on Lux.

To serve with MTP speculative decoding on, prefix the same command with `ZENITH_MTP=1`. It makes a single user 1.7-3.6x
faster (more at long contexts) but the server as a whole slower from 4 concurrent users up; see
[section 6.1](#61-mtp-on-zenith_mtp1) before choosing it.

## 2. Use it

Any OpenAI-compatible client works. Point it at `http://<head node>:8000/v1` with model `Inkling-MXFP4`.

```bash
curl -s http://<head>:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Inkling-MXFP4",
  "messages": [{"role": "user", "content": "What is 17*23?"}],
  "max_tokens": 2000}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://<head>:8000/v1", api_key="unused")
r = client.chat.completions.create(model="Inkling-MXFP4",
                                   messages=[{"role": "user", "content": "What is 17*23?"}],
                                   max_tokens=2000)
print(r.choices[0].message.reasoning)   # the model's thinking
print(r.choices[0].message.content)     # the answer
```

* **Reasoning is on.** The chat template thinks before answering at a default effort of 0.9. The server's
  `inkling` reasoning parser returns the thinking in each message's `reasoning` field (streamed as
  `delta.reasoning`) and only the answer in `content`. Pass
  `"chat_template_kwargs": {"reasoning_effort": <0-0.99>}` to change the effort per request. Leave enough
  `max_tokens` for the thinking: on MMLU-Pro questions it averaged ~2,300 tokens, and a reply cut off while
  still thinking has no `content` at all.
* **Tool calls work** with `"tools": [...]` and `"tool_choice": "auto"`; they come back in `tool_calls` with
  `finish_reason: "tool_calls"` (the `inkling` tool-call parser).
* **Context** is 1,048,576 tokens, prompt plus output. vLLM sizes the KV pool at 10.3M tokens, so about 10
  full-length requests, or many more shorter agent sessions, fit at once.

`ZENITH_REASONING_PARSER=none` / `ZENITH_TOOL_CALL_PARSER=none` turn the parsers off. The thinking then comes
back inline in `content`.

## 3. What the defaults are

| Setting | 2 nodes (default) | 1 node (`--nodes=1`) | Knob |
|---|---|---|---|
| parallelism | TP8 x PP2 (16 GPUs) | TP8 | `ZENITH_TP`, `sbatch --nodes` |
| context | 1,048,576 | 100,000 (1,048,576 works too, [section 7.1](#71-1m-context-on-one-node)) | `ZENITH_MAX_MODEL_LEN` |
| KV cache per GPU | 200 GB (utilization 0.84) | 128 GB | `ZENITH_TARGET_KV_GB` |
| MTP speculative decoding | off (`ZENITH_MTP=1`: on, 3 draft tokens) | on, 3 draft tokens | `ZENITH_MTP`, `ZENITH_MTP_TOKENS` |
| scheduler token budget | 32768 | 32768 | `ZENITH_MAX_NUM_BATCHED_TOKENS` |
| largest CUDA-graph batch | vLLM's (512) | 2048 | `ZENITH_MAX_CUDAGRAPH_CAPTURE_SIZE` |
| reasoning / tool-call parser | inkling / inkling | inkling / inkling | `ZENITH_REASONING_PARSER`, `ZENITH_TOOL_CALL_PARSER` |
| cross-node RCCL | RCCL-ANP plugin, GPUDirect RDMA, pinned | (XGMI only) | `docker/patches/anp` |
| tuned kernels | on (gfx950) | on (gfx950) | `docker/patches/hl_overlay` |

The values live in `docker/lux_tuned_defaults.sh`. They apply only to Inkling-MXFP4 on 1 or 2 nodes and
only fill in what is unset: set any knob to override it, or `ZENITH_TUNED_DEFAULTS=0` for the generic defaults.
The launcher prints `lux: tuned defaults for Inkling-MXFP4 on 2 nodes (TP8 x PP2): ...` with the values in
force. MTP is off on two nodes because it lowers total throughput there from 4 concurrent users up.
`ZENITH_MTP=1` turns it on and fills in the rest: `ZENITH_MTP_PP=1` (the launcher otherwise refuses MTP with
pipeline parallelism), 3 draft tokens, CUDA graphs to 2048 and `ZENITH_PP_DEFERRED_RECV=0`
([section 6.1](#61-mtp-on-zenith_mtp1)).

## 4. The cross-node fabric: RCCL-ANP with GPUDirect RDMA, no silent fallback

Lux's nodes talk over 8 x Pollara 400G RoCE NICs per node. RCCL reaches them only through AMD's RCCL-ANP net
plugin and the vendor `libionic` verbs provider. Neither ships with the hosts or the image, so the package carries
a copy built on Lux in `docker/patches/anp/` (with its licences and rebuild scripts; see its `README.md`).
Without it RCCL quietly uses TCP sockets, which carry the pipeline handoff at about 1/12 of the speed.

The package makes that fallback fail loudly instead:

* `run_container.sh` sets `NCCL_NET=RCCL-ANP` on multi-node runs, so RCCL errors at communicator creation if
  the plugin cannot load, rather than picking sockets.
* The launcher's RCCL health check measures the TP all-reduce and the PP stage handoff before vLLM starts. Below
  `ZENITH_HEALTH_MIN_PP_GBPS` (1500 Gbps with GPUDirect RDMA) it fails, and nothing is served.
* `serve_lux.slurm.sh` reads vLLM's own RCCL INFO lines once it is up and stops the service unless every
  cross-node channel is `via NET/RCCL-ANP/<n>/GDRDMA`.

`ZENITH_ALLOW_SOCKET_FALLBACK=1` turns the first two off, for debugging only.

Measured on the tutorial configuration, four separate allocations (2026-10-10 on the previous image, 2026-10-12
on this one):

| | job 176839 | job 177168 | job 237537 | job 237538 |
|---|---|---|---|---|
| TP all-reduce, 1 GiB, bus bandwidth | 2855-2865 Gbps | 2852-2860 Gbps | 2853-2857 Gbps | 2851-2854 Gbps |
| PP stage 0 -> 1 handoff, aggregate over 8 lanes | 2943 Gbps | 2922 Gbps | 2923 Gbps | 2932 Gbps |
| PP handoff, slowest GPU | 364 Gbps | 362 Gbps | 363 Gbps | 362 Gbps |
| vLLM cross-node channels | 49, all `RCCL-ANP/<n>/GDRDMA` | 45, all `RCCL-ANP/<n>/GDRDMA` | 44, all `RCCL-ANP/<n>/GDRDMA` | 46, all `RCCL-ANP/<n>/GDRDMA` |

Each GPU's handoff runs at ~91% of its NIC's 400 Gbps line rate. Over TCP sockets the same handoff measured
~237 Gbps aggregate.

## 5. Correctness

**MMLU-Pro subset.** `bench/inkling/mmlu_pro_quick.py` runs 280 questions, 20 from each of MMLU-Pro's 14
categories (TIGER-Lab/MMLU-Pro test split, MIT licence, fixed seed), zero-shot chain of thought at temperature 0
with reasoning on. It takes about 7 minutes on two nodes:

```bash
python3 bench/inkling/mmlu_pro_quick.py --host <head> --port 8000 --out mmlu_pro.json
```

| | accuracy (95% CI) | wall time | mean completion tokens |
|---|---|---|---|
| 2 nodes, TP8 x PP2 (tutorial) | **0.864** (0.819-0.900) | 381 s | 2,190 |
| 2 nodes, TP8 x PP2, MTP-3 (`ZENITH_MTP=1`) | 0.868 (0.823-0.903) | 261 s | 2,121 |
| 1 node, TP8, MTP-3 | 0.886 (0.843-0.918) | 183 s | 2,165 |

280 questions resolve differences of about 5 points, so the three agree; the subset is for catching a broken
deployment, not for a leaderboard number. Two more 2-node runs scored 0.861 and 0.882; on the previous image the three
configurations scored 0.875, 0.889 and 0.896. Per category, the 2-node run scored 0.60 (law) to 1.00 (math).
Truncated replies (7 of 280 on two nodes, at the default 16,384 `max_tokens`) are scored from their reasoning.

**Long-context retrieval.** `bench/inkling/long_context_check.py` hides one sentence (an access code) in a
haystack of public-domain prose at 10%, 50% and 90% depth and asks for it, at prompt lengths up to 1,040,000
tokens, one request at a time:

```bash
python3 bench/inkling/long_context_check.py --host <head> --port 8000 --out long_context.json
```

On the tutorial configuration (job 237537, and again in jobs 237538 and 239568), all 15 cells passed, up to a 1,039,797-token prompt. Timings are
for one request alone on the server; decode includes the reasoning tokens:

| prompt tokens | needle found (depth 10 / 50 / 90%) | time to first token | prefill | decode |
|---|---|---|---|---|
| 32k | 3 / 3 | 0.9 s (2.1 s for the first request) | 35k tok/s | 78 tok/s |
| 128k | 3 / 3 | 3.1 s | 41k tok/s | 58 tok/s |
| 256k | 3 / 3 | 7.2 s | 36k tok/s | 42-44 tok/s |
| 512k | 3 / 3 | 19.9 s | 26k tok/s | 28-29 tok/s |
| 1,040k | 3 / 3 | 64 s | 16k tok/s | 16.6-16.8 tok/s |

The whole run takes about 6 minutes. With `ZENITH_MTP=1` all 15 cells also pass (job 237539), and decode at
every length is 2.5-3.6x faster (59-62 tok/s at 1,040k), for a 5-15% longer time to first token (74 s at 1,040k).

`bench/inkling/correctness_probe.py` (10 prompts with checkable answers) runs in every serve job.

## 6. Baseline performance (2 nodes, not tuned)

These are baselines of the tutorial configuration; nothing was tuned for throughput on two nodes.

8k-token prompts, 1k-token outputs (every request ran to 1,024 tokens), natural prose through the chat API at
temperature 0.7 with reasoning on, 16+ requests per concurrency level, zero failures (job 237537). Output tokens/s
per GPU (16 GPUs) and per user:

| c | tok/s/GPU | tok/s/user | TTFT mean | total output tok/s |
|---|---|---|---|---|
| 1 | 5.2 | 84.9 | 0.23 s | 83 |
| 2 | 9.7 | 79.4 | 0.32 s | 155 |
| 4 | 17.6 | 73.1 | 0.52 s | 282 |
| 8 | 30.6 | 65.5 | 1.1 s | 490 |
| 16 | 55.6 | 61.2 | 1.7 s | 889 |
| 32 | 94.7 | 53.9 | 2.6 s | 1,515 |
| 64 | 155.3 | 46.3 | 4.1 s | 2,485 |
| 128 | 234.8 | 37.3 | 6.9 s | 3,756 |
| 256 | 308.1 | 26.0 | 12.5 s | 4,929 |
| 512 | 377.2 | 17.1 | 23.5 s | 6,036 |
| 1024 | 375.2 | 8.4 | 46.3 s | 6,003 |

The curve score (geomean over c of sqrt(tok/s/user x tok/s/GPU)) is 54.5 per GPU (54.5 again in job 237538,
54.7 in job 239568, run from this package's own tarball; 51.5 on the previous image), against 94.9 for the tuned single node on 8 GPUs. The pipeline stage, MTP being off and the untuned 2-node settings all cost per-GPU
efficiency; in total the two nodes still produce more tokens than one from c=128 up (6,036 vs 3,903 output tok/s
at c=512). Use `--nodes=1` when throughput matters. It serves the full 1M context too, at about half the 2-node
prefill rate on 1M-token prompts ([section 7.1](#71-1m-context-on-one-node)).

Long prompts, ~119k tokens in / 1k out, same client:

| c | tok/s/GPU | tok/s/user | TTFT mean | total (prompt + output) tok/s |
|---|---|---|---|---|
| 1 | 3.1 | 57.0 | 2.9 s | 5,720 |
| 2 | 5.3 | 52.6 | 4.7 s | 9,873 |
| 4 | 8.5 | 45.5 | 7.4 s | 15,804 |
| 8 | 12.7 | 37.9 | 12.4 s | 23,596 |

Raw results: `results/inkling/lux_gate_v8mi250_2026-10-12/`; the previous image's in
`results/inkling/lux_2node_1m_2026-10-10/`.

### 6.1 MTP on (`ZENITH_MTP=1`)

```bash
ZENITH_MTP=1 ZENITH_MODELPATH=... ZENITH_IMAGE_TAR=... sbatch -A <account> -p batch docker/serve_lux.slurm.sh
```

MTP (multi-token prediction) drafts 3 tokens per step with the model's own MTP head, and the main model verifies
them in the same forward. Accuracy holds: MMLU-Pro 0.868 and the long-context check 15/15
([section 5](#5-correctness)). The model accepts 2.7-2.75 tokens per step at low load and ~2.15 at c=256 and
above.

On two nodes it has a cost. The deferred PP recv in `docker/patches/pp_broadcast_deferred_recv.py` lets the
two pipeline stages work on different requests at the same time. With MTP, on Lux, it hangs vLLM's warmup (an
RCCL send between the stages is never matched; the watchdog kills the server after 10 minutes). So
`ZENITH_MTP=1` also sets `ZENITH_PP_DEFERRED_RECV=0`, which undoes that patch at container start. The two stages
then mostly take turns. One user does not notice, because one request only ever occupies one stage at a time.
Many users do.

Same benchmark as above (job 237539; MTP off from the table above for comparison):

| c | tok/s/user, MTP | MTP off | tok/s/GPU, MTP | MTP off | TTFT mean, MTP |
|---|---|---|---|---|---|
| 1 | **147.0** | 84.9 | **8.9** | 5.2 | 0.24 s |
| 2 | **99.1** | 79.4 | **10.8** | 9.7 | 0.45 s |
| 4 | 68.7 | 73.1 | 15.7 | 17.6 | 0.81 s |
| 8 | 62.0 | 65.5 | 27.0 | 30.6 | 1.5 s |
| 16 | 56.9 | 61.2 | 45.6 | 55.6 | 2.7 s |
| 32 | 49.0 | 53.9 | 74.9 | 94.7 | 4.6 s |
| 64 | 36.7 | 46.3 | 108.1 | 155.3 | 7.7 s |
| 128 | 27.3 | 37.3 | 148.8 | 234.8 | 13.6 s |
| 256 | 18.6 | 26.0 | 188.6 | 308.1 | 25.4 s |
| 512 | 11.6 | 17.1 | 217.2 | 377.2 | 49.3 s |
| 1024 | 6.8 | 8.4 | 237.3 | 375.2 | 98.8 s |

Curve score 46.8, against 54.5 with MTP off (43.4 vs 51.5 on the previous image). With ~119k-token prompts it is
124.6 vs 57.0 tok/s/user at c=1, 111.5 vs 52.6 at c=2, and about even at c=4-8. Decode at a 1M-token prompt
goes from ~17 to 59-62 tok/s.

Use it when one agent, or a few, work on long contexts and each reply's latency matters. Leave it off for a
server shared by many users at once. Raw results: `results/inkling/lux_gate_v8mi250_2026-10-12/` (the previous
image's in `results/inkling/lux_2node_1m_mtp3_2026-10-11/`).

## 7. One node, tuned for throughput

`sbatch --nodes=1 docker/serve_lux.slurm.sh` (or `submit_lux.slurm.sh`) serves **Inkling-MXFP4 on one node,
TP8 over all 8 GPUs**, with the serving defaults and kernels tuned for that configuration in October 2026. One
TP8 instance on the whole node gives the highest total token rate a single job gets on this model. The context
is 100,000 tokens by default; `ZENITH_MAX_MODEL_LEN=1048576` gives the full 1M
([section 7.1](#71-1m-context-on-one-node)).

The kernels are two files in `docker/patches/hl_overlay/vllm/`, copied over the image's vLLM at container start
by `patches/zz_hl_overlay.py`:

* `distributed/device_communicators/cuda_communicator.py`: every TP all-gather of fp32/fp16/bf16 goes
  through AITER's IPC all-gather. RCCL in a replayed graph has a floor of ~170 us on half of the GPUs where AITER
  takes 4-11 us. `ZENITH_AITER_AG_MIN_SHARD_BYTES=<n>` keeps shards up to n bytes on RCCL.
* `models/inkling/amd/ops/fa4_rel_attention.py`: the guarded RAHS relative-position attention kernel for
  gfx950.

The overlay only applies on gfx950 and refuses to serve if the image's vLLM differs from the ones it was made
against (the zenith_v6 image and the v8 gfx950 image, listed in `hl_overlay/BASE.sha256`), so a run never serves
the untuned kernels by accident. On gfx942 / gfx90a it is skipped. It applies on
two nodes as well.

8k-token prompts, 1k-token outputs, natural prose through the chat API at temperature 0.7, 16+ requests per
concurrency level, one node (job 237540). Output tokens/s per GPU and per user:

| c | tok/s/GPU | tok/s/user |
|---|---|---|
| 1 | 19 | 157.5 |
| 2 | 33 | 144.4 |
| 4 | 59 | 137.9 |
| 8 | 100 | 122.8 |
| 16 | 162 | 104.9 |
| 32 | 230 | 78.9 |
| 64 | 316 | 59.2 |
| 128 | 388 | 39.1 |
| 256 | 446 | 24.0 |
| 512 | 488 | 14.2 |
| 1024 | 471 | 7.2 |

The curve score (geomean over c of sqrt(tok/s/user x tok/s/GPU)) is 94.9. The same job on the previous image,
started at the same time on another node, scored 89.5 (89.2 when the kernels were tuned). The tuning started from
70.1, with TP8, KV 128 and MTP-3 already set and everything else vLLM's defaults: +27% on the previous image. Every run passed the
correctness gate against that baseline. (These were measured before the reasoning and tool-call parsers were
turned on by default; the parsers only change how the reply is split, not what is generated.)

### 7.1 1M context on one node

```bash
ZENITH_MAX_MODEL_LEN=1048576 ZENITH_MODELPATH=... ZENITH_IMAGE_TAR=... \
  sbatch -A <account> -p batch --nodes=1 docker/serve_lux.slurm.sh
```

The other single-node defaults stay as they are (MTP-3, KV 128). Tested from this package's tarball on this image
(job 255825, 2026-10-12). Serving 11 min after launch, at utilization 0.7649 with a KV pool of **4,503,472
tokens**, 4.29 requests of 1,048,576 tokens (two nodes: 10.3M, 9.82). Correctness 10/10, GSM8K 0.968, MMLU-Pro
subset 246/280 = 0.879 (183 s at 100k context, 179 s here), and the needle was found at all 15 places:

| prompt | found | time to first token | prefill | decode |
|---|---|---|---|---|
| 32k | 3 / 3 | 0.95 s (1.7 s for the first request) | 33k tok/s | 182-211 tok/s |
| 128k | 3 / 3 | 5.0 s | 26k tok/s | 162-171 tok/s |
| 256k | 3 / 3 | 13.0 s | 20k tok/s | 131-149 tok/s |
| 512k | 3 / 3 | 38.2 s | 13k tok/s | 95-107 tok/s |
| 1,040k | 3 / 3 | 126 s | 8.2k tok/s | 58-64 tok/s |

Against the 2-node tutorial (section 5), one node takes twice as long to the first token of a 1M prompt (126 vs
64 s) and then decodes it 3.5x faster (MTP-3, ~60 vs ~17 tok/s). With 8k-token prompts (same client as above,
three points) it matches the 100k-context node at c=1 (157.4 vs 157.5 tok/s/user) and c=1024 (3,649 vs 3,764
output tok/s); at c=64 it gave 2,013 vs 2,529 output tok/s, TTFT 13.6 vs 7.0 s (one run each). With ~119k-token
prompts: 133.0 tok/s/user at c=1 and 103.1 at c=2, against 57.0 and 52.6 on two nodes. Choose it for one or a few
agents on long contexts on a single node; keep 100k for the highest token rate on short prompts. Raw results:
`results/inkling/lux_gate_v8mi250_2026-10-12/1node_1m_255825/`.

## 8. Second model: Mistral Large 2411, and the science suite

```bash
ZENITH_MODEL=Mistral-Large-Instruct-2411 ZENITH_MODELPATH=<dir holding it> ZENITH_IMAGE_TAR=<image tar> \
  sbatch -A <account> --nodes=1 -J mistral.serve -o mistral.serve.%j.out docker/serve_lux.slurm.sh
```

serves the dense 123B Mistral-Large-Instruct-2411 on one node (TP8) with a 1,048,576-token context, eight times
its trained 131,072, through YaRN. One node is enough: its KV pool holds about three full-length requests, so
more users means more one-node copies, not more nodes per copy. `bench/science/` scores any endpoint on GPQA
Diamond (physics and chemistry), MaScQA (materials science) and MMLU-Pro (physics, chemistry, engineering);
`bench/mistral/mistral_bench.slurm.sh` serves the model, or a fine-tune of it, and runs the suite in one job.
How it was tuned, the YaRN setting that holds up at 1M, what was rejected and the baseline scores are in
[LUX_MISTRAL_LARGE.md](LUX_MISTRAL_LARGE.md).

On this image (job 237542, 2026-10-12) it scored 10/10 on the correctness probe and 0.968 on GSM8K, the tool
call worked, and the needle was found at all 15 places from 32k to 1,040k tokens (310 s to the first token at
1,040k, 303 s on the previous image). The science suite scored 0.656 and, in a second run, 0.661, against 0.661
and 0.646 for the previous image: the same within the suite's run-to-run spread.

## Caveats

* Use `ZENITH_TP=4 ZENITH_USE_GPUS=4` on 4 GPUs for per-GPU efficiency rather than a full node's total
  rate. It scored 91.1 on 4 GPUs, but two TP4 instances on one node fall 3-4% below one TP8 instance at 2-16
  users. They are only ahead from 32 users up.
* DP2 x TP4 on one node runs the MoE in lockstep across the two replicas and is much slower. Don't use it.
* The all-gather change is in vLLM's generic communicator, so on gfx950 it applies to every model served
  from this package, not just Inkling. It is dtype-gated (fp8 and integer all-gathers stay on RCCL).
* `kv_cache_sizing.py` estimates far more KV per token for this model than vLLM allocates (vLLM's pool at KV 200
  is 10.3M tokens on two nodes, at KV 128 on one node 4.5M). Size the KV pool from vLLM's `GPU KV cache size` line.
* `ZENITH_MODEL=Inkling-bf16` on two nodes serves the bf16 checkpoint (1.9 TB) with the generic defaults (100k
  context, KV 64, MTP off). From this package on this image (job 255824, 2026-10-12): 10/10, GSM8K 0.980, MMLU-Pro
  subset 0.861, the needle found at 32k and 90k, every cross-node channel on `RCCL-ANP/<n>/GDRDMA`; 8k-token
  prompts at 71-81% of the MXFP4 tutorial's rate (64.8 tok/s/user at c=1, 4,885 output tok/s at c=1024).
  Its cold start reads 1.9 TB; budget up to an hour (README §1).
