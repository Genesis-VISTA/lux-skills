# Mistral Large 2411 on Lux (MI355X) at 1M tokens, and a science benchmark

The second model of the hackathon: **Mistral-Large-Instruct-2411** (dense, 123B parameters, bf16) served on
**one Lux node (TP8 over 8 x MI355X) with a 1,048,576-token context**, as an OpenAI-compatible endpoint. It was
tuned in an afternoon (October 2026) with serving options only (parallelism, attention backend, speculative
decoding, scheduler limits, KV cache size and dtype), no kernel work. `bench/science/` scores it, or a fine-tune
of it, on physics, chemistry and materials-science questions ([section 7](#7-science-baseline)).

## 1. Serve it

```bash
cd <this package>
ZENITH_MODEL=Mistral-Large-Instruct-2411 \
ZENITH_MODELPATH=<dir holding Mistral-Large-Instruct-2411/> \
ZENITH_IMAGE_TAR=<path to inference_image_gfx950_v8mi250_20261011.tar> \
  sbatch -A <account> -p batch --nodes=1 -J mistral.serve -o mistral.serve.%j.out docker/serve_lux.slurm.sh
```

`serve_lux.slurm.sh` is the tutorial's serve job ([LUX_HACKATHON.md](LUX_HACKATHON.md)); with this model and one
node, `docker/lux_tuned_defaults.sh` fills in the settings below. The job log reaches these lines in about
7 minutes from a cold start (weights already on Lustre):

```
lux: tuned defaults for Mistral-Large-Instruct-2411 on 1 node: MAX_MODEL_LEN=1048576 KV_GB=200 ...
=== UP at http://lux001:8000/v1 after 7 min
GPU KV cache size: 3,341,520 tokens, Maximum concurrency for 1,048,576 tokens per request: 3.19x
  overall: 10/10 = 100.0%
=== SERVING at http://lux001:8000/v1 (model Mistral-Large-Instruct-2411; ...)
```

The checkpoint is the original `mistralai/Mistral-Large-Instruct-2411` from Hugging Face (Mistral Research
Licence: research and non-commercial use). `docker/download_models.sh mistral` fetches it (224 GB, after you
accept the licence on its Hugging Face page) without the duplicate `consolidated*` weights. The server reads the
Hugging Face weights and config, plus `params.json` and `tokenizer.model.v7` for Mistral's own tokenizer.

The correctness probe (ten short raw completions) scores 10/10. The YaRN settings that stretch the model to 1M
were chosen so that they do not cost accuracy on ordinary prompts ([section 4](#4-the-1m-context-yarn-and-the-rope_theta-trap)).
The final check of this package with these defaults (job 224645, `results/mistral/lux_1node_1m_2026-10-11/final_1m/`)
scored 10/10, GSM8K 0.976 and science 0.661, and found the needle at all 15 places tested, from 32k to 1,040k
tokens at depths 0.1, 0.5 and 0.9 (the 1,040k prompts take 303 s to the first token). That was on the previous
image (zenith_v6). On this package's image, `inference_image_gfx950_v8mi250_20261011.tar`, the same check (job
237542, the shipping gate, 2026-10-12) scored 10/10, GSM8K 0.968 and science 0.656, with all 15 needles found
(310 s at 1,040k).
For prompts that fit in 131,072 tokens, `ZENITH_MAX_MODEL_LEN=131072` serves the model at its native context,
faster on short prompts ([section 3](#3-what-the-defaults-are-and-how-many-nodes)).

**A fine-tune** of the same model under another directory name gets the same defaults with
`ZENITH_TUNED_AS=Mistral-Large-Instruct-2411 ZENITH_MODEL=<its directory>` ([section 8](#8-scoring-a-fine-tune)).
If it ships no `params.json` / `tokenizer.model.v7`, copy them from the base model, or set
`ZENITH_VLLM_EXTRA_ARGS` without `--tokenizer-mode mistral` (copy the rest from the `lux: tuned defaults`
line).

## 2. Use it

```python
from openai import OpenAI
client = OpenAI(base_url="http://<head>:8000/v1", api_key="unused")
r = client.chat.completions.create(model="Mistral-Large-Instruct-2411",
                                   messages=[{"role": "user", "content": "What is 17*23?"}],
                                   max_tokens=1000)
print(r.choices[0].message.content)
```

* No reasoning parser: the model does not think in a separate channel, so everything is in `content`.
* Tool calls work with `"tools": [...]` and `"tool_choice": "auto"` (the `mistral` tool-call parser).
* The context is 1,048,576 tokens, prompt plus output. The KV pool holds 3.34M tokens: 3.2 full-length
  requests at once, or many more shorter ones.

## 3. What the defaults are, and how many nodes

| Setting | Value | Knob |
|---|---|---|
| parallelism | TP8, one node | `ZENITH_TP`, `sbatch --nodes` |
| context | 1,048,576 (YaRN factor 16 over the native 131,072, attention factor 1.15) | `ZENITH_MAX_MODEL_LEN`, `--hf-overrides` |
| KV cache | bf16, 200 GB per GPU target: 3.34M tokens | `ZENITH_TARGET_KV_GB` |
| attention backend | `ROCM_AITER_FA` | `--attention-backend` |
| tokenizer | Mistral's (`--tokenizer-mode mistral`) | |
| scheduler token budget | 32768 | `ZENITH_MAX_NUM_BATCHED_TOKENS` |
| largest CUDA-graph batch | 1024 | `ZENITH_MAX_CUDAGRAPH_CAPTURE_SIZE` |
| speculative decoding | off | |
| reasoning / tool-call parser | none / mistral | `ZENITH_REASONING_PARSER`, `ZENITH_TOOL_CALL_PARSER` |

The three vLLM flags go in `ZENITH_VLLM_EXTRA_ARGS`. Setting that variable replaces all three, so copy them from
the `lux: tuned defaults` line and add to them.

**Sizing.** Mistral Large has 88 layers with 8 KV heads of 128, so its bf16 KV cache takes 352 KiB per token:
**352 GiB for one 1M-token request**, on top of 229 GiB of weights. One node has 8 x 288 GiB.

| layout | weights per GPU | KV pool (vLLM) | full 1M requests at once | 8k tok/s/user at c=1 |
|---|---|---|---|---|
| TP8, 1 node (default) | 28.6 GiB | 3.34M tokens (4.4M with the vLLM default token budget and graphs) | 3.2 | 89.7 (95.7 with the unified backend) |
| TP8, 1 node, native 131,072 context (`ZENITH_MAX_MODEL_LEN=131072`) | 28.6 GiB | 4.23M tokens | - (32 at 131k) | 95.3 |
| TP4, 4 GPUs (x2 per node with `ZENITH_DP=2`) | 57 GiB | 2.22M tokens per copy | 2.1 per copy | 67.9 |
| TP2, 2 GPUs | 115 GiB | ~0.6M tokens | does not fit | |
| fp8 KV cache, TP8 | 28.6 GiB | 8.8M tokens | 8.4 | (fails retrieval, section 5) |

So **one node per copy of the model is enough for 1M tokens**, and TP8 over the whole node is the fastest
layout per user. A second node buys nothing for one copy: Mistral Large is dense, and splitting it TP8 x PP2
would double the KV pool (which one node already has room for) while adding a cross-node hop to every decode
step. To serve more users, run more one-node jobs and spread clients over them; each holds 3.2 full-length
requests.

**Two nodes: split by prompt length.** A 1M-token prompt stalls every other user on its node for minutes
(section 6), and the 1M configuration is slower on short prompts (section 5). With two nodes, run two one-node jobs and route by prompt length:

* a **native** node, `ZENITH_MAX_MODEL_LEN=131072`, for every request that fits in 128k tokens. Below 131,072
  the tuned defaults drop YaRN and use the `ROCM_AITER_UNIFIED_ATTN` backend (fastest at short prompts, 8k curve
  54.7 vs 48.2, job 222056), and the KV pool holds about 32 full-length requests;
* a **1M** node with the defaults, for the long prompts only.

```bash
ZENITH_MAX_MODEL_LEN=131072 ZENITH_MODEL=Mistral-Large-Instruct-2411 ... sbatch ... docker/serve_lux.slurm.sh
ZENITH_MODEL=Mistral-Large-Instruct-2411 ...                             sbatch ... docker/serve_lux.slurm.sh
```

The client (or a proxy in front of the two) counts the prompt's tokens, or estimates them at ~4.5 characters
per token for English, and sends anything over ~120k to the 1M node. Short users then never see a 1M prefill,
and short prompts get the faster backend. This is the layout to use for the hackathon's science work.

## 4. The 1M context: YaRN, and the `rope_theta` trap

Mistral Large 2411 is trained to 131,072 tokens. vLLM serves it past that with YaRN, passed as
`--hf-overrides '{"rope_scaling":{...}}'`.

**The override replaces the config's whole rope dict.** In this image's vLLM / transformers, the model's
`rope_theta` of 1,000,000 lives in that dict, so an override without it silently falls back to base 10,000.
The server comes up and short prompts look fine, but every prompt past a few thousand tokens comes back as
garbage, tool calls break (stray `}`) and GSM8K drops from 0.96 to 0.87. The default override therefore repeats
`"rope_theta":1000000.0`. `bench/mistral/ropecheck.py` prints the rope base, factor and attention scale vLLM
will actually build from an override, without loading the model:

```bash
cp bench/mistral/ropecheck.py <datasets dir>/      # the container sees the datasets dir at /app/datasets
podman exec <container> python3 /app/datasets/ropecheck.py '{"rope_scaling":{...}}' /app/models/<model dir>
```

**Factor 16, not 8.** 8 x 131,072 is exactly 1,048,576 positions, but with factor 8 the model lost the needle
at 1,040,000 tokens (it found it at 512k). Factor 16 found it, with the same GSM8K and short-context results.
Early screening, one needle at 50% depth:

| YaRN override (all with `rope_theta` 1e6) | GSM8K (250) | needle 128k / 512k / 1,040k |
|---|---|---|
| factor 8 | 0.948-0.956 | pass / pass / **fail** |
| factor 12 | 0.948 | pass / pass / fail |
| factor 16 | 0.956 | pass / pass / **pass** |
| factor 16, attention factor 1.4 | 0.928 | fail / fail / - |

**The attention factor is the knob that matters.** YaRN also multiplies the rotary embedding by an attention
factor, 0.1 ln(factor) + 1 unless set: 1.277 at factor 16. It sharpens attention at every length, so it changes
the model's answers to short questions too, and the science suite showed it: 0.609 at 1.277 against 0.662 at the
native context. Too little sharpening and the model cannot pick a needle out of 1M tokens. Scanning the factor at
YaRN factor 16 (science suite as in section 7; needle at 5 depths, 10% to 90%, unless noted):

| configuration | attention factor | science suite | GPQA / MaScQA / MMLU-Pro | GSM8K | probe | needle 512k | needle 1,040k |
|---|---|---|---|---|---|---|---|
| native 131,072 context, no YaRN | - | **0.662** | 0.531 / 0.779 / 0.676 | 0.964 | 10/10 | - | - |
| factor 16 | 1.0 | 0.664 | 0.531 / 0.783 / 0.678 | 0.968 | 10/10 | 2/3 | **0/3** |
| **factor 16 (default)** | **1.15** | **0.662** | 0.536 / 0.786 / 0.662 | **0.972** | 10/10 | **5/5** | **5/5** |
| factor 16 | 1.2 | 0.653 | 0.547 / 0.757 / 0.655 | 0.964 | 10/10 | 5/5 | 4/5 |
| factor 16 | 1.277 (YaRN's own) | 0.609, 0.609 (two runs) | 0.464 / 0.742 / 0.622 | 0.944, 0.956 | 9/10 | 3/3, 4/5 | 1/3, 4/5 |
| factor 8 | 1.0 | 0.651 | 0.497 / 0.780 / 0.676 | 0.968 | 10/10 | 3/3 | 0/3 |
| factor 8 | 1.208 (YaRN's own) | 0.644 | 0.520 / 0.769 / 0.644 | 0.960 | 10/10 | 3/3 | 2/3 |

So the default is factor 16 with `"attention_factor":1.15`: as accurate as the unscaled model on the science suite
and GSM8K, and it found the needle at all ten depths of 512k and 1,040k in that run. The needle test is noisy (each
run salts its haystack differently, and the 1.277 rows differ between two runs), so read the 1M column as "usually"
rather than "always" (Caveats). The attention factor does not change speed.

## 5. What was tuned

All at one node, TP8, bf16 weights, `--tokenizer-mode mistral`, 1M context unless noted. 8k-token prompts and
1k-token replies of natural prose through the chat API at temperature 0.7, 16+ requests per concurrency level.
The **curve score** is the geomean over concurrency c of sqrt(tok/s/user x tok/s/GPU), the same score as the
Inkling tuning (higher is better).

The 128k score is the same geomean over c = 1, 2, 4, 8 with 128,000-token prompts. GSM8K (250 questions) and the
correctness probe are quality gates, not tuning targets.

| configuration (TP8 unless noted) | 8k curve score | 8k c=1 tok/s/user | 8k c=128 tok/s/GPU | 128k score | 128k TTFT, c=1 | quality gates |
|---|---|---|---|---|---|---|
| `ROCM_AITER_UNIFIED_ATTN` | 54.4 | 95.7 | 199.9 | 16.8 | 13.7 s | 10/10, 0.956 |
| unified + 32k token budget | **54.8** | 95.2 | 204.1 | 17.5 | 12.8 s | 10/10, 0.952 |
| **`ROCM_AITER_FA`** | 47.5 | 89.7 | 168.4 | **18.4** | **10.3 s** | 10/10, 0.948 |
| unified + ngram speculative decoding | 46.1 | 90.5 | 161.9 | 14.5 | 14.0 s | 10/10, 0.952 |
| TP4 x DP2 (two copies per node), unified | 47.3 | 67.9 | 199.5 | 15.5 | 25.2 s | 10/10, 0.960 |
| unified + fp8 KV cache | | | | 19.5 | 10.7 s | 10/10, 0.964, **fails the 128k and 1M needles** |
| unified + fp8 weights (`--quantization fp8`) | | | | | | **8/10, 0.916** |
| default: FA + 32k token budget + graphs to 1024 | 48.2 | 90.2 | 168.6 | 18.4 | 10.3 s | 10/10, 0.972 (with attention factor 1.15) |

What each option did:

* **Attention backend.** This is the biggest knob. vLLM's default on this image (`ROCM_ATTN`) is very slow at a
  1M `max_model_len` and was dropped early; the Triton backend is slower still. `ROCM_AITER_UNIFIED_ATTN` decodes
  fastest, which wins the 8k curve. `ROCM_AITER_FA` prefills long prompts fastest: 128k TTFT 10.3 s vs 13.7 s, a
  512k prompt in 87 s vs 139 s, a 1,040,000-token prompt in **~5.2 minutes vs ~8.7**. At 8k it is 6% slower at
  c=1 and up to 17% slower per GPU at high concurrency. The default is FA because this is a 1M-context server and
  waiting minutes longer for every long prompt is the bigger cost; for a short-prompt workload set
  `--attention-backend ROCM_AITER_UNIFIED_ATTN` (section 6).
* **Scheduler token budget** (`ZENITH_MAX_NUM_BATCHED_TOKENS=32768`, from vLLM's 16384 here): +4% on the 128k
  score, neutral at 8k. It costs some KV pool (activations for bigger chunks), which a 3.2x pool can afford.
* **CUDA graphs to batch 1024** (`ZENITH_MAX_CUDAGRAPH_CAPTURE_SIZE`, from vLLM's 512): decode at c > 512 stays
  on graphs.
* **Speculative decoding.** Mistral Large has no MTP head, and training a draft model was out of scope, so the
  option on the table is n-gram (prompt lookup). It slows natural-prose chat at every concurrency (-15% curve
  score) because few drafts are accepted. Off.
* **Parallelism.** One copy fits a node (section 3). TP4 x DP2 runs two copies per node: about equal per GPU at
  high concurrency, 29% slower per user at c=1, and twice the TTFT at 128k. TP8.
* **KV cache dtype.** fp8 KV doubles the pool to 8.8M tokens and is slightly faster at 128k, but the model then
  misses the needle at 128k and at 1M. bf16.
* **Weights.** Dynamic fp8 weights lose correctness and GSM8K. bf16.

**Disaggregated prefill/decode was not used.** Cross-node NIXL over Lux's `ionic` NICs works as an experiment
(README section 9: ~198 GB/s of GPU-to-GPU reads per node), but `docker/serve_disagg.sh` has produced no result on
Lux, and making it dependable was more than the few hours this tuning had. For this model it would mean one TP8
prefill node and one TP8 decode node per copy: twice the hardware. Moving a 1M prompt's 352 GiB of KV at that
bandwidth takes ~2 s, small next to its 5-minute prefill, so the transfer is not the obstacle. What disaggregation
buys is shielding the decode of other users from a long prefill, and that turns out to matter a great deal
(section 6): **while one 1M-token prompt prefills, everyone else on the node gets almost nothing**. Capping the
long prompt's share of each step (`--long-prefill-token-threshold 2048`) only raises the others from ~0.1 to
~1 token/s each and stretches the 1M prefill. Disaggregation is one way out; the simpler one, which needs nothing
new, is to keep long prompts off the nodes that serve short ones (section 3, "Two nodes").

## 6. Performance (default configuration)

Job 219229, one node, the package defaults (FA, YaRN factor 16, 32k token budget, graphs to 1024; it ran with
YaRN's own attention factor, which does not change speed). Same clients as section 5. Files:
`results/mistral/lux_1node_1m_2026-10-11/`.

8k-token prompts, 1k-token replies, chat API, T=0.7 (curve score **48.2**; 54.4 with the unified backend):

| c | tok/s/user | tok/s/GPU | TTFT mean |
|---|---|---|---|
| 1 | 90.2 | 10.8 | 0.48 s |
| 4 | 82.2 | 36.7 | 1.48 s |
| 16 | 59.3 | 93.0 | 4.55 s |
| 64 | 27.0 | 151.1 | 14.6 s |
| 128 | 15.6 | 168.6 | 27.9 s |
| 256 | 8.5 | 179.4 | 54.4 s |
| 1024 | 5.0 | 178.6 | 312 s |

Past c=128 the extra requests mostly queue for prefill: tok/s/GPU flattens near 180 and TTFT grows. 128,000-token
prompts (score **18.4**):

| c | tok/s/user | tok/s/GPU | TTFT mean |
|---|---|---|---|
| 1 | 78.6 | 5.5 | 10.3 s |
| 2 | 57.0 | 7.3 | 15.9 s |
| 4 | 37.7 | 9.0 | 25.5 s |
| 8 | 21.9 | 9.8 | 46.3 s |

Long prompts, one at a time (`long_context_check.py`, job 219230): prefill 16.7k tok/s at 32k, 12k at 128k, 8.9k at
256k, 5.7k at 512k and 3.3k at 1,040,000 tokens, which is **a 1M-token prompt in 5.1-5.3 minutes**. Decode slows
with depth: 89 tok/s at 32k, 78 at 128k, 53 at 512k, 35.5 at 1M.

**One 1M prompt and other users** (job 221159): 16 users with 8k prompts decode 1,080 tok/s in total (59 tok/s
each). Sixty seconds after a 1,040,000-token prompt arrives, their total drops to 0-3 tok/s and stays there for
the **7.5 minutes** that prompt takes to prefill under load (451 s instead of 305 s alone); new short requests wait
the whole time. Each scheduler step spends its 32k-token budget on a chunk of the long prompt, and at a depth of
hundreds of thousands of tokens such a step takes seconds. With `--long-prefill-token-threshold 2048` (job 221160)
the others get 10-25 tok/s in total, about 1 tok/s each, and the long prefill slows to ~1.5k tok/s.

## 7. Science baseline

`bench/science/` ([its README](../bench/science/README.md)) runs three public benchmarks zero-shot with chain
of thought at temperature 0:

* **GPQA Diamond**, physics and chemistry (179 questions): graduate-level, written by domain experts to resist
  look-up;
* **MaScQA** (646): materials science from India's GATE graduate entrance exams;
* **MMLU-Pro** physics, chemistry and engineering (3,400): broad undergraduate coverage, with tight intervals.

The suite score is the unweighted mean of the three accuracies. Fetch the data once on a login node, then run
the suite against the server (about 10 minutes at concurrency 128):

```bash
ZENITH_DATASETSPATH=<datasets dir> python3 bench/science/fetch_science_data.py
python3 bench/science/science_suite.py --host <head> --port 8000 \
    --data-dir <datasets dir>/science --concurrency 128 --out science.json | tee science.txt
```

**Keep GPQA and `science.json` private**: the GPQA authors ask that its questions not be posted online, and the
JSON holds every question and reply. `science.txt` has only the scores.

**Baseline: suite score 0.662** for Mistral-Large-Instruct-2411 on the native configuration
(`ZENITH_MAX_MODEL_LEN=131072`, the configuration to score fine-tunes on; job 222056,
`results/mistral/lux_1node_1m_2026-10-11/native_131k/science.txt`):

| benchmark | correct | accuracy | 95% interval |
|---|---|---|---|
| GPQA Diamond (physics 0.628, chemistry 0.441) | 95 / 179 | **0.531** | 0.458-0.602 |
| MaScQA | 503 / 646 | **0.779** | 0.745-0.809 |
| MMLU-Pro (physics 0.737, chemistry 0.734, engineering 0.528) | 2,300 / 3,400 | **0.676** | 0.661-0.692 |
| **suite score** (mean) | | **0.662** | about +-0.027 |

By field over all three: physics 0.730, chemistry 0.712, materials 0.779, engineering 0.528. The run takes 10.5
minutes at concurrency 128, and fewer than 2% of replies hit the 8,192-token limit or give no answer.

Two more runs of the same model at native context agree within the noise: 0.669 (job 218296, same backend) and
0.659 (job 220263, `ROCM_AITER_FA`). The 1M defaults score the same, 0.662 (job 222053) and 0.661 (job 224645, the
final check of the zenith v7 package, `final_1m/`), and on the v8mi250 image 0.656 and 0.661 (jobs 237542 and
239566; the previous image, rerun alongside the second, scored 0.646 in job 239567), thanks to their YaRN attention factor of 1.15; with YaRN's
own 1.277 the score is 0.609 (section 4). Still score fine-tunes on the native configuration: it is faster (the
run above took 10.5 minutes, 16 at 1M) and leaves out one more setting that a fine-tune could interact with.

## 8. Scoring a fine-tune

Put the fine-tune's directory next to the base model's (same `ZENITH_MODELPATH`), check that it has
`params.json` and `tokenizer.model.v7` (copy them from the base model if it was saved with Hugging Face files
only, as long as the vocabulary is unchanged), then run the same job on it with the native context:

```bash
cd <this package>
ZENITH_MODELPATH=<dir holding both model dirs> \
ZENITH_IMAGE_TAR=<path to inference_image_gfx950_v8mi250_20261011.tar> \
ZENITH_DATASETSPATH=<datasets dir> \
ZENITH_MAX_MODEL_LEN=131072 \
ML_MODEL=<fine-tune dir name> ML_STEPS="correctness tool gsm8k science" \
  sbatch -A <account> -p batch bench/mistral/mistral_bench.slurm.sh
```

`ML_MODEL` other than the base name sets `ZENITH_TUNED_AS=Mistral-Large-Instruct-2411`, so the fine-tune gets
the same defaults, and the results land in `bench/mistral/runs/<name>.<jobid>/` (`science.txt` has the
scores). The whole job takes about 30 minutes. Compare it with the base model's score from the same configuration
(section 7). Mind the noise: GPQA's 179 questions give it a 95% interval of about +-7 points, which makes the
suite score's about +-2.7; two runs of the base model with different attention backends differed by 1 point.
MMLU-Pro (3,400 questions, +-1.6) is the most precise of the three.
Add `ML_STEPS="... needle"` without `ZENITH_MAX_MODEL_LEN` to check that a fine-tune still handles long context.

## Caveats

* **Retrieval at 1M is not guaranteed.** The model is trained to 131k. With the default YaRN settings it found
  a single needle at all ten depths of 512k and 1,040k in one run, but needle results vary from run to run and
  with the settings (section 4), and one needle is an easy test. Check retrieval on your own documents.
* **YaRN's attention factor matters**: YaRN's own value (1.277) costs about 5 points on the science suite at
  every prompt length; the default 1.15 does not. If you change the YaRN override, keep `rope_theta` in it and
  re-check both (section 4).
* **A 1M prompt stalls its node** for the 5-8 minutes it prefills (section 6). Keep long prompts on their own node.
* **The 1M defaults trade short-prompt speed for long-prompt speed** (`ROCM_AITER_FA`, 8k curve 48.2 vs 54.7
  on a native node).
* **No speculative decoding or disaggregation** was found worth shipping in the time available (section 5).
* **GPQA is private**: never commit or publish its questions or `science.json` (section 7).
* Numbers are from single runs of each configuration on one Lux node; run-to-run spread is ~1-2% for throughput and
  larger for the needle and GPQA results.
