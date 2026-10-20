# lux_gate_v8mi250_2026-10-12

The shipping gate for this package's Lux image, `inference_image_gfx950_v8mi250_20261011.tar`: Inkling-MXFP4 on
MI355X (and Inkling-bf16 in one run), each run from a fresh copy of the tree with the package's plain defaults (no `ZENITH_*` knobs beyond the
model and image paths), 2026-10-11/12. `docs/LUX_HACKATHON.md` sections 4-7 discuss the numbers. Hostnames are
rewritten to `<headnode>` / `<node2>`, IP addresses to `<ip>`.

| directory | configuration | correctness | GSM8K | MMLU-Pro subset | needles | 8k curve score |
|---|---|---|---|---|---|---|
| `2node_237537/` | the tutorial: 2 nodes, TP8 x PP2, 1M context, reasoning on | 10/10 | 0.976 | 242/280 = 0.864 | 15/15 to 1,039,797 tokens | 54.5 |
| `2node_repeat_237538/` | the same, another allocation | 10/10 | 0.968 | 241/280 = 0.861 | 15/15 | 54.5 |
| `2node_from_package_239568/` | the same, extracted from this package's tarball (a release candidate that differs from the final one only in documentation and this results directory) | 10/10 | 0.968 | 247/280 = 0.882 | 15/15 | 54.7 |
| `2node_mtp3_237539/` | the tutorial with `ZENITH_MTP=1` | 10/10 | 0.976 | 243/280 = 0.868 | 15/15 | 46.8 |
| `1node_237540/` | 1 node, TP8, MTP-3 (the tuned single-node defaults) | 10/10 | 0.968 | 248/280 = 0.886 | 32k only (below) | 94.9 |
| `1node_previous_image_237541/` | the same on the previous image (zenith_v6) with the zenith v7 package, started at the same time | 10/10 | 0.976 | 242/280 = 0.864 | 32k only | 89.5 |
| `1node_1m_255825/` | 1 node as above with `ZENITH_MAX_MODEL_LEN=1048576`, from this package's tarball (KV 4,503,472 tokens) | 10/10 | 0.968 | 246/280 = 0.879 | 15/15 to 1,039,797 tokens | c=1, 64, 1024 only (below) |
| `2node_bf16_255824/` | `ZENITH_MODEL=Inkling-bf16`, 2 nodes, TP8 x PP2, 100k context (generic defaults), from this package's tarball | 10/10 | 0.980 | 241/280 = 0.861 | 6/6 at 32k and 90k | c=1, 64, 1024 only (below) |

The curve score is the geomean over c=1..1024 of sqrt(tok/s/user x tok/s/GPU), over 16 GPUs on two nodes and 8 on
one. The 1-node runs served with a 100k-token `max_model_len`, so their 128k and 256k needles were rejected (HTTP
400) and they have no 128k benchmark; that is the gate's configuration, the same for both images.

The last two are quick tests run after the gate, from a package tarball that differs from the shipped one only
in documentation and this results directory. They measured three points of the 8k curve (no curve score):

| | c=1 tok/s/user | c=64 output tok/s (TTFT) | c=1024 output tok/s |
|---|---|---|---|
| `1node_1m_255825/` | 157.4 | 2,013 (13.6 s) | 3,649 |
| `1node_237540/` (100k context) | 157.5 | 2,529 (7.0 s) | 3,764 |
| `2node_bf16_255824/` | 64.8 | 1,764 (4.9 s) | 4,885 |
| `2node_237537/` (MXFP4, 1M context) | 84.9 | 2,485 (4.1 s) | 6,003 |

`1node_1m_255825/` also ran the ~119k-token benchmark at c=1 and 2.

| file | what |
|---|---|
| `net_transports.txt` (2-node runs) | every cross-node channel vLLM's RCCL communicators logged (all `RCCL-ANP/<n>/GDRDMA`), plus the launch health check's TP all-reduce and PP handoff bandwidth |
| `patches.txt` | the fix-up and overlay lines the containers printed at start (the 25 MI250X fix-ups are reported as skipped; on this image they are already baked in, README §8) |
| `correctness.json`, `.txt` | `bench/inkling/correctness_probe.py` |
| `gsm8k.json`, `.txt` | `bench/inkling/gsm8k_chat.py --n 250` |
| `mmlu_pro.json`, `.txt` | `bench/inkling/mmlu_pro_quick.py`, 280-question subset, reasoning on |
| `long_context.json`, `.txt` | `bench/inkling/long_context_check.py`: needle at 10/50/90% depth in prompts of 32k to 1,039,797 tokens |
| `bench_8k.jsonl`, `bench_8k_c*.txt` | `bench/inkling/bench_throughput.py`, 8k in / 1k out, c=1..1024 |
| `bench_128k.jsonl`, `bench_128k_c*.txt` | the same client with ~119k-token prompts, c=1..8 (2-node MXFP4 runs; c=1, 2 in `1node_1m_255825/`) |
