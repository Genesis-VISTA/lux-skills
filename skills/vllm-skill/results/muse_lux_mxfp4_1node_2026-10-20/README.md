# muse_lux_mxfp4_1node_2026-10-20

Muse-Spark-Open-816B-A42B-Quark-MXFP4 on one Lux node (8x MI355X), zenith v0.1.9's best single-node recipe, run on
Lux under account stf218 (job 361678, 2026-10-20, 1 h 22 min, exit 0). Hostnames are rewritten to `<headnode>`, the
user's scratch directory to `<scratch>` and the job's output directory to `<out>`.

How it ran:
- **Launcher:** zenith's own `bench/muse/muse_ladder_spec.sbatch` from a zenith v0.1.9 checkout (commit `70e5f0b`),
  with only the account and paths changed. It was not run through this skill's `muse-serve` job, which serves with
  the same settings but has no throughput ladder (MUSE_SPARK.md §4).
- **Image:** `inference_image_gfx950_musemain2.tar`, built on Lux with the engine's `build_image.sh gfx950` and the
  Muse pins (MUSE_SPARK.md §2; 44 min, build checks passed). It is not the shared tar.
- **Weights:** `amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4` (78 files, sizes checked against Hugging Face) and the
  DSpark draft (sha256 verified), downloaded on a Lux login node.
- **Command:** the one in MUSE_SPARK.md §4:
  `ZENITH_TARGET_KV_GB=180 MUSE_SPEC=dspark MUSE_CG_MAX=1024 MUSE_MNBT=8192 MUSE_EXTRA_ENV=ZENITH_MUSE_FLYDSL_TUNED=1`
  with the K schedule `[[1,4,7],[5,16,4],[17,256,3],[257,2048,0]]`. The full environment is in `config.txt`.

| check | this run (361678) | zenith release re-run (360800) |
|---|---|---|
| start-up (job start to serving) | 12 min | not recorded in its results |
| GPU KV cache | 12,499,640 tokens | not recorded in its results |
| correctness probe | 20/20 | 20/20 |
| gsm8k, 250 questions, effort low | 0.972, 0 errors, 0 truncated, 36 s | 0.976 (on the `-r63` mirror) |
| DSpark acceptance during gsm8k | 3.11 tokens per step | not recorded in its results |
| curve score, 8k in / 1k out, c = 1..1024 | **72.23** | 71.96 |

The curve score is zenith's: the geometric mean over the 11 rungs of sqrt(output tok/s/user x output tok/s/GPU), a
rung scoring zero if its median TTFT is over 60 s. Recomputed from `ladder.txt` the same way, 360800 gives 71.96.

**Throughput ladder** (`points.txt`, `ladder.txt`). Every rung completed all its requests (`clean`). tok/s/user is
1000 / median TPOT; total tok/s counts input and output tokens over the node's 8 GPUs.

| c | tok/s/user | total tok/s/GPU | output tok/s (node) | median TPOT | median TTFT | tokens accepted per step |
|---|---|---|---|---|---|---|
| 1 | 184.6 | 202 | 180 | 5.4 ms | 0.26 s | 3.90 |
| 2 | 138.9 | 307 | 273 | 7.2 ms | 0.32 s | 3.34 |
| 4 | 116.1 | 428 | 381 | 8.6 ms | 0.33 s | 2.73 |
| 8 | 101.9 | 824 | 732 | 9.8 ms | 0.34 s | 2.81 |
| 16 | 81.1 | 1,204 | 1,071 | 12.3 ms | 0.36 s | 2.77 |
| 32 | 59.7 | 1,702 | 1,513 | 16.7 ms | 0.58 s | 2.65 |
| 64 | 36.7 | 2,022 | 1,798 | 27.3 ms | 0.65 s | 2.74 |
| 128 | 21.8 | 2,423 | 2,153 | 45.8 ms | 0.98 s | 2.77 |
| 256 | 12.4 | 2,850 | 2,534 | 80.9 ms | 2.07 s | 2.70 |
| 512 | 5.9 | 3,237 | 2,877 | 169.4 ms | 1.33 s | 2.64 |
| 1024 | 3.9 | 3,613 | 3,212 | 258.5 ms | 43.96 s | 2.68 |

Against 360800, tok/s/user is within +-1% at 8 of the 11 rungs (+5.4% at c = 4 at most), and tok/s/GPU within
+-5% except at c = 2 (-6.7%). At c = 1024 the KV cache is full
(99.5% used, about 900 requests running and the rest waiting), hence the 44 s TTFT, the same as upstream.

**Real prompts** (`realprompt_c1.json`, `realprompt_c8.json`; chat prompts, not the 8k/1k ladder):

| c | requests | median decode tok/s/user | output tok/s | mean reply tokens | tokens accepted per step |
|---|---|---|---|---|---|
| 1 | 8 | 280.7 | 260.7 | 707 | 4.08 |
| 8 | 64 | 176.0 | 1,129.9 | 657 | 3.55 |

| file | what |
|---|---|
| `config.txt` | the launch environment (`ZENITH_*`, `MUSE_*`, the vLLM args) |
| `correctness.json`, `.txt` | the 20-prompt correctness probe |
| `realprompt_c1.json`, `realprompt_c8.json` | the real-prompt check at c = 1 and 8 |
| `ladder.txt`, `points.txt` | `bench/ix_ladder.sh` per rung, and `bench/points.py`'s summary |
| `gsm8k.json`, `.txt`, `accept_gsm8k.txt` | `gsm8k_chat.py --n 250` at effort low, and DSpark's acceptance during it |
