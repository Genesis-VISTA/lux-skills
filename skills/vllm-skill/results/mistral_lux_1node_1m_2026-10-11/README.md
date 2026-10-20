# Mistral-Large-Instruct-2411 on 1 x 8 MI355X, 2026-10-11

Runs behind `docs/LUX_MISTRAL_LARGE.md`, made with the zenith_v6 image and the package's `docker/` launchers on
Lux. Each directory is one Slurm job running `bench/mistral/mistral_bench.slurm.sh`'s steps against one server.
Hostnames, user names and scratch paths are rewritten to `<headnode>`, `<user>`, `<models>`, `<package>` and so
on. `science.txt` files hold scores only; the per-question `science.json` files are not shipped (the GPQA
authors ask that its questions not be posted online).

| directory | job | configuration | what |
|---|---|---|---|
| `final_1m/` | 224645 | the shipped defaults: 1,048,576 context, YaRN factor 16 with attention factor 1.15, `ROCM_AITER_FA`, KV 200, 32k token budget, graphs to 1024 | the final check of the shipped package: `startup.txt` (KV 3,341,520 tokens), `correctness.txt` (10/10), `tool.txt`, `gsm8k.txt` (0.976), `long_context.txt` / `.json` (15/15: 32k-1,040k at depths 0.1/0.5/0.9; 1,040k in 303 s), `science.txt` (0.661) |
| `default_1m/` | 219229 | the same with YaRN's own attention factor (1.277), which the defaults used until the scan in `yarn_scale/`; the attention factor does not change speed | `startup.txt` (KV 3,341,520 tokens), `correctness.txt` (9/10), `tool.txt`, `gsm8k.txt` (0.944), `science.txt` (0.609), 8k curve c=1..1024 (`bench_8k.jsonl`, `bench_8k_c*.txt`; score 48.2), 128k curve c=1..8 (`bench_128k*`; score 18.4) |
| `needle_1m/` | 219230 | as `default_1m/` | `long_context.txt` / `.json`: needle retrieval at 32k-1,040k, depths 0.1/0.5/0.9 (12/12 to 512k, 1/3 at 1,040k), with prefill and decode rates |
| `mixed_1m/` | 221159 | as `default_1m/` | 16 users with 8k prompts, alone (`bench_mixed_alone_c16.txt`) and while a 1,040,000-token prompt arrives (`bench_mixed_with1M_c16.txt`, `mixed_long_c16.txt`); `mixed_log_c16.txt` is vLLM's periodic throughput log over that window |
| `mixed_1m_lpt2k/` | 221160 | as `default_1m/` + `--long-prefill-token-threshold 2048` | the same test (`mixed_log_c16_partial.txt`: the log was cut short) |
| `native_131k/` | 222056 | `ZENITH_MAX_MODEL_LEN=131072`: no YaRN, `ROCM_AITER_UNIFIED_ATTN` | the science baseline `science.txt` (0.662), `correctness.txt` (10/10), `tool.txt`, `gsm8k.txt` (0.964), 8k curve (score 54.7) |
| `yarn_scale/<variant>/` | 220264-6, 222053-5 | 1M context, YaRN factor 16 (`f16_`) or 8 (`f8_`) with attention factor 1.0, 1.15, 1.2 or YaRN's own (`default`) | `science.txt`, `gsm8k.txt`, `correctness.txt`, `long_context.txt`: the table in section 4 of the doc |

`mlscore.py` in `bench/mistral/` recomputes the curve scores from the `.jsonl` files:

```bash
python3 bench/mistral/mlscore.py default=results/mistral/lux_1node_1m_2026-10-11/default_1m/bench_8k.jsonl
```
