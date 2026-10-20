#!/usr/bin/env bash
# Serve, benchmark, stop: one job.   scripts/bench.sh [nodes] [KEY=VAL ...]
#   scripts/bench.sh                                  # 2 nodes, full science suite (GPQA + MaScQA + MMLU-Pro)
#   scripts/bench.sh 2 LUX_SCIENCE_ARGS="--n 20"      # smoke test: first 20 questions of each benchmark
#   scripts/bench.sh 2 LUX_BENCH="science mmlu needle"  # + MMLU-Pro 280-q subset + 1M-token needle check
#   scripts/bench.sh 1 LUX_MODEL=Mistral-Large-Instruct-2411   # score another model (1 node, 1M context)
# Steps: science (~4,225 q), mmlu (280 q, ~7 min), needle (15 cells to 1,040k tokens, ~6 min).
# Results land in runs/serve.<jobid>/ (science.txt: scores only; science.json: every GPQA question, keep private).
set -eu
SK=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
N=${1:-2}; shift || true
export LUX_BENCH=${LUX_BENCH:-science}
export JOB_NAME=${JOB_NAME:-inkling.bench}            # not inkling.serve: tutorial 1 must not latch onto it
export SBATCH_ARGS="-t ${BENCH_TIME:-02:00:00} ${SBATCH_ARGS:-}"
exec "$SK/scripts/serve.sh" "$N" "$@"
