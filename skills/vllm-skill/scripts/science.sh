#!/usr/bin/env bash
# Run the science suite against a server that is already up (from the login node).
#   scripts/science.sh <head node | http://head:8000/v1> [science_suite.py args ...]
#   scripts/science.sh lux326 --n 20                      # smoke test, ~60 questions
#   scripts/science.sh "$(scripts/endpoint.sh | tail -1)"  # your running inkling.serve job
#   scripts/science.sh lux326 --benchmarks gpqa,mascqa     # a subset
# Writes runs/science.<host>.<timestamp>.{txt,json}. The JSON holds every GPQA question: keep it private.
set -eu
SK=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
source "$SK/env.lux.sh"
T=${1:?usage: science.sh <head node or URL> [science_suite.py args]}; shift
H=${T#http://}; H=${H%%/*}; PORT=8000
case "$H" in *:*) PORT=${H#*:}; H=${H%%:*} ;; esac
mkdir -p "$LUX_RUNS_DIR"
O=$LUX_RUNS_DIR/science.$H.$(date +%Y%m%d-%H%M%S)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY   # compute nodes are not behind the proxy
PY=${PYTHON:-$(for p in python3.12 python3.11 python3.10 python3.9 python3; do command -v $p >/dev/null && { echo $p; break; }; done)}   # needs >= 3.9
( umask 077; $PY "$ENGINE_DIR/bench/science/science_suite.py" --host "$H" --port "$PORT" \
    --data-dir "${LUX_SCIENCE_DATA:-$LUX_DATASETSPATH/science}" --concurrency 256 --max-tokens 16384 "$@" \
    --out "$O.json" ) | tee "$O.txt"
echo "wrote $O.txt (scores) and $O.json (per-question, private)"
