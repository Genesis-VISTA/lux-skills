#!/usr/bin/env bash
# Find your running server, wait until it answers, print its OpenAI base URL.
#   scripts/endpoint.sh               # newest RUNNING inkling.serve job of yours
#   scripts/endpoint.sh <jobid>       # a specific job (also inkling.bench)
#   export INKLING_URL=$(scripts/endpoint.sh | tail -1)    # for tutorial 1 / any OpenAI client
set -eu
if [ -n "${1:-}" ]; then
  job="$1 $(squeue -h -j "$1" -o '%N')"
else
  job=$(squeue -h -u "$USER" -n inkling.serve -t RUNNING -o '%i %N' | sort -n | tail -1)
fi
[ -n "${job#* }" ] || { echo "no running server job (squeue -u $USER); start one with scripts/serve.sh" >&2; exit 1; }
head=$(scontrol show hostnames "${job#* }" | head -1)
URL=http://$head:8000/v1
echo "job ${job%% *}, head node $head; waiting for $URL/models (cold start ~11-13 min) ..." >&2
for i in $(seq 1 120); do
  curl -sf -m 10 -o /dev/null "$URL/models" && break
  [ "$i" = 120 ] && { echo "no answer after 60 min; see scripts/summarize.sh ${job%% *}" >&2; exit 1; }
  sleep 30
done
echo "up: model $(curl -sf -m 10 "$URL/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')" >&2
echo "$URL"
