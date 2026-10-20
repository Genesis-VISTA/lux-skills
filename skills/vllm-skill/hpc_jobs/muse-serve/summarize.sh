#!/bin/bash
# Summarize a muse-serve job from its output dir: startup milestones, fabric, KV pool, correctness,
# benchmark scores, failure signatures.   summarize.sh <output dir> [job log (default <dir>/job.log)]
D=${1:?usage: summarize.sh <output dir> [job log]}
L=${2:-$D/job.log}
[ -f "$L" ] || { echo "no job log at $L"; exit 1; }
grep -E '^=== (BUILD|UP|FABRIC|SERVING|BENCH|server exited|exit)' "$L" || echo "(not up yet: loading the image and model takes ~10-15 min; a build ~1 h more)"
grep -hE 'GPU KV cache size|Maximum concurrency' "$L" | sort -u | sed -E 's/^.*(GPU KV cache size)/\1/'
grep -E 'overall: [0-9]+/[0-9]+' "$L" | tail -1 | sed 's/^ */correctness probe: /'
grep -hE 'lux: tuned defaults' "$D/server.log" 2>/dev/null | head -1 | cut -c1-200
grep -hE 'TP all-reduce \(|stage [0-9]+ -> [0-9]+:' "$D/net_transports.txt" 2>/dev/null
[ -f "$D/science.txt" ] && { echo "--- science suite"; sed -n '/^suite score/,/^{/p' "$D/science.txt" | grep -v '^{'; }
[ -f "$D/mmlu_pro.txt" ] && { echo "--- MMLU-Pro subset"; grep -E '^MMLU-Pro subset' "$D/mmlu_pro.txt"; }
[ -f "$D/gsm8k.txt" ] && { echo "--- gsm8k"; tail -1 "$D/gsm8k.txt"; }
# Failure signatures worth surfacing (server.log is long and noisy on purpose; never grep it for "error")
F=$(grep -hE 'FABRIC CHECK FAILED|INFEASIBLE|OutOfMemory|out of memory|HIP error|RuntimeError|Watchdog caught|NCCL error|health check FAILED|container startup failed|BUILD FAILED|^ERROR' \
      "$L" "$D/server.log" 2>/dev/null | cut -c1-240 | sort | uniq -c | sort -rn | head -8)
[ -n "$F" ] && { echo "--- possible failures"; echo "$F"; }
exit 0
