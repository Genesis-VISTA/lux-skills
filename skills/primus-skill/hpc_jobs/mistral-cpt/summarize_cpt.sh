#!/bin/bash
# Summarize a mistral-cpt train.log: preflight, every iteration (step time, TFLOP/s/GPU, loss, grad norm),
# validation losses, checkpoints, exit and verdict.   summarize_cpt.sh <train.log> [exit code]
L=${1:?usage: summarize_cpt.sh <train.log> [exit code]}
rc=${2:-}
clean() { sed 's/\x1b\[[0-9;]*m//g' "$L"; }
head -2 "$L" | grep -E '^job |^warning'
clean | grep -aE 'nodes RAILS OK|RAILS BAD|preflight FAILED|nodes GPU_CHECK ALL OK|^=== stopped by' | sort -u
run=$(clean | grep -aoE '^=== [A-Za-z0-9._-]+_n[0-9]+_j[0-9]+_[0-9-]+$' | head -1 | cut -c5-)
[ -n "$run" ] && echo "run: $run"
# Iteration lines come from the last rank, each printed twice; keep one per iteration, in order.
clean | grep -a 'iteration .*/' | sed -E 's/.*(iteration +[0-9]+\/ *[0-9]+).*per iteration \(ms\): ([0-9.]+).*TFLOP\/s\/GPU\): ([0-9.]+).*lm loss: ([0-9.E+-]+).*grad norm: ([0-9.]+).*/\1  step \2 ms  \3 TFLOP\/s\/GPU  loss \4  grad norm \5/' \
  | sed -E 's/ +\/ */\//' | awk '!seen[$2]++' | sort -k2,2n
clean | grep -a 'validation loss at' | sed -E 's/.*(validation loss at[^|]*\| lm loss value: [0-9.E+-]+).*/\1/' | sort -u
clean | grep -aE 'saved checkpoint|successfully saved checkpoint' | sed -E 's/.*(saved checkpoint.*)/\1/' | sort -u | tail -3
[ -n "$run" ] && grep -aqE 'saved checkpoint' "$L" && echo "checkpoints: ${WORKDIR:-<work dir>}/output/amd/root/$run/checkpoints/"
clean | grep -aE '^=== exit|^GRAD GUARD|Training execution failed|OutOfMemory|out of memory|HIP error|NCCL error' | sort -u | head -6
n=$(clean | grep -a 'iteration .*/' | grep -oE 'iteration +[0-9]+' | sort -u | wc -l)
if [ "$rc" = 0 ]; then echo "VERDICT: OK ($n iterations)"; elif [ -n "$rc" ]; then echo "VERDICT: FAILED (exit $rc after $n iterations)"; fi
exit 0
