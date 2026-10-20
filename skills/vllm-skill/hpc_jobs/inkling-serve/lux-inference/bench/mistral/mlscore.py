#!/usr/bin/env python3
"""usage: mlscore.py [--gpus 8] name=path.jsonl ...  -> per-c table and curve score over the c values all runs share."""
import json, math, sys

args = sys.argv[1:]
gpus = 8
if args and args[0] == "--gpus":
    gpus, args = int(args[1]), args[2:]
runs = {}
for a in args:
    name, path = a.split("=", 1)
    runs[name] = {r["concurrency"]: r for r in map(json.loads, open(path)) if r.get("ok")}
common = sorted(set.intersection(*(set(d) for d in runs.values()))) if runs else []


def score(d, cs):
    v = [math.sqrt(d[c]["per_user_decode_tok_s"] * d[c]["output_tok_s"] / gpus) for c in cs]
    return math.exp(sum(map(math.log, v)) / len(v)) if v else float("nan")


print(f"curve score over c={common} ({gpus} GPUs): " +
      ", ".join(f"{n} {score(d, common):.1f}" for n, d in runs.items()))
for n, d in runs.items():
    print(f"## {n}  (all c: score {score(d, sorted(d)):.1f})")
    print(f"{'c':>5} {'tok/s/user':>10} {'tok/s/GPU':>10} {'ttft s':>8}")
    for c in sorted(d):
        r = d[c]
        print(f"{c:5d} {r['per_user_decode_tok_s']:10.1f} {r['output_tok_s'] / gpus:10.1f} {r['ttft_mean_s']:8.2f}")
