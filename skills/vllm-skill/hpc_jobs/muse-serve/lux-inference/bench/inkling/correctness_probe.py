#!/usr/bin/env python3
"""
Correctness gate for build / kernel-backend changes (OPTIMIZATION_PLAN.md step 1).

WHY THIS IS NOT A BYTE DIFF
---------------------------
The obvious design -- fixed seed, temperature 0, diff the text against a
reference build -- does not work on this model, and the first version of this
script made exactly that mistake. Measured on job 46645 (stock build,
bc0f47cd0), running the *same* prompts twice against the *same* server with
the same seed gave **1 of 8 outputs byte-identical**. Common prefixes ranged
from 5 to 583 characters before diverging.

That is not a bug: batched MoE serving is nondeterministic by construction.
Expert routing and the order of floating-point reductions both depend on how
requests happen to be batched together, which depends on arrival timing. Greedy
sampling removes sampling noise but not this. Once two builds' logits differ in
the last bits, argmax eventually picks a different token and the texts diverge
irreversibly -- while both remain perfectly correct.

So a byte diff has a false-positive rate near 100% and gates nothing. Worse, it
would have "detected" a regression in MTP that was not there: MTP vs baseline
scored 3/8 identical, *better* than baseline vs itself at 1/8.

WHAT THIS CHECKS INSTEAD
------------------------
Prompts with verifiable answers, checked semantically, sampled N times each to
average over batching noise. The pass condition is that the accuracy rate does
not drop against the reference build. This is what would actually have caught
PR #56433, whose symptom was "loads, serves, returns subtly wrong tokens" --
wrong content, not different-but-equivalent content.

Usage:
  python3 correctness_probe.py --host <ip> --out results/correctness/ref.json
  python3 correctness_probe.py --host <ip> --out results/correctness/new.json
  python3 correctness_probe.py --compare results/correctness/ref.json \
                                         results/correctness/new.json
"""
import argparse
import json
import re
import sys
import urllib.request

# Each check: (prompt, predicate, description). The predicate is applied to the
# lowercased completion. These are deliberately easy -- the gate is looking for
# a model that has become subtly broken, not measuring capability. A healthy
# model should score at or near 100%.
CHECKS = [
    ("The capital city of France is the city of",
     lambda t: "paris" in t,
     "factual: Paris"),
    ("Q: What is 17 multiplied by 23?\nA: The answer is exactly",
     lambda t: "391" in t,
     "arithmetic: 17*23=391"),
    ("Q: List the first 8 prime numbers in ascending order, comma separated.\nA:",
     lambda t: all(re.search(rf"\b{n}\b", t) for n in
                   (2, 3, 5, 7, 11, 13, 17, 19)),
     "sequence: first 8 primes"),
    ("Q: A train travels 120 km in 1.5 hours. What is its average speed in km/h?\n"
     "A: The average speed is",
     lambda t: "80" in t,
     "arithmetic: 120/1.5=80"),
    ("Q: What is the chemical symbol for gold?\nA: The symbol is",
     lambda t: re.search(r"\bau\b", t) is not None,
     "factual: Au"),
    ("Q: In which year did the Apollo 11 mission first land humans on the Moon?\nA:",
     lambda t: "1969" in t,
     "factual: 1969"),
    ("Q: Translate exactly this English word to French: 'water'.\nA: The French word is",
     lambda t: "eau" in t,
     "translation: eau"),
    ("Q: What is 2 to the power of 10?\nA: It equals",
     lambda t: "1024" in t or "1,024" in t,
     "arithmetic: 2^10=1024"),
    ("Q: Complete the sequence: 1, 1, 2, 3, 5, 8, 13, ?\nA: The next number is",
     lambda t: "21" in t,
     "sequence: fibonacci"),
    ("Q: How many sides does a hexagon have?\nA: A hexagon has",
     lambda t: "6" in t or "six" in t,
     "factual: 6 sides"),
]


def generate(base, model, prompt, max_tokens, seed):
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "seed": seed}
    req = urllib.request.Request(
        f"{base}/v1/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["text"]


def run(base, model, trials, max_tokens):
    results = {}
    for prompt, pred, desc in CHECKS:
        passes = 0
        samples = []
        for i in range(trials):
            txt = generate(base, model, prompt, max_tokens, 20260922 + i)
            ok = bool(pred(txt.lower()))
            passes += ok
            samples.append(txt[:160])
        rate = passes / trials
        results[desc] = {"passes": passes, "trials": trials, "rate": rate,
                         "samples": samples}
        flag = "ok  " if rate == 1.0 else ("WARN" if rate >= 0.5 else "FAIL")
        print(f"  [{flag}] {desc:32s} {passes}/{trials}")
    overall = sum(r["passes"] for r in results.values())
    total = sum(r["trials"] for r in results.values())
    print(f"  overall: {overall}/{total} = {overall / total:.1%}")
    return {"checks": results, "overall_rate": overall / total,
            "overall_passes": overall, "overall_trials": total}


def compare(ref_path, new_path):
    ref = json.load(open(ref_path))
    new = json.load(open(new_path))
    print(f"reference : {ref['overall_passes']}/{ref['overall_trials']} "
          f"= {ref['overall_rate']:.1%}")
    print(f"new build : {new['overall_passes']}/{new['overall_trials']} "
          f"= {new['overall_rate']:.1%}")
    print()
    regressed = []
    for desc, r in ref["checks"].items():
        n = new["checks"].get(desc)
        if n is None:
            print(f"  missing in new: {desc}")
            continue
        if n["rate"] < r["rate"]:
            regressed.append((desc, r["rate"], n["rate"]))
            print(f"  REGRESSED {desc:32s} {r['rate']:.0%} -> {n['rate']:.0%}")
            print(f"    new sample: {n['samples'][0]!r}")
    delta = new["overall_rate"] - ref["overall_rate"]
    print()
    if regressed or delta < -0.05:
        print(f"GATE FAIL: {len(regressed)} check(s) regressed, "
              f"overall {delta:+.1%}")
        return 1
    print(f"GATE PASS: no per-check regressions, overall {delta:+.1%}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host")
    # The disagg proxy (vLLM's toy_proxy_server.py) exposes only
    # /v1/completions, /v1/chat/completions and /healthcheck -- there is no
    # /v1/models to auto-discover the id from, so it must be supplied. A plain
    # vLLM server still auto-discovers; this is only needed behind the proxy.
    ap.add_argument("--model", default=None,
                    help="model id; required when the endpoint has no "
                         "/v1/models (e.g. the disaggregated-prefill proxy)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--trials", type=int, default=3,
                    help="samples per check; averages over batching noise")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("REF", "NEW"))
    a = ap.parse_args()

    if a.compare:
        sys.exit(compare(*a.compare))

    if not a.host or not a.out:
        ap.error("--host and --out are required unless --compare is given")

    base = f"http://{a.host}:{a.port}"
    if a.model:
        model = a.model
    else:
        with urllib.request.urlopen(f"{base}/v1/models", timeout=60) as r:
            model = json.load(r)["data"][-1]["id"]
    print(f"probing {base} ({model}), {a.trials} trials/check")
    res = run(base, model, a.trials, a.max_tokens)
    res["host"] = a.host
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
