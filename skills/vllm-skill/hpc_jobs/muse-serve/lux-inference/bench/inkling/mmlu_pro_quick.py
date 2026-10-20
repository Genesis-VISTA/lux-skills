#!/usr/bin/env python3
"""Quick MMLU-Pro accuracy check against an OpenAI-compatible chat endpoint (stdlib only).

usage: mmlu_pro_quick.py --host H --port P --out OUT.json [--concurrency 64] [--effort E]

Runs mmlu_pro_subset.jsonl: 280 questions, 20 from each of MMLU-Pro's 14
categories, drawn from the TIGER-Lab/MMLU-Pro test split (MIT licence) with a
fixed seed (20261011). The full test split is 12,032 questions; this subset is
meant to finish in minutes and catch a broken deployment, not to reproduce a
leaderboard number. 280 questions give a 95% interval of about +/-5 points.

Prompting is zero-shot chain of thought with MMLU-Pro's own instruction
("... finish your answer with 'the answer is (X)'"), temperature 0. Reasoning
is left on: the request omits reasoning_effort unless --effort is given, so
the chat template's default applies (0.9 for Inkling). With the server's
reasoning parser on, the thinking comes back in `reasoning` and the answer is
read from `content`; without it, both are in `content` and the last
"answer is (X)" in the text is used. A reply cut off by --max-tokens while
still thinking has no `content`; it is scored from its reasoning, as it would
be without the parser, so both server setups score the same. Extraction follows the official
evaluate_from_api.py: "answer is (X)", then "Answer: X", then the last
standalone letter A-J. A reply with no letter is scored wrong.

Output: overall accuracy with a Wilson 95% interval, per-category accuracy,
truncation/error counts, and token usage, plus every reply in OUT.json.
"""
import argparse, collections, json, math, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="localhost")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--n", type=int, default=0, help="first N questions only (0 = all 280)")
ap.add_argument("--concurrency", type=int, default=64)
ap.add_argument("--max-tokens", type=int, default=16384)
ap.add_argument("--effort", default="", help="chat_template reasoning_effort (e.g. none, 0.5); omitted if empty")
ap.add_argument("--timeout", type=float, default=3600)
ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "mmlu_pro_subset.jsonl"))
ap.add_argument("--out", required=True)
a = ap.parse_args()

base = f"http://{a.host}:{a.port}"
model = json.load(urllib.request.urlopen(f"{base}/v1/models", timeout=60))["data"][0]["id"]
rows = [json.loads(l) for l in open(a.data)]
if a.n:
    rows = rows[: a.n]
LETTERS = "ABCDEFGHIJ"


def prompt(r):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(r["options"]))
    return (f"The following is a multiple choice question about {r['category']}. Think step by "
            f"step and then finish your answer with \"the answer is (X)\" where X is the correct "
            f"letter choice.\n\nQuestion: {r['question']}\nOptions:\n{opts}\n")


def extract(text):
    m = re.findall(r"answer is \(?([A-J])\)?", text)
    if m:
        return m[-1]
    m = re.findall(r"[aA]nswer:\s*\(?([A-J])\)?", text)
    if m:
        return m[-1]
    m = re.findall(r"\b([A-J])\b(?!.*\b[A-J]\b)", text, re.S)
    return m[-1] if m else None


def one(i):
    r = rows[i]
    body = {"model": model, "temperature": 0.0, "max_tokens": a.max_tokens,
            "messages": [{"role": "user", "content": prompt(r)}]}
    if a.effort:
        body["chat_template_kwargs"] = {"reasoning_effort": a.effort}
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    reasoning, usage = "", {}
    try:
        d = json.load(urllib.request.urlopen(req, timeout=a.timeout))
        c = d["choices"][0]
        text = c["message"].get("content") or ""
        reasoning = c["message"].get("reasoning") or c["message"].get("reasoning_content") or ""
        fr = c.get("finish_reason")
        usage = d.get("usage") or {}
    except Exception as e:  # noqa: BLE001
        text, fr = f"ERROR {type(e).__name__}: {e}", "error"
    pred = extract(text if text.strip() else reasoning) if fr != "error" else None
    return {"question_id": r["question_id"], "category": r["category"], "gold": r["answer"],
            "pred": pred, "ok": pred == r["answer"], "finish": fr, "secs": round(time.time() - t0, 2),
            "completion_tokens": usage.get("completion_tokens"), "reasoning_chars": len(reasoning),
            "text": text, "reasoning": reasoning}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


t0 = time.time()
with ThreadPoolExecutor(a.concurrency) as ex:
    res = list(ex.map(one, range(len(rows))))
wall = time.time() - t0

k = sum(r["ok"] for r in res)
cats = collections.defaultdict(list)
for r in res:
    cats[r["category"]].append(r["ok"])
toks = [r["completion_tokens"] for r in res if r["completion_tokens"]]
summ = {"model": model, "n": len(res), "accuracy": round(k / len(res), 4), "ci95": wilson(k, len(res)),
        "per_category": {c: round(sum(v) / len(v), 3) for c, v in sorted(cats.items())},
        "no_answer": sum(r["pred"] is None for r in res), "truncated": sum(r["finish"] == "length" for r in res),
        "errors": sum(r["finish"] == "error" for r in res),
        "reasoning_separated": sum(r["reasoning_chars"] > 0 for r in res),
        "completion_tokens_mean": round(sum(toks) / len(toks)) if toks else None,
        "completion_tokens_total": sum(toks), "wall_s": round(wall, 1),
        "effort": a.effort or "template default", "max_tokens": a.max_tokens}
print(f"MMLU-Pro subset: {k}/{len(res)} = {summ['accuracy']:.3f}  95% CI {summ['ci95']}  "
      f"({summ['wall_s']} s, mean {summ['completion_tokens_mean']} completion tokens, "
      f"{summ['truncated']} truncated, {summ['errors']} errors, {summ['no_answer']} no answer)")
for c, v in summ["per_category"].items():
    print(f"  {c:18s} {v:.3f}")
print(json.dumps(summ))
json.dump({"summary": summ, "results": res}, open(a.out, "w"), indent=1)
