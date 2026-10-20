#!/usr/bin/env python3
"""Science knowledge suite (physics, chemistry, materials science) against an OpenAI-compatible chat endpoint.

usage: science_suite.py --host H --port P --out OUT.json [--data-dir DIR] [--benchmarks gpqa,mascqa,mmlu_pro]
                        [--concurrency 64] [--max-tokens 8192] [--effort E] [--n N]

Three benchmarks, prepared by fetch_science_data.py into DIR (default $ZENITH_DATASETSPATH/science):

  GPQA Diamond, physics + chemistry   179 graduate-level questions written to resist look-up (4 options)
  MaScQA                              646 materials-science exam questions: multiple choice (some multiple-select),
                                      matching, and numerical answers scored against the exam's accepted range
  MMLU-Pro physics/chemistry/eng.     3,400 questions, 10 options

Prompting is zero-shot chain of thought at temperature 0, one request per question, each ending with an instruction
to finish on "The answer is ...". Reasoning models work too: as in bench/inkling/mmlu_pro_quick.py, a reply with no
`content` (cut off while thinking) is scored from its `reasoning`. Extraction takes the last "answer is"; failing
that, "Answer: X"; for multiple choice, failing both, the last standalone option letter. No answer scores wrong.

Scores: accuracy with a Wilson 95% interval per benchmark, broken down by domain / topic / question type, then per
field (physics = GPQA physics + MMLU-Pro physics, chemistry likewise, materials = MaScQA, engineering = MMLU-Pro
engineering), and a suite score: the unweighted mean of the three benchmark accuracies, so the 3,400 MMLU-Pro
questions do not swamp the other two. Every reply is kept in OUT.json for inspection. GPQA's authors ask that its
questions not be posted online, so treat OUT.json like the data: keep it off the web.
"""
import argparse, collections, json, math, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="localhost")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--data-dir", default=os.path.join(os.environ.get("ZENITH_DATASETSPATH", "."), "science"))
ap.add_argument("--benchmarks", default="gpqa,mascqa,mmlu_pro")
ap.add_argument("--gpqa-domains", default="physics,chemistry", help="'all' adds GPQA's 19 biology questions")
ap.add_argument("--n", type=int, default=0, help="first N questions of each benchmark only (smoke test)")
ap.add_argument("--concurrency", type=int, default=64)
ap.add_argument("--max-tokens", type=int, default=8192)
ap.add_argument("--effort", default="", help="chat_template reasoning_effort; omitted if empty")
ap.add_argument("--timeout", type=float, default=3600)
ap.add_argument("--out", required=True)
a = ap.parse_args()

base = f"http://{a.host}:{a.port}"
model = json.load(urllib.request.urlopen(f"{base}/v1/models", timeout=60))["data"][0]["id"]

FINISH_MC = ("Think step by step, then finish your answer with \"The answer is (X)\" where X is the letter of the "
             "correct option.")


def load(name):
    path = os.path.join(a.data_dir, name)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: run bench/science/fetch_science_data.py first")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return rows[: a.n] if a.n else rows


def options(opts):
    return "\n".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(opts))


# ---- per-benchmark items: (benchmark, id, groups, prompt, kind, gold) ----
def gpqa_items():
    doms = None if a.gpqa_domains == "all" else set(a.gpqa_domains.split(","))
    for r in load("gpqa_diamond.jsonl"):
        if doms and r["domain"] not in doms:
            continue
        p = f"Answer the following multiple choice question. {FINISH_MC}\n\n{r['question']}\n\n{options(r['options'])}\n"
        yield ("gpqa", r["id"], {"domain": r["domain"], "subdomain": r["subdomain"]}, p, "mc", [r["answer"]])


def mascqa_items():
    for r in load("mascqa.jsonl"):
        head = "The following is a materials science question from a graduate engineering entrance exam."
        if r["type"] == "NUM":
            p = (f"{head} The answer is a number. Think step by step, then finish your answer with \"The answer is "
                 f"N\" where N is the number alone, in the units and to the precision the question asks for.\n\n"
                 f"Question: {r['question']}\n")
            gold = r["ranges"]
        else:
            p = (f"{head} {FINISH_MC} If more than one option is correct, give every correct letter, for example "
                 f"\"The answer is (A, C)\".\n\nQuestion: {r['question']}\n")
            gold = r["choices"]
        yield ("mascqa", r["id"], {"topic": r["topic"], "type": r["type"]}, p,
               "num" if r["type"] == "NUM" else "mc", gold)


def mmlu_pro_items():
    for r in load("mmlu_pro_sci.jsonl"):
        opts = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(r["options"]))
        # MMLU-Pro's own zero-shot CoT instruction, as in bench/inkling/mmlu_pro_quick.py
        p = (f"The following is a multiple choice question about {r['category']}. Think step by step and then "
             f"finish your answer with \"the answer is (X)\" where X is the correct letter choice.\n\n"
             f"Question: {r['question']}\nOptions:\n{opts}\n")
        yield ("mmlu_pro", r["question_id"], {"category": r["category"]}, p, "mc", [r["answer"]])


# ---- answer extraction ----
LET = r"[A-J]"
SET_RE = re.compile(rf"(?i:answer is)[:\s]*\**\s*\(?\s*({LET}(?:\s*\)?\s*(?:,|and|&|/)\s*\(?\s*{LET})*)\b")


def extract_mc(text):
    m = SET_RE.findall(text)
    if not m:
        m = re.findall(rf"[aA]nswer:\s*\**\s*\(?\s*({LET}(?:\s*,\s*{LET})*)\b", text)
    if m:
        return "".join(sorted(set(re.findall(rf"\b{LET}\b", m[-1]))))
    m = re.findall(rf"\b({LET})\b(?!.*\b{LET}\b)", text, re.S)
    return m[-1] if m else None


NUM_RE = r"[-−]?\d[\d,]*(?:\.\d+)?|[-−]?\.\d+"


def to_float(tok, tail):
    v = float(tok.replace(",", "").replace("−", "-"))
    e = re.match(r"\s*(?:[eE]|\s*[x×\*]\s*10\s*\^?\s*\{?)\s*([-−+]?\d+)", tail)   # 1.2e-3, 1.2 x 10^-3, 1.2×10^{-3}
    return v * 10 ** int(e.group(1).replace("−", "-")) if e else v


def extract_num(text):
    text = text.replace("$", "").replace("\\times", "×").replace("\\,", "")
    for pat in (r"answer is[:\s]*\**\s*(?:approximately|about|~|≈)?\s*", r"\\boxed\{\s*", r"[aA]nswer:\s*\**\s*"):
        hits = list(re.finditer(pat + f"({NUM_RE})", text))
        if hits:
            h = hits[-1]
            return to_float(h.group(1), text[h.end():])
    hits = list(re.finditer(NUM_RE, text))
    return to_float(hits[-1].group(0), text[hits[-1].end():]) if hits else None


def correct(kind, pred, gold):
    if pred is None:
        return False
    if kind == "num":
        return any(lo - 1e-9 * abs(lo) <= pred <= hi + 1e-9 * abs(hi) for lo, hi in gold)
    return pred in gold


def one(item):
    bench, qid, groups, prompt, kind, gold = item
    body = {"model": model, "temperature": 0.0, "max_tokens": a.max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    if a.effort:
        body["chat_template_kwargs"] = {"reasoning_effort": a.effort}
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, reasoning, usage = time.time(), "", {}
    try:
        d = json.load(urllib.request.urlopen(req, timeout=a.timeout))
        c = d["choices"][0]
        text = c["message"].get("content") or ""
        reasoning = c["message"].get("reasoning") or c["message"].get("reasoning_content") or ""
        fr, usage = c.get("finish_reason"), d.get("usage") or {}
    except Exception as e:  # noqa: BLE001
        text, fr = f"ERROR {type(e).__name__}: {e}", "error"
    src = text if text.strip() else reasoning
    pred = None if fr == "error" else (extract_num(src) if kind == "num" else extract_mc(src))
    return {"benchmark": bench, "id": qid, **groups, "kind": kind, "gold": gold, "pred": pred,
            "ok": correct(kind, pred, gold), "finish": fr, "secs": round(time.time() - t0, 2),
            "completion_tokens": usage.get("completion_tokens"), "text": text, "reasoning": reasoning}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def acc(rs):
    k = sum(r["ok"] for r in rs)
    return {"n": len(rs), "correct": k, "accuracy": round(k / len(rs), 4) if rs else None, "ci95": wilson(k, len(rs))}


gens = {"gpqa": gpqa_items, "mascqa": mascqa_items, "mmlu_pro": mmlu_pro_items}
benches = a.benchmarks.split(",")
items = [it for b in benches for it in gens[b]()]
print(f"science suite: {model} at {base}, {len(items)} questions ({', '.join(benches)}), "
      f"concurrency {a.concurrency}, max_tokens {a.max_tokens}", flush=True)
t0 = time.time()
res, step = [], max(1, len(items) // 10)
with ThreadPoolExecutor(a.concurrency) as ex:
    for r in ex.map(one, items):
        res.append(r)
        if len(res) % step == 0:
            print(f"  {len(res)}/{len(items)} done, {time.time() - t0:.0f} s", flush=True)
wall = time.time() - t0

summ = {"model": model, "wall_s": round(wall, 1), "max_tokens": a.max_tokens, "effort": a.effort or "template default",
        "gpqa_domains": a.gpqa_domains, "benchmarks": {}, "fields": {}}
by = collections.defaultdict(list)
for r in res:
    by[r["benchmark"]].append(r)
GROUP = {"gpqa": ("domain", "subdomain"), "mascqa": ("type", "topic"), "mmlu_pro": ("category",)}
for b in benches:
    rs = by[b]
    s = acc(rs)
    s.update({"truncated": sum(r["finish"] == "length" for r in rs), "errors": sum(r["finish"] == "error" for r in rs),
              "no_answer": sum(r["pred"] is None for r in rs)})
    for g in GROUP[b]:
        groups = collections.defaultdict(list)
        for r in rs:
            groups[r[g]].append(r)
        s["by_" + g] = {k: acc(v) for k, v in sorted(groups.items())}
    summ["benchmarks"][b] = s
FIELDS = {"physics": lambda r: r.get("domain") == "physics" or r.get("category") == "physics",
          "chemistry": lambda r: r.get("domain") == "chemistry" or r.get("category") == "chemistry",
          "materials": lambda r: r["benchmark"] == "mascqa",
          "engineering": lambda r: r.get("category") == "engineering"}
for f, sel in FIELDS.items():
    rs = [r for r in res if sel(r)]
    if rs:
        summ["fields"][f] = acc(rs)
summ["suite_score"] = round(sum(summ["benchmarks"][b]["accuracy"] for b in benches) / len(benches), 4)

print(f"suite score {summ['suite_score']:.3f} (mean of {len(benches)} benchmark accuracies), {summ['wall_s']} s")
for b, s in summ["benchmarks"].items():
    print(f"  {b:9s} {s['correct']:4d}/{s['n']:<4d} = {s['accuracy']:.3f}  95% CI {s['ci95']}  "
          f"({s['truncated']} truncated, {s['errors']} errors, {s['no_answer']} no answer)")
    g = GROUP[b][0]
    for k, v in s["by_" + g].items():
        print(f"      {k:22s} {v['correct']:4d}/{v['n']:<4d} = {v['accuracy']:.3f}")
print("  by field:")
for f, v in summ["fields"].items():
    print(f"      {f:22s} {v['correct']:4d}/{v['n']:<4d} = {v['accuracy']:.3f}  95% CI {v['ci95']}")
print(json.dumps({k: summ[k] for k in ("model", "suite_score", "wall_s")} |
                 {b: summ["benchmarks"][b]["accuracy"] for b in benches}))
json.dump({"summary": summ, "results": res}, open(a.out, "w"), indent=1)
