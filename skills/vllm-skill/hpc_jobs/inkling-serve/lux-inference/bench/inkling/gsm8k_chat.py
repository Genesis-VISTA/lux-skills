#!/usr/bin/env python3
"""Zero-shot gsm8k accuracy against an OpenAI-compatible chat endpoint (stdlib only).

usage: gsm8k_chat.py --host H --port P [--n 250] [--concurrency 64] [--effort none] --out OUT.json
Score = exact match of the last number in the reply vs the '#### N' gold answer.
"""
import argparse, json, os, re, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="localhost")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--n", type=int, default=250)
ap.add_argument("--concurrency", type=int, default=64)
ap.add_argument("--max-tokens", type=int, default=2048)
ap.add_argument("--effort", default="none", help="chat_template reasoning_effort; '' to omit")
ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsm8k_test.jsonl"))
ap.add_argument("--out", required=True)
a = ap.parse_args()

base = f"http://{a.host}:{a.port}"
model = json.load(urllib.request.urlopen(f"{base}/v1/models", timeout=60))["data"][0]["id"]
rows = [json.loads(l) for l in open(a.data)][: a.n]
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def norm(s):
    s = s.replace(",", "").rstrip(".")
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return s


def one(i):
    r = rows[i]
    body = {"model": model, "temperature": 0.0, "max_tokens": a.max_tokens,
            "messages": [{"role": "user", "content": r["question"] +
                          "\nSolve this step by step, then end with 'The answer is N.' where N is the final number."}]}
    if a.effort:
        body["chat_template_kwargs"] = {"reasoning_effort": a.effort}
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        c = json.load(urllib.request.urlopen(req, timeout=1800))["choices"][0]
        text = c["message"].get("content") or ""
        fr = c.get("finish_reason")
    except Exception as e:  # noqa: BLE001
        text, fr = f"ERROR {type(e).__name__}: {e}", "error"
    m = re.search(r"answer is\s*\$?\s*(-?\d[\d,]*\.?\d*)", text, re.I)
    nums = NUM.findall(text)
    pred = norm(m.group(1)) if m else (norm(nums[-1]) if nums else None)
    gold = norm(r["answer"].split("####")[-1].strip())
    return {"i": i, "gold": gold, "pred": pred, "ok": pred == gold, "finish": fr,
            "secs": round(time.time() - t0, 2), "text": text}


t0 = time.time()
with ThreadPoolExecutor(a.concurrency) as ex:
    res = list(ex.map(one, range(len(rows))))
wall = time.time() - t0
acc = sum(r["ok"] for r in res) / len(res)
summ = {"model": model, "n": len(res), "accuracy": round(acc, 4), "wall_s": round(wall, 1),
        "errors": sum(r["finish"] == "error" for r in res),
        "truncated": sum(r["finish"] == "length" for r in res), "effort": a.effort}
print(json.dumps(summ))
json.dump({"summary": summ, "results": res}, open(a.out, "w"), indent=1)
