#!/usr/bin/env python3
"""Long-context needle retrieval and single-request prefill/decode rates (stdlib only).

usage: long_context_check.py --host H --port P --out OUT.json
           [--lengths 32k,128k,256k,512k,1040k] [--depths 0.1,0.5,0.9] [--effort E]

For each target prompt length and needle depth, this sends ONE chat request
and streams the reply. The prompt is a haystack of natural_corpus.txt (public
domain prose, about 170k tokens) with one unique sentence, the needle, at the
given fraction of the way through. The question comes after the haystack. The
corpus is repeated, paragraph-shuffled with a fixed seed per pass, until the
target length is reached. The needle sentence appears nowhere else.

A cell passes if the needle's code appears in the reply's content, or in its
reasoning when there is no content. Requests run one at a time, so the timings
are single-request figures:

  ttft_s        time to the first streamed token (reasoning or content), i.e.
                the whole prefill plus one decode step
  prefill_tok_s prompt tokens / ttft
  decode_tok_s  completion tokens after the first / time after the first

Prompt sizes come from the server: /tokenize calibrates the chars-per-token
ratio, and each request's usage block gives the exact count. Lengths take a k
suffix (x1000). The default top length, 1,040,000, leaves room for
--max-tokens inside a 1,048,576 max_model_len.
"""
import argparse, json, os, random, re, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="localhost")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--lengths", default="32k,128k,256k,512k,1040k")
ap.add_argument("--depths", default="0.1,0.5,0.9")
ap.add_argument("--max-tokens", type=int, default=4096)
ap.add_argument("--effort", default="", help="chat_template reasoning_effort; omitted if empty")
ap.add_argument("--timeout", type=float, default=3600)
ap.add_argument("--corpus", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "natural_corpus.txt"))
ap.add_argument("--out", required=True)
a = ap.parse_args()

base = f"http://{a.host}:{a.port}"
model = json.load(urllib.request.urlopen(f"{base}/v1/models", timeout=60))["data"][0]["id"]
paras = [p.strip() for p in open(a.corpus, encoding="utf-8").read().split("\n") if p.strip()]


def post(path, body, timeout=600):
    req = urllib.request.Request(f"{base}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def n_tokens(text):
    return json.load(post("/tokenize", {"model": model, "prompt": text}))["count"]


def parse_len(s):
    s = s.strip().lower()
    return int(float(s[:-1]) * 1000) if s.endswith("k") else int(s)


sample = "\n\n".join(paras[:2000])
CHARS_PER_TOK = len(sample) / n_tokens(sample)
QUESTION = ("\n\n---\n\nSomewhere in the text above there is one sentence giving the access code for "
            "the Zenith archive. What is the access code? Reply with the code only.")


def haystack(n_chars, seed):
    out, size, k = [], 0, 0
    while size < n_chars:
        ps = paras[:]
        random.Random(seed * 1000 + k).shuffle(ps)
        for p in ps:
            out.append(p)
            size += len(p) + 2
            if size >= n_chars:
                break
        k += 1
    return out


def run(target, depth, idx):
    rng = random.Random(20261011 + idx)
    code = f"{rng.choice(['amber', 'cobalt', 'violet', 'saffron', 'indigo'])}-{rng.randint(100000, 999999)}"
    needle = f"The access code for the Zenith archive is {code}."
    body_chars = int((target - 64) * CHARS_PER_TOK)
    hs = haystack(body_chars, idx)
    pos = int(len(hs) * depth)
    hs.insert(pos, needle)
    content = "\n\n".join(hs) + QUESTION
    # the chars/token ratio is an average; trim from the far end of the
    # haystack until the exact count fits, so 1040k never overruns max_model_len
    for _ in range(5):
        over = n_tokens(content) - target
        if over <= 0:
            break
        drop = int(over * CHARS_PER_TOK * 1.1) + 1
        while drop > 0 and len(hs) - 1 > pos:
            drop -= len(hs.pop()) + 2
        content = "\n\n".join(hs) + QUESTION
    body = {"model": model, "temperature": 0.0, "max_tokens": a.max_tokens, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": content}]}
    if a.effort:
        body["chat_template_kwargs"] = {"reasoning_effort": a.effort}
    t0 = time.time()
    t_first, text, reasoning, usage, err = None, "", "", {}, None
    try:
        with post("/v1/chat/completions", body, timeout=a.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[5:])
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    piece_r = d.get("reasoning") or d.get("reasoning_content") or ""
                    piece_c = d.get("content") or ""
                    if (piece_r or piece_c) and t_first is None:
                        t_first = time.time()
                    reasoning += piece_r
                    text += piece_c
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    t_end = time.time()
    ptoks, ctoks = usage.get("prompt_tokens"), usage.get("completion_tokens")
    ttft = (t_first - t0) if t_first else None
    found = code in (text if text.strip() else reasoning)
    r = {"target": target, "depth": depth, "prompt_tokens": ptoks, "completion_tokens": ctoks,
         "pass": bool(found and not err), "code": code, "ttft_s": round(ttft, 2) if ttft else None,
         "prefill_tok_s": round(ptoks / ttft) if (ttft and ptoks) else None,
         "decode_tok_s": round((ctoks - 1) / (t_end - t_first), 1) if (t_first and ctoks and ctoks > 1 and t_end > t_first) else None,
         "total_s": round(t_end - t0, 1), "error": err,
         "answer": text.strip()[:200], "reasoning_chars": len(reasoning)}
    print(f"{target:>8} depth {depth:.2f}: {'PASS' if r['pass'] else 'FAIL'}  prompt {ptoks}  "
          f"ttft {r['ttft_s']} s  prefill {r['prefill_tok_s']} tok/s  decode {r['decode_tok_s']} tok/s  "
          f"answer {r['answer'][:60]!r}{'  ERR ' + err if err else ''}", flush=True)
    return r


lengths = [parse_len(s) for s in a.lengths.split(",")]
depths = [float(s) for s in a.depths.split(",")]
print(f"model {model}, {CHARS_PER_TOK:.2f} chars/token, lengths {lengths}, depths {depths}", flush=True)
res, idx = [], 0
for L in lengths:
    for d in depths:
        res.append(run(L, d, idx))
        idx += 1
        json.dump({"model": model, "chars_per_token": CHARS_PER_TOK, "results": res}, open(a.out, "w"), indent=1)
ok = sum(r["pass"] for r in res)
print(json.dumps({"model": model, "passed": ok, "cells": len(res),
                  "max_prompt_tokens_passed": max([r["prompt_tokens"] or 0 for r in res if r["pass"]] or [0])}))
