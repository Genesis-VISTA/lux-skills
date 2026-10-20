#!/usr/bin/env python3
"""
Throughput benchmark against a vLLM OpenAI-compatible endpoint.

client_query.py is an interactive single-prompt tool -- useful for eyeballing
output quality, but it streams one request and reports no timing, so it cannot
answer "how many tokens/sec". This driver issues N concurrent completions and
reports both decode throughput and end-to-end throughput.

Reported metrics:
  output tok/s (decode)  -- generated tokens / wall time. The headline number.
  total  tok/s           -- (prompt + generated) / wall time
  TTFT                   -- time to first token, per request (streaming mode)
  per-request tok/s      -- decode rate seen by a single client

Usage:
  python3 bench_throughput.py --host <head-node-ip> --port 8000 \
      --concurrency 32 --prompt-tokens 8000 --max-tokens 256
"""

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request


def post(url, payload, timeout=3600):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer EMPTY"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def get_model(base):
    with urllib.request.urlopen(f"{base}/v1/models", timeout=60) as r:
        data = json.load(r)
    return data["data"][-1]["id"]


def make_prompt(tokenish):
    """
    Build a prompt of roughly `tokenish` tokens. We use a varied word list so
    the tokenizer does not collapse it, and so prefix caching between concurrent
    requests does not make the prefill artificially free.
    """
    base = ("The quick brown fox jumps over the lazy dog near the riverbank "
            "while seventeen curious ravens observe from the ancient oak. ")
    # ~20 tokens per repetition of `base`
    reps = max(1, tokenish // 20)
    return base * reps


class Result:
    __slots__ = ("ok", "ttft", "elapsed", "out_tokens", "prompt_tokens", "err")

    def __init__(self):
        self.ok = False
        self.ttft = None
        self.elapsed = 0.0
        self.out_tokens = 0
        self.prompt_tokens = 0
        self.err = None


def one_request(base, model, prompt, max_tokens, stream, idx, results, barrier):
    r = Result()
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        # greedy + ignore_eos so every request generates exactly max_tokens.
        # Without ignore_eos the model may stop early and the throughput number
        # becomes a function of how chatty it feels, not of the hardware.
        "temperature": 0.0,
        "stream": stream,
        "ignore_eos": True,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}

    barrier.wait()          # release all threads together
    t0 = time.perf_counter()
    try:
        resp = post(f"{base}/v1/completions", payload)
        if stream:
            ntok = 0
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                chunk = json.loads(body)
                if chunk.get("usage"):
                    r.prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                    r.out_tokens = chunk["usage"].get("completion_tokens", ntok)
                ch = chunk.get("choices") or []
                if ch and ch[0].get("text"):
                    if r.ttft is None:
                        r.ttft = time.perf_counter() - t0
                    ntok += 1
            if not r.out_tokens:
                r.out_tokens = ntok
        else:
            data = json.load(resp)
            u = data.get("usage", {})
            r.prompt_tokens = u.get("prompt_tokens", 0)
            r.out_tokens = u.get("completion_tokens", 0)
        r.elapsed = time.perf_counter() - t0
        r.ok = True
    except Exception as exc:                                # noqa: BLE001
        r.elapsed = time.perf_counter() - t0
        r.err = f"{type(exc).__name__}: {exc}"
    results[idx] = r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--prompt-tokens", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--no-stream", action="store_true",
                    help="disable streaming (TTFT will not be measured)")
    ap.add_argument("--warmup", type=int, default=2,
                    help="warmup requests before the timed run")
    ap.add_argument("--label", default="", help="tag for the results line")
    ap.add_argument("--json-out", default="", help="append a JSON result line here")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    model = get_model(base)
    prompt = make_prompt(args.prompt_tokens)
    stream = not args.no_stream

    print(f"endpoint     : {base}")
    print(f"model        : {model}")
    print(f"concurrency  : {args.concurrency}")
    print(f"prompt       : ~{args.prompt_tokens} tokens")
    print(f"max_tokens   : {args.max_tokens}")

    # --- warmup: compile triton/hip kernels, populate graphs ---
    if args.warmup:
        print(f"warmup       : {args.warmup} requests ...", flush=True)
        wres = [None] * args.warmup
        wb = threading.Barrier(args.warmup)
        wt = [threading.Thread(target=one_request,
                               args=(base, model, prompt, 32, stream, i, wres, wb))
              for i in range(args.warmup)]
        for t in wt:
            t.start()
        for t in wt:
            t.join()
        bad = [w for w in wres if w and not w.ok]
        if bad:
            print(f"warmup errors: {bad[0].err}")

    # --- timed run ---
    n = args.concurrency
    results = [None] * n
    barrier = threading.Barrier(n)
    threads = [threading.Thread(target=one_request,
                                args=(base, model, prompt, args.max_tokens,
                                      stream, i, results, barrier))
               for i in range(n)]
    print("running      : timed run ...", flush=True)
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    ok = [r for r in results if r and r.ok]
    bad = [r for r in results if r and not r.ok]
    if not ok:
        print("ALL REQUESTS FAILED")
        if bad:
            print("first error:", bad[0].err)
        sys.exit(1)

    out_tok = sum(r.out_tokens for r in ok)
    in_tok = sum(r.prompt_tokens for r in ok)
    out_tps = out_tok / wall
    tot_tps = (out_tok + in_tok) / wall
    per_req = [r.out_tokens / r.elapsed for r in ok if r.elapsed > 0]
    ttfts = [r.ttft for r in ok if r.ttft is not None]

    print()
    print("=" * 62)
    if args.label:
        print(f"  {args.label}")
        print("-" * 62)
    print(f"  requests ok/failed  : {len(ok)}/{len(bad)}")
    print(f"  wall time           : {wall:.2f} s")
    print(f"  prompt tokens       : {in_tok}")
    print(f"  generated tokens    : {out_tok}")
    print(f"  OUTPUT THROUGHPUT   : {out_tps:.1f} tok/s   <-- decode")
    print(f"  TOTAL  THROUGHPUT   : {tot_tps:.1f} tok/s   (prompt+decode)")
    if per_req:
        print(f"  per-request decode  : {statistics.mean(per_req):.1f} tok/s "
              f"(median {statistics.median(per_req):.1f})")
    if ttfts:
        print(f"  TTFT mean/p50/p99   : {statistics.mean(ttfts):.2f} / "
              f"{statistics.median(ttfts):.2f} / {max(ttfts):.2f} s")
    print("=" * 62)
    if bad:
        print(f"  first error: {bad[0].err}")

    if args.json_out:
        rec = {
            "label": args.label, "model": model,
            "concurrency": n, "prompt_tokens_req": args.prompt_tokens,
            "max_tokens": args.max_tokens, "wall_s": round(wall, 3),
            "in_tokens": in_tok, "out_tokens": out_tok,
            "output_tok_s": round(out_tps, 1), "total_tok_s": round(tot_tps, 1),
            "ttft_mean_s": round(statistics.mean(ttfts), 3) if ttfts else None,
            "ok": len(ok), "failed": len(bad),
        }
        with open(args.json_out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
