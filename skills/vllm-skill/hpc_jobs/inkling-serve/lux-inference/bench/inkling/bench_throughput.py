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

Prompts (--text):
  natural (default)  a distinct window of real prose per request, drawn from
                     natural_corpus.txt next to this script, followed by an
                     instruction to write an essay about it. What the model
                     generates is new text, so speculative decoding (MTP) is
                     measured at the acceptance rate it gets on real work.
  repeat             the old prompt: one sentence repeated. Generation just
                     continues the repetition, which the MTP drafter predicts
                     almost perfectly (acceptance 3.98 of 4 at k=3 and 8.09 of 9
                     at k=8 on tioga, 2026-10-06, against 2.5-3.9 on real
                     questions), so MTP numbers from it are upper bounds. Fine
                     for non-speculative runs, where decode speed does not
                     depend on the text.

The prompt goes through /v1/chat/completions by default (--api chat), so the
model's chat template frames it as a request. Sent as raw text (--api
completions), a chat-tuned Inkling answered the trailing "Essay:" cue with
"Essay:\nEssay:\n..." for 1k tokens in 99.8% of greedy outputs and 13% of
temperature-0.7 ones (tioga, 2026-10-07), which the MTP drafter predicts for
free. --text-out keeps the generated texts so this can be checked.

natural_corpus.txt is public-domain prose taken from Project Gutenberg
(Darwin, On the Origin of Species, 6th ed.; Thoreau, Walden; Mill, On Liberty;
The Federalist Papers; Smith, The Wealth of Nations), about 150 kB of body
paragraphs from each, with the Project Gutenberg headers removed.

Usage:
  python3 bench_throughput.py --host <head-node-ip> --port 8000 \
      --concurrency 32 --prompt-tokens 8000 --max-tokens 1024
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request
import zlib


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


def make_prompt(tokenish, salt=None):
    """
    Build a prompt of roughly `tokenish` tokens.

    `salt` prepends a unique run of digits. This is not decoration: vLLM
    enables prefix caching by default, and it matches on the *prefix*, so a
    varied word list does nothing when every request sends the same varied
    words. Handing all N threads one identical string means request 0 pays
    the prefill and requests 1..N-1 hit cache -- measured here at 91-98% hit
    rate on a 99k-token prompt, which collapsed TTFT to 0.31 s against a true
    cold cost of 4.67 s and inflated "total throughput" by the same factor.
    Salting per request restores a cold prefill for each one. Pass salt=None
    only when you deliberately want the cache-hot number.

    The token count is approximate and runs ~20% over the estimate here (the
    tokenizer it was calibrated on gives ~25 tokens per repetition, not 20). Calibrate against
    the server's /tokenize endpoint when the prompt has to land under a hard
    max_model_len -- the request is rejected with a bare HTTP 400 otherwise.
    """
    base = ("The quick brown fox jumps over the lazy dog near the riverbank "
            "while seventeen curious ravens observe from the ancient oak. ")
    # ~20 tokens per repetition of `base`
    reps = max(1, tokenish // 20)
    body = base * reps
    if salt is None:
        return body
    return f"{salt} {body}"


# Words of natural_corpus.txt, loaded on first use.
_CORPUS = None

# Older English prose runs about 1.3 tokens per word on the Inkling tokenizer
# (and on most BPE tokenizers). Approximate, like make_prompt's estimate; the
# timed run reports the real prompt token count.
TOKENS_PER_WORD = 1.3

ESSAY_INSTRUCTION = ("\n\nWrite a long, detailed and original essay that "
                     "discusses the passage above: its arguments, its evidence "
                     "and its historical context.")
# the raw-completions form ends on a cue for the essay to start; through the
# chat API the model's own template does that
COMPLETIONS_CUE = "\n\nEssay:\n"


def make_natural_prompt(tokenish, corpus_path, salt, cue=""):
    """
    Build a ~`tokenish`-token prompt from real prose: a window of
    natural_corpus.txt followed by an essay instruction.

    The window's start comes from a hash of `salt`, so every request reads a
    different passage. The salt is also prepended, so as in make_prompt no two
    requests share a prefix and every one pays its own prefill. A window longer
    than the corpus wraps around.
    """
    global _CORPUS
    if _CORPUS is None:
        with open(corpus_path, encoding="utf-8") as fh:
            _CORPUS = fh.read().split()
    nwords = max(1, int(tokenish / TOKENS_PER_WORD))
    start = zlib.crc32(str(salt).encode()) % len(_CORPUS)
    idx = range(start, start + nwords)
    passage = " ".join(_CORPUS[i % len(_CORPUS)] for i in idx)
    return f"{salt} {passage}{ESSAY_INSTRUCTION}{cue}"


def repeat_fraction(text, n=8):
    """
    Fraction of the word n-grams in `text` that already occurred earlier in it:
    0 for text that never repeats itself, approaching 1 for a loop.

    Greedy decoding with ignore_eos sometimes runs past the natural end of the
    answer into a repeating loop, and MTP drafts a loop almost perfectly. This
    tells such a run apart from one whose acceptance came from real prose.
    """
    words = text.split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return None
    return 1.0 - len(set(grams)) / len(grams)


def choice_text(choice):
    """Text of one choice: `text` from /v1/completions; from /v1/chat/completions
    the delta (streamed) or message, reasoning included, since a reasoning
    model's thinking is decoded tokens like any other."""
    if "text" in choice:
        return choice.get("text") or ""
    m = choice.get("delta") or choice.get("message") or {}
    return "".join(m.get(k) or "" for k in ("reasoning_content", "reasoning",
                                            "content"))


class Result:
    __slots__ = ("ok", "ttft", "elapsed", "out_tokens", "prompt_tokens", "err",
                 "text")

    def __init__(self):
        self.ok = False
        self.ttft = None
        self.elapsed = 0.0
        self.out_tokens = 0
        self.prompt_tokens = 0
        self.err = None
        self.text = ""


def one_request(base, model, prompt, max_tokens, stream, idx, results, barrier,
                temperature=0.0, seed=None, api="chat"):
    r = Result()
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        # greedy (by default) + ignore_eos so every request generates exactly
        # max_tokens. Without ignore_eos the model may stop early and the
        # throughput number becomes a function of how chatty it feels, not of
        # the hardware.
        "temperature": temperature,
        "stream": stream,
        "ignore_eos": True,
    }
    if temperature > 0:
        # a fixed seed per request keeps sampled runs repeatable
        payload["seed"] = idx if seed is None else seed
    if api == "chat":
        payload["messages"] = [{"role": "user", "content": prompt}]
    else:
        payload["prompt"] = prompt
    if stream:
        payload["stream_options"] = {"include_usage": True}

    barrier.wait()          # release all threads together
    t0 = time.perf_counter()
    try:
        resp = post(f"{base}/v1/{'chat/completions' if api == 'chat' else 'completions'}",
                    payload)
        finish = None
        if stream:
            ntok = 0
            parts = []
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
                if ch and ch[0].get("finish_reason"):
                    finish = ch[0]["finish_reason"]
                piece = ch and choice_text(ch[0])
                if piece:
                    if r.ttft is None:
                        r.ttft = time.perf_counter() - t0
                    ntok += 1
                    parts.append(piece)
            if not r.out_tokens:
                r.out_tokens = ntok
            r.text = "".join(parts)
        else:
            data = json.load(resp)
            c0 = (data.get("choices") or [{}])[0]
            r.text = choice_text(c0)
            finish = c0.get("finish_reason")
            u = data.get("usage", {})
            r.prompt_tokens = u.get("prompt_tokens", 0)
            r.out_tokens = u.get("completion_tokens", 0)
        r.elapsed = time.perf_counter() - t0
        # HTTP 200 is not success: vLLM ends a request whose KV load failed
        # (disagg) with finish_reason "error" and no tokens.
        if finish == "error":
            r.err = f"finish_reason=error after {r.out_tokens} tokens"
        else:
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
    # 8k in / 1k out -- an 8:1 prompt:output ratio. This is the house standard
    # as of 2026-09-22 and it is chosen to match the shape SemiAnalysis's
    # InferenceX suite quotes cost-per-token at ("8k in / 1k out"), so our
    # numbers are directly comparable to their published MI355X / B200 / GB300
    # curves. It replaces the old 8000/256 default (38:1), which was so
    # prefill-heavy that per-user decode rate mostly measured queueing delay.
    ap.add_argument("--prompt-tokens", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--text", choices=("natural", "repeat"), default="natural",
                    help="prompt text: distinct windows of real prose "
                         "(default) or the old repeated sentence; see the "
                         "module docstring")
    ap.add_argument("--corpus",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "natural_corpus.txt"),
                    help="prose for --text natural")
    ap.add_argument("--api", choices=("chat", "completions"), default="chat",
                    help="chat (default): the prompt is a user message and the "
                         "server applies the model's chat template. "
                         "completions: raw text, which a chat-tuned model can "
                         "answer by looping on the prompt's last line")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="sampling temperature (default 0.7, seeded per "
                         "request). 0 is greedy, which can loop in a long "
                         "output and so flatters MTP; see repeat_fraction()")
    ap.add_argument("--rounds", type=int, default=1,
                    help="timed bursts of --concurrency requests, run one "
                         "after another and pooled. At low concurrency one "
                         "burst is one or two passages, and MTP acceptance "
                         "swings with the passage")
    ap.add_argument("--no-stream", action="store_true",
                    help="disable streaming (TTFT will not be measured)")
    ap.add_argument("--warmup", type=int, default=2,
                    help="warmup requests before the timed run")
    ap.add_argument("--label", default="", help="tag for the results line")
    ap.add_argument("--json-out", default="", help="append a JSON result line here")
    ap.add_argument("--text-out", default="",
                    help="append every generated text here, one JSON line per "
                         "request, to read what the model actually wrote")
    ap.add_argument("--shared-prefix", action="store_true",
                    help="send one identical prompt to every request, letting "
                         "vLLM's prefix cache serve all but the first prefill. "
                         "Measures the cache-hot case; off by default because "
                         "it makes TTFT and total throughput look far better "
                         "than a cold workload would")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    model = get_model(base)
    stream = not args.no_stream

    # Distinct salt per request -> distinct prefix -> each request actually
    # prefills. The warmup pool gets its own salts so it cannot seed the cache
    # for the timed run either. Salts are fixed, not random, so a rerun of the
    # same command is comparable -- but a rerun against a *still-warm server*
    # would hit cache on these same prefixes, which is why --label is mixed
    # into the salt below rather than merely tagging the output row.
    #
    # This is not hypothetical. Measured on job 46655 (rebased main): a c=64
    # run that was interrupted and re-run against the same live server
    # reported 1,818.8 out tok/s at TTFT 1.23 s, against 963.9 tok/s at TTFT
    # 20.66 s for the genuinely cold first attempt -- a fictitious +20% over
    # baseline, produced entirely by the second run replaying the first run's
    # prefixes. A sub-2-second TTFT at 8k prompts is the tell.
    #
    # So: give every run a distinct --label. Identical labels against a warm
    # server measure the prefix cache, not the model.
    def prompt_for(tag, idx):
        if args.text == "natural":
            # --shared-prefix: one fixed salt, so every request sends the same
            # passage and the prefix cache serves all but the first prefill
            salt = "shared" if args.shared_prefix else \
                f"{9173 + idx * 7919:08d}{tag}{args.label}"
            return make_natural_prompt(args.prompt_tokens, args.corpus, salt,
                                       COMPLETIONS_CUE if args.api == "completions" else "")
        if args.shared_prefix:
            return make_prompt(args.prompt_tokens)
        return make_prompt(args.prompt_tokens,
                           salt=f"{9173 + idx * 7919:08d}{tag}{args.label}")

    print(f"endpoint     : {base}")
    print(f"model        : {model}")
    print(f"concurrency  : {args.concurrency}")
    print(f"prompt       : ~{args.prompt_tokens} tokens, {args.text} text, "
          f"{args.api} API")
    print(f"max_tokens   : {args.max_tokens}, temperature {args.temperature}")

    # --- warmup: compile triton/hip kernels, populate graphs ---
    if args.warmup:
        print(f"warmup       : {args.warmup} requests ...", flush=True)
        wres = [None] * args.warmup
        wb = threading.Barrier(args.warmup)
        wt = [threading.Thread(target=one_request,
                               args=(base, model, prompt_for("w", i), 32,
                                     stream, i, wres, wb),
                               kwargs={"temperature": args.temperature,
                                       "api": args.api})
              for i in range(args.warmup)]
        for t in wt:
            t.start()
        for t in wt:
            t.join()
        bad = [w for w in wres if w and not w.ok]
        if bad:
            print(f"warmup errors: {bad[0].err}")

    # --- timed run: --rounds bursts of n, one after another ---
    n = args.concurrency
    results = []
    wall = 0.0
    print(f"running      : timed run, {args.rounds} round(s) ...", flush=True)
    for rnd in range(args.rounds):
        part = [None] * n
        barrier = threading.Barrier(n)
        # round 0 keeps the salts a single-round run has always used
        tag = "t" if rnd == 0 else f"t{rnd}"
        threads = [threading.Thread(target=one_request,
                                    args=(base, model, prompt_for(tag, i),
                                          args.max_tokens,
                                          stream, i, part, barrier),
                                    kwargs={"temperature": args.temperature,
                                            "seed": rnd * n + i,
                                            "api": args.api})
                   for i in range(n)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall += time.perf_counter() - t0
        results += part

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
    # Steady-state per-user decode rate, TTFT excluded -- the interactivity
    # axis. per_req above divides by r.elapsed, which still includes the
    # request's own prefill wait; at a 38:1 prompt:output ratio that is most
    # of elapsed and the two differ by an order of magnitude.
    steady = [r.out_tokens / (r.elapsed - r.ttft) for r in ok
              if r.ttft is not None and r.elapsed > r.ttft]
    if steady:
        print(f"  INTERACTIVITY       : {statistics.mean(steady):.1f} tok/s/user"
              f"   (decode only, TTFT excluded)")
    if ttfts:
        # max, labelled as max. The old label said "p99" while printing
        # max(ttfts), which is p100 -- and at the concurrencies actually used
        # here (1..256) those are frequently the same sample anyway, so the
        # label was writing a check the sample size could not cash. A real p99
        # needs >=100 requests; below that, report the worst case and say so.
        print(f"  TTFT mean/p50/max   : {statistics.mean(ttfts):.2f} / "
              f"{statistics.median(ttfts):.2f} / {max(ttfts):.2f} s")
    reps = [f for f in (repeat_fraction(r.text) for r in ok) if f is not None]
    if reps:
        print(f"  repeated 8-grams    : {statistics.mean(reps):.1%} mean, "
              f"{max(reps):.1%} max   (output looping on itself)")
    print("=" * 62)
    if bad:
        print(f"  first error: {bad[0].err}")

    if args.json_out:
        rec = {
            "label": args.label, "model": model,
            "concurrency": n, "rounds": args.rounds,
            "prompt_tokens_req": args.prompt_tokens,
            "max_tokens": args.max_tokens, "text": args.text, "api": args.api,
            "temperature": args.temperature,
            "wall_s": round(wall, 3),
            "in_tokens": in_tok, "out_tokens": out_tok,
            "output_tok_s": round(out_tps, 1), "total_tok_s": round(tot_tps, 1),
            "ttft_mean_s": round(statistics.mean(ttfts), 3) if ttfts else None,
            "ttft_max_s": round(max(ttfts), 3) if ttfts else None,
            "shared_prefix": bool(args.shared_prefix),
            "ok": len(ok), "failed": len(bad),
            # see repeat_fraction(); high values mean MTP acceptance was
            # helped by the output looping, not earned on prose
            "repeat_frac_mean": round(statistics.mean(reps), 4) if reps else None,
            "repeat_frac_max": round(max(reps), 4) if reps else None,
            # Interactivity: the decode rate ONE user experiences. This is the
            # x-axis of a throughput-vs-interactivity curve, and it is not
            # derivable from the aggregate fields above -- output_tok_s/n would
            # charge every user for the whole run's wall time including other
            # users' queueing, which understates a fast request and overstates
            # a slow one. Measured per request from its own first token to its
            # own last, then averaged.
            #
            # Excluding TTFT is deliberate: this is steady-state decode, the
            # rate at which text appears once it starts appearing. TTFT is
            # reported separately above and is a latency, not a rate.
            "per_user_decode_tok_s": round(
                statistics.mean([r.out_tokens / (r.elapsed - r.ttft)
                                 for r in ok
                                 if r.ttft is not None and r.elapsed > r.ttft]), 2
            ) if ttfts else None,
        }
        with open(args.json_out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    if args.text_out:
        with open(args.text_out, "a") as fh:
            for i, r in enumerate(results):
                if r and r.ok:
                    fh.write(json.dumps({"label": args.label, "i": i,
                                         "repeat_frac": repeat_fraction(r.text),
                                         "text": r.text}) + "\n")


if __name__ == "__main__":
    main()
