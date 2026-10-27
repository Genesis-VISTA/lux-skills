# Builds summary.txt and results.json from one ringx-bench job's output folder.
#   python summarize.py <VISTA_OUT>
# Reads runs.tsv (one line per kernel run: point, kernel, rep, rc, log), the RESULT line in each run's log,
# correctness/*.log (+ .rc) and ceiling.txt. Python 3.6+, standard library only.
import json
import os
import re
import sys

OUT = sys.argv[1]
TOL = {"out diff": 0.02, "lse diff": 1e-3, "dq diff": 0.1, "dk diff": 0.1, "dv diff": 0.1}   # bf16, 8 GPUs
lines, fails = [], []


def say(s=""):
    lines.append(s)


def kv(line):
    return dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)


def why(log):
    t = open(log, errors="replace").read() if os.path.exists(log) else ""
    if re.search(r"out of memory|OutOfMemoryError", t):
        return "out of memory"
    if "CUDA driver error" in t:
        return "CUDA driver error (likely out of memory)"
    if "timed out" in t or "TIMEOUT" in t:
        return "timed out"
    m = re.findall(r"(\w+Error: .{0,120})", t)
    return m[-1] if m else "failed, see " + os.path.relpath(log, OUT)


def tflops(r):
    b, s, H, d, n, it, sec = (int(r["b"]), int(r["s"]), int(r["H"]), int(r["d"]), int(r["ngpus"]),
                              int(r["iters"]), float(r["sec"]))
    S, h, t = s * n, H * d, sec / it
    utils = 3 * (8 * b * S * h * h + 4 * b * S * S * h) / t / 1e12 / n          # RingX benchmark/utils.py
    real = 4 * b * S * S * H * d * 3.5 / (2 if r["causal"] == "1" else 1) / t / 1e12 / n
    return utils, real


# --- environment ---
env = os.path.join(OUT, "env.txt")
if os.path.exists(env):
    say(open(env).read().strip())
    say()

# --- correctness ---
cdir = os.path.join(OUT, "correctness")
tests = sorted(f[:-4] for f in os.listdir(cdir) if f.endswith(".log")) if os.path.isdir(cdir) else []
results = {"correctness": {}, "benchmark": [], "ceiling": []}
if tests:
    say("Correctness vs single-GPU flash_attn_func (max |diff| over ranks; tolerance in brackets)")
    for t in tests:
        log = os.path.join(cdir, t + ".log")
        rc = open(os.path.join(cdir, t + ".rc")).read().strip() if os.path.exists(os.path.join(cdir, t + ".rc")) else "?"
        mx, cur = {}, None
        for l in open(log, errors="replace"):
            m = re.match(r"(\w+ diff):", l)
            if m:
                cur = m.group(1)
                continue
            m = re.match(r"\[\d+\] max ([0-9.e+-]+)", l)
            if m and cur:
                mx[cur] = max(mx.get(cur, 0.0), float(m.group(1)))
        ok = rc == "0" and set(mx) == set(TOL) and all(mx[k] <= TOL[k] for k in TOL)
        detail = ", ".join(f"{k.split()[0]} {mx[k]:.3g}" for k in TOL if k in mx) or why(log)
        say(f"  {t:<14} {'PASS' if ok else 'FAIL'}  {detail}  (rc={rc})")
        results["correctness"][t] = {"pass": ok, "rc": rc, "max_diff": mx}
        if not ok:
            fails.append(f"correctness {t}")
    say("  [" + ", ".join(f"{k.split()[0]} {v:g}" for k, v in TOL.items()) + "]")
    say()

# --- single-GPU ceiling ---
ceil = {}
cfile = os.path.join(OUT, "ceiling.txt")
if os.path.exists(cfile):
    for l in open(cfile):
        if l.startswith("CEILING"):
            r = kv(l)
            key = (r["causal"], r["b"], r["s"], r["H"], r["d"])
            if "tflops" in r:
                ceil[key] = float(r["tflops"])
            results["ceiling"].append(r)

# --- benchmark ---
runs = []
tsv = os.path.join(OUT, "runs.tsv")
if os.path.exists(tsv):
    for l in open(tsv):
        point, kernel, rep, rc, log = l.rstrip("\n").split("\t")
        r = None
        if os.path.exists(log):
            for x in open(log, errors="replace"):
                if x.startswith("RESULT "):
                    r = kv(x)
        runs.append((point, kernel, rep, rc, log, r))

if runs:
    say("Benchmark, fwd+bwd, bf16 (utils = RingX benchmark/utils.py TFLOPS/GPU; real = attention FLOPs only,")
    say("halved for causal; % = real / single-GPU flash_attn_func at the same per-GPU shape)")
    for point in dict.fromkeys(p for p, *_ in runs):
        mode, b, s = point.split(":")
        say(f"  {mode}, batch {b}, {s} tokens/GPU")
        best = {}
        for p, kernel, rep, rc, log, r in runs:
            if p != point:
                continue
            if rc == "0" and r:
                u, real = tflops(r)
                c = ceil.get((r["causal"], r["b"], r["s"], r["H"], r["d"]))
                pct = f"{100 * real / c:4.0f}%" if c else "   - "
                say(f"    {kernel:<8} {float(r['sec']):8.3f} s/{r['iters']} it  utils {u:6.1f}  real {real:6.1f}  {pct}"
                    f"  peak {r['peak_mem_gib']} GiB" + (f"  (run {rep})" if rep != "1" else ""))
                results["benchmark"].append(dict(r, point=point, rep=int(rep), ok=True, utils_tflops=round(u, 1),
                                                 real_tflops=round(real, 1)))
                if kernel not in best or float(r["sec"]) < best[kernel]:
                    best[kernel] = float(r["sec"])
            else:
                w = why(log)
                say(f"    {kernel:<8} FAILED: {w} (rc={rc})" + (f"  (run {rep})" if rep != "1" else ""))
                results["benchmark"].append(dict(point=point, kernel=kernel, rep=int(rep), ok=False, rc=rc, error=w))
        if best:
            k = min(best, key=best.get)
            base = {x: best[x] for x in best if not x.startswith("ringX")}
            vs = f", {min(base.values()) / best[k]:.2f}x the best baseline" if base and k.startswith("ringX") else ""
            say(f"    fastest: {k}{vs}")
        c = [v for (cz, cb, cs, _, _), v in ceil.items() if (cz, cb, cs) == ("1" if mode == "causal" else "0", b, s)]
        if c:
            c = c[0]
            say(f"    single-GPU flash_attn_func at b={b}, s={s}: {c:.1f} real TFLOPS")
    say()
    n_bad = sum(1 for *_, rc, _, r in runs if rc != "0" or not r)
    if n_bad:
        say(f"{n_bad} of {len(runs)} kernel runs failed (listed above).")
    if n_bad == len(runs):
        fails.append("no kernel run succeeded")

say(f"=== BENCH FAILED: {'; '.join(fails)}" if fails else "=== BENCH OK")
open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(lines) + "\n")
json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1)
print("\n".join(lines))
sys.exit(1 if fails else 0)
