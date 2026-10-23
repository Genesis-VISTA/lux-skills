"""Run the ICF 4-agent app headless: one campaign or a laser-energy sweep, optional movie, PI debrief.

Drives app.handle_request, the same entry point the Gradio UI calls, with requests whose route is
fixed by the app's keyword rules (ICF_ROUTER=keywords), so a batch run does exactly what was asked:

    python run_campaign.py --out DIR --energy 3 [--movie] [--debrief]
    python run_campaign.py --out DIR --sweep 2:4:1 [--movie] [--debrief]

The endpoint and every other setting come from the ICF_* environment variables app.py reads
(ICF_BASE_URL, ICF_MAX_TOKENS, ICF_REASONING_EFFORT, ICF_MAX_DELTA, ...). Writes to DIR:
    summary.txt      the result in a few lines (robust and nominal yield, best design, timings)
    campaign.json    the session record: every simulation's design, defects and yield
    transcript.md    the agents' conversation, request by request
    debrief.md       the PI's written analysis of the session (--debrief)
    yield_vs_laser.png (sweep), time_histories.png, implosion.mp4 and implosion.gif (--movie)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

os.environ["ICF_ROUTER"] = "keywords"      # deterministic routing; see the docstring
sys.path.insert(0, os.environ.get("ICF_APP_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "icf_4agent"))
import app  # noqa: E402  (reads the ICF_* environment and the endpoint's model at import)


def _val(u):
    """The value of a gr.update(...) the handler yields, or None."""
    v = u.get("value") if isinstance(u, dict) else getattr(u, "value", None)
    return v if isinstance(v, str) else None


def ask(message, history, state, log):
    """One request through the app; returns (reply text, image path, video path, seconds)."""
    t0 = time.time()
    img = vid = None
    last = None
    for last in app.handle_request(message, history, state):
        img = _val(last[1]) or img
        vid = _val(last[2]) or vid
    chat = last[0]
    history[:] = chat
    reply = chat[-1]["content"] if chat and chat[-1]["role"] == "assistant" else ""
    secs = time.time() - t0
    log.write(f"\n\n# Request: {message}\n\n_{secs:.0f} s_\n\n{reply}\n")
    log.flush()
    print(f"[{secs:6.0f} s] {message}", flush=True)
    return reply, img, vid, secs


def _jsonable(state):
    out = []
    for c in state.get("campaigns", []):
        c = dict(c)
        c["results"] = [{"params": p, "yield_MJ": y} for p, y in c.get("results", [])]
        out.append(c)
    return {"campaigns": out, "media": state.get("media", [])}


def _gif(mp4, gif):
    """Animated GIF next to the MP4, for viewers that show images but not video."""
    ff = shutil.which("ffmpeg")
    if not ff:
        return None
    r = subprocess.run([ff, "-loglevel", "error", "-y", "-i", mp4, "-vf",
                        "fps=12,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
                        gif], capture_output=True, text=True)
    return gif if r.returncode == 0 and os.path.exists(gif) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--energy", type=float, help="laser energy in MJ for one campaign (default 3)")
    g.add_argument("--sweep", help="LO:HI:STEP in MJ, one campaign per energy, e.g. 2:4:1")
    ap.add_argument("--movie", action="store_true", help="render the best design's implosion movie and plots")
    ap.add_argument("--debrief", action="store_true", help="ask the PI to analyse the session")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.sweep:
        lo, hi, step = (float(x) for x in a.sweep.split(":"))
        request = f"sweep the laser energy from {lo:g} to {hi:g} MJ in steps of {step:g} MJ"
        energies = [e / 1e6 for e in app._sweep_energies(request)]
        what = f"sweep {lo:g}-{hi:g} MJ, {len(energies)} points: {', '.join(f'{e:g}' for e in energies)} MJ"
    else:
        e = a.energy if a.energy is not None else 3.0
        request = f"optimize an ICF capsule for a {e:g} MJ laser"
        what = f"single campaign at {e:g} MJ"
    # The keyword rules must route the request to the campaign it names, never to a chat answer.
    want = "sweep" if a.sweep else "single"
    if app._detect_mode(request) != want:
        sys.exit(f"internal error: {request!r} routes to {app._detect_mode(request)}, not {want}")

    cfg = app._cfg0
    head = [f"model {cfg['model']} at {cfg.get('base_url')}",
            f"run: {what}" + (", movie" if a.movie else "") + (", PI debrief" if a.debrief else ""),
            f"settings: max_tokens {cfg.get('max_tokens')}, reasoning_effort {cfg.get('reasoning_effort', 'server default')}, "
            f"temperature {app.config.get('temperature')}, defect budget delta<={app.DEFECT_CAPS['delta']:g} "
            f"roughness<={app.DEFECT_CAPS['roughness']:g} m"]
    print("\n".join(head), flush=True)

    state, history, timings, files = {"campaigns": [], "chat": []}, [], {}, {}
    t_all = time.time()
    with open(os.path.join(a.out, "transcript.md"), "w") as log:
        reply, img, _vid, timings["campaign"] = ask(request, history, state, log)
        if img:
            files["yield_vs_laser.png"] = img
        if a.movie:
            _r, img, vid, timings["movie"] = ask("show the implosion movie of the best design", history, state, log)
            if img:
                files["time_histories.png"] = img
            if vid:
                files["implosion.mp4"] = vid
        if a.debrief:
            state["chat"].append({"role": "user", "content":
                "Summarize this session for a report: the best robust design and its parameters, how much "
                "the adversary's defects cost it, how the result depends on laser energy if a sweep was run, "
                "and any caveats (few defect probes, energy-budget issues)."})
            t0 = time.time()
            answer = app.chat_with_pi(state)
            timings["debrief"] = time.time() - t0
            state["chat"].append({"role": "assistant", "content": answer})
            with open(os.path.join(a.out, "debrief.md"), "w") as f:
                f.write(answer + "\n")

    for name, src in files.items():
        shutil.copy(src, os.path.join(a.out, name))
    if "implosion.mp4" in files and _gif(os.path.join(a.out, "implosion.mp4"), os.path.join(a.out, "implosion.gif")):
        files["implosion.gif"] = "implosion.gif"
    with open(os.path.join(a.out, "campaign.json"), "w") as f:
        json.dump(_jsonable(state), f, indent=1)

    # summary.txt: what a reader needs without opening the transcript.
    c = state["campaigns"][0] if state["campaigns"] else {}
    lines = head + [""]
    bp = c.get("best_params")
    if c.get("sweep_points"):
        lines.append("laser MJ   no defects MJ   worst case MJ (robust)")
        for e_mj, clean, robust in c["sweep_points"]:
            lines.append(f"{e_mj:8.2f}   {clean:13.2f}   " + (f"{robust:.2f}" if robust is not None else "n/a"))
        lines.append(f"best robust design: {c.get('best_yield', 0):.2f} MJ at {c.get('best_e_laser', 0) / 1e6:g} MJ laser")
    elif c:
        g_ = app._group_by_design(c.get("results", [])).get(app._design_key(bp)) if bp else None
        lines.append(f"robust yield {c.get('best_yield', 0):.2f} MJ (worst case over "
                     f"{g_['n_probes'] if g_ else 0} defect probes); best single run "
                     f"{max(c.get('yields') or [0]):.2f} MJ")
    if bp:
        lines.append("best design: " + app._fmt_params({k: bp[k] for k in app.DESIGN_KEYS if k in bp}))
    nsim = len(c.get("results", []))
    lines.append(f"simulations: {nsim}; mean yield {c.get('mean_yield', 0):.2f} +- {c.get('std_yield', 0):.2f} MJ")
    lines.append("time: " + ", ".join(f"{k} {v:.0f} s" for k, v in timings.items()) + f"; total {time.time() - t_all:.0f} s")
    if files:
        lines.append("files: " + " ".join(sorted(os.path.basename(k) for k in files)))
    ok = bool(bp) and nsim > 0 and (not a.movie or "implosion.mp4" in files)
    lines.append("=== CAMPAIGN OK" if ok else "=== CAMPAIGN FAILED: " +
                 ("no simulation was recorded" if not nsim else "no design found" if not bp else "no movie"))
    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
