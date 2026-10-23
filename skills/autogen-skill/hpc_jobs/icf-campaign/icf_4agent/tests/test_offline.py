"""Offline checks of app.py's routing rules, RUN_SIMULATION parser and session record
(no LLM, no venv needed).

    python3 tests/test_offline.py

The functions are pulled out of app.py with `ast`, so this runs without importing autogen.
"""
import ast, os, re, sys

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app.py")
KEEP = {"SWEEP_KEYWORDS", "MEDIA_KEYWORDS", "RUN_KEYWORDS", "ACTION_KEYWORDS", "QUESTION_STARTS",
        "DEFINITION_STARTS", "ENERGY_RANGE_RE", "_is_run_request", "_detect_mode",
        "_SIM_KEYS", "_SIM_PARAM_RE", "_parse_sim_directives",
        "ETA_COUPLING", "DESIGN_KEYS", "DEFECT_KEYS", "_design_key", "_defect_key", "_group_by_design",
        "_fmt_params", "_RECORD_LEGEND", "_format_campaigns", "_record_campaign", "_sweep_holder",
        "_env_first", "DEFECT_CAPS", "_cap_defects", "MIN_PROBES_FOR_SCORE", "_robust_best",
        "STRESS_MODES", "MAX_STRESS_DESIGNS", "_stress_calls", "_stress_check", "_rescale_design",
        "_recorder_tls", "_get_recorder", "_set_recorder", "KE_TOLERANCE", "_get_ke_budget",
        "_set_ke_budget", "_check_energy_budget", "_parse_yield_mj", "_record_result"}
tree = ast.parse(open(APP).read())
nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in KEEP) or
         (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in KEEP for t in n.targets))]
import threading
ns = {"re": re, "os": os, "threading": threading}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "app_subset", "exec"), ns)

# --- keyword routing (the fallback when the LLM router is off or fails) ---
route_cases = [
    ("what can you do?", "chat"), ("hello", "chat"), ("How does this work", "chat"),
    ("What limits the yield of an ICF implosion?", "chat"), ("explain Rayleigh-Taylor growth", "chat"),
    ("tell me about the Hurricane asymmetric piston model", "chat"),
    ("what is a diagnostic?", "chat"), ("what is a sweep?", "chat"),
    ("why did the best design win?", "chat"), ("how robust is it to defects?", "chat"),
    ("compare the 3 MJ and 5 MJ campaigns", "chat"), ("thanks!", "chat"),
    ("optimize a capsule for a 3 MJ laser energy", "single"), ("design a capsule for 5 MJ", "single"),
    ("capsule for a 2 MJ laser", "single"), ("can you optimize a capsule for 4 MJ?", "single"),
    ("what is the best capsule for a 3 MJ laser?", "chat"), ("why did the 3 MJ design win?", "chat"), ("explain how a sweep is run", "chat"), ("what does the movie show?", "chat"),
    ("sweep yield vs laser energy from 1 to 5 MJ with 6 points", "sweep"),
    ("can you sweep yield vs laser energy from 1 to 5 MJ?", "sweep"),
    ("show me a movie of the implosion", "media"), ("can you show me the implosion?", "media"),
    ("render the diagnostics", "media"),
    ("sweep defects for the 3-MJ best design", "chat"), ("run a sweep", "chat"),
    ("scan yield over 2-8 MJ", "sweep"),
    ("Optimize a design at 4 MJ and show me a movie of the implosion", "media"),
]

bad = 0
for msg, want in route_cases:
    got = ns["_detect_mode"](msg)
    if got != want:
        bad += 1
        print(f"ROUTE MISMATCH: {msg!r} -> {got}, want {want}")
print(f"routing: {len(route_cases)} cases, {bad} mismatches")

# --- RUN_SIMULATION parser, on directive shapes gpt-oss produced on lux ---
P = ns["_parse_sim_directives"]
FULL = dict(R0=1.45e-4, v0=-4.48e5, T0=2.0, M_sh=4.5e-7, M_hs=1.2e-8, delta=0.14, mode=4.0, roughness=3.0e-6)
parse_cases = {
 "one line": ("RUN_SIMULATION: R0=1.45e-4, v0=-4.48e5, T0=2.0, M_sh=4.5e-7, M_hs=1.2e-8, delta=0.14, mode=4, roughness=3.0e-6", [FULL]),
 "wrapped": ("```yaml\nRUN_SIMULATION: R0=1.45e-4, v0=-4.48e5, T0=2.0, M_sh=4.5e-7,\n                M_hs=1.2e-8, delta=0.14, mode=4, roughness=3.0e-6\n```\n\n**Why** R0=9 v0=9", [FULL]),
 "yaml": ("```yaml\nRUN_SIMULATION:\n  R0: 1.45e-4\n  v0: -4.48e5\n  T0: 2.0\n  M_sh: 4.5e-7\n  M_hs: 1.2e-8\n  defect:\n    delta: 0.14\n    mode: 4\n    roughness: 3.0e-6\n```", [FULL]),
 "json": ('RUN_SIMULATION: {\n    "R0": 1.45e-4,\n    "v0": -4.48e5,\n    "T0": 2.0,\n    "M_sh": 4.5e-7,\n    "M_hs": 1.2e-8,\n    defect: { delta: 0.14, mode: 4, roughness: 3.0e-6 }\n}', [FULL]),
 "bold": ("**RUN_SIMULATION:** R0=1.45e-4, v0=-4.48e5, T0=2.0, M_sh=4.5e-7, M_hs=1.2e-8, delta=0.14, mode=4, roughness=3.0e-6", [FULL]),
 "two": ("RUN_SIMULATION: R0=1.45e-4, v0=-4.48e5, T0=2.0, M_sh=4.5e-7, M_hs=1.2e-8, delta=0.14, mode=4, roughness=3.0e-6\nRUN_SIMULATION: R0=1.45e-4, v0=-4.48e5, T0=2.0, M_sh=4.5e-7, M_hs=1.2e-8, delta=0.14, mode=4, roughness=3.0e-6", [FULL, FULL]),
 "prose": ("I will now issue a RUN_SIMULATION directive.\n\nThe last design had R0=1e-4.", []),
 "partial": ("RUN_SIMULATION: R0=1.45e-4, v0=-4.48e5", [dict(R0=1.45e-4, v0=-4.48e5)]),
 "extra keys": ("RUN_SIMULATION: R0=1.45e-4, v0=-4.48e5, T0=2.0, M_sh=4.5e-7, M_hs=1.2e-8, \n delta=0.14, mode=4, roughness=3.0e-6, \n counter_push=[{mode:2, amplitude:0.0045}]", [FULL]),
}

pbad = 0
for name, (text, want) in parse_cases.items():
    got = P(text)
    if got != want:
        pbad += 1
        print(f"PARSE MISMATCH [{name}]:\n    got  {got}\n    want {want}")
print(f"parser: {len(parse_cases)} cases, {pbad} mismatches")

# --- session record of a sweep: best design by robust yield, at the energy it was found ---
rbad = 0
def check(name, ok):
    global rbad
    if not ok:
        rbad += 1
        print(f"RECORD MISMATCH: {name}")
D = dict(R0=1.5e-4, v0=-4.5e5, T0=1.0, M_sh=2.5e-7, M_hs=5e-9)
lo_d = dict(D, M_sh=1.5e-7)
hi_d = dict(D, M_sh=4.0e-7)
results = [(dict(lo_d, delta=0.0, mode=10.0, roughness=0.0), 9.0),   # 2 MJ point: fragile
           (dict(lo_d, delta=0.1, mode=4.0, roughness=2e-6), 0.5),
           (dict(hi_d, delta=0.0, mode=10.0, roughness=0.0), 8.0),   # 5 MJ point: robust
           (dict(hi_d, delta=0.1, mode=4.0, roughness=2e-6), 6.0)]
best = [(0.5, 2e6, results[0][0]), (6.0, 5e6, results[2][0])]
holder, best_e = ns["_sweep_holder"](best, results, 5e6, [2e6, 2e6, 5e6, 5e6])
check("robust winner", holder["best_yield"] == 6.0 and holder["best_params"] == results[2][0] and best_e == 5e6)
check("mean over all runs", abs(holder["mean_yield"] - 5.875) < 1e-9 and holder["std_yield"] > 0)
check("empty sweep", ns["_sweep_holder"]([], [], 7e6)[0]["best_params"] is None)
state = {}
ns["_record_campaign"](state, "sweep 2.00-5.00 MJ (2 pts)", "sweep 2 to 5 MJ", 5e6, holder,
                       sweep_points=[(2.0, 9.0, 0.5), (5.0, 8.0, 6.0)], best_e_laser=best_e)
text = ns["_format_campaigns"](state)
check("energy range line", "Laser energies: 2.00-5.00 MJ (2 points). Best design found at 5.00 MJ" in text)
check("best = worst case", "Best yield: 6.000 MJ = the best design's worst yield over the 2 defect point(s)" in text
      and "most favourable run gave 8.000 MJ" in text)
check("runs tagged with energy", "yield 9.000 MJ at 2.00 MJ laser [no defects]" in text
      and "yield 6.000 MJ at 5.00 MJ laser [defect probe]" in text)
check("per-point no-defect and robust", "2.00 MJ: no defects 9.00 MJ, robust 0.50 MJ; 5.00 MJ: no defects 8.00 MJ, robust 6.00 MJ" in text)
ns["_record_campaign"](state, "single @ 3.00 MJ", "optimize at 3 MJ", 3e6, {})
text = ns["_format_campaigns"](state)
check("single campaign line", "Laser energy: 3.00 MJ | KE budget: 45000 J" in text)
check("no-design sweep point", "robust n/a (no agent design)" in ns["_format_campaigns"](
    {"campaigns": [dict(state["campaigns"][0], sweep_points=[(2.0, 1.0, None)])]}))
# --- the Defect_Adversary's budget (defaults: delta <= 0.3, roughness <= 3 um) ---
call = dict(FULL, delta=0.4, roughness=5e-6)
notes = ns["_cap_defects"](call)
check("defects capped", call["delta"] == ns["DEFECT_CAPS"]["delta"] and
      call["roughness"] == ns["DEFECT_CAPS"]["roughness"] and call["mode"] == 4.0 and
      notes.count("capped to") == 2)
small = dict(FULL, delta=0.02, roughness=1e-6)
check("small defects untouched", ns["_cap_defects"](small) == "" and small == dict(FULL, delta=0.02, roughness=1e-6))
# --- stress check: a winner probed once, mildly, is re-run at the top of the budget ---
cap = ns["DEFECT_CAPS"]["roughness"]
dA, dB, dC = (dict(R0=r, v0=-4.6e5, T0=1.1, M_sh=2.84e-7, M_hs=6e-9) for r in (1.6e-4, 1.8e-4, 1.4e-4))
rec = [(dict(dA, delta=0.04, mode=8.0, roughness=1.5e-6), 7.59)] + \
      [(dict(dB, delta=0.05, mode=m, roughness=cap), y) for m, y in ((2.0, 0.03), (5.0, 0.04), (7.0, 0.02))] + \
      [(dict(dC, delta=0.05, mode=5.0, roughness=cap), 0.05)]
fake = {dA["R0"]: 0.025, dC["R0"]: 0.04}            # the designs' yields at the top of the budget
checked, runs = ns["_stress_check"](rec, lambda c: fake[c["R0"]] if c["roughness"] >= cap else 7.5)
y, params, n = ns["_robust_best"](rec, only=checked)
check("collapsed winner gives way", runs == 6 and len(checked) == 2 and y == 0.04 and params["R0"] == dC["R0"] and n == 4)
rec = [(dict(dA, delta=0.04, mode=8.0, roughness=1.5e-6), 7.59), (dict(dB, delta=0.02, mode=10.0, roughness=1e-6), 9.0)]
checked, runs = ns["_stress_check"](rec, lambda c: 8.0 if c["R0"] == dB["R0"] else 0.02)
y, params, n = ns["_robust_best"](rec, only=checked)
check("robust winner kept, rest skipped", runs == 3 and len(checked) == 1 and y == 8.0 and params["R0"] == dB["R0"])
# --- sweep carry-forward: a design rescaled to a new energy, then stress-checked for free ---
R = ns["_rescale_design"]
d4 = R(dict(dA, delta=0.3, mode=2.0, roughness=cap), 4e6)
check("rescaled onto the new budget", set(d4) == set(ns["DESIGN_KEYS"]) and d4["v0"] == dA["v0"] and
      d4["R0"] == dA["R0"] and abs(0.5 * d4["M_sh"] * d4["v0"] ** 2 - 0.015 * 4e6) < 1e-6)
slow = R(dict(dA, v0=-1.5e5), 10e6)
check("M_sh clamped, v0 rescaled", slow["M_sh"] == 2e-6 and abs(0.5 * 2e-6 * slow["v0"] ** 2 - 1.5e5) < 1e-6)
seed = ns["_stress_calls"](d4)           # carried design, already run at every stress mode
agents = [dict(dA, R0=1e-4 + j * 1e-5, delta=0.02, mode=10.0, roughness=1e-6) for j in range(5)]
rec = [(c, 5.0) for c in seed] + [(a, 9.0) for a in agents]
checked, runs = ns["_stress_check"](rec, lambda c: 0.05)
y, params, n = ns["_robust_best"](rec, only=checked)
check("carried design checked without a slot", runs == 12 and len(checked) == 5 and y == 5.0 and
      ns["_design_key"](params) == ns["_design_key"](d4))
# --- a design off the KE budget is not recorded, so it cannot win ---
ns["_set_recorder"]([]); ns["_set_ke_budget"](0.015 * 1e6)
on = dict(dA, M_sh=2 * 15000 / dA["v0"] ** 2, delta=0.3, mode=8.0, roughness=cap)
ns["_record_result"](on, "Total Fusion Yield:        0.010 MJ")
ns["_record_result"](dict(on, v0=-6.0e5, M_sh=8.3e-7), "Total Fusion Yield:        6.575 MJ")
check("off-budget run not recorded", ns["_get_recorder"]() == [(on, 0.010)])
print(f"record: 17 checks, {rbad} mismatches")
sys.exit(1 if bad or pbad or rbad else 0)
