import os
import re
import socket
import sys
import threading
import time
import autogen
import gradio as gr
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- LLM backend: an OpenAI-compatible endpoint (local vLLM or a remote server) ---
# Open-weight models sometimes return null content or stray tool-call artifacts; patch the
# AutoGen OpenAI client's message retrieval to tolerate that instead of raising.
from autogen.oai.client import OpenAIClient

_orig_message_retrieval = OpenAIClient.message_retrieval
def _safe_message_retrieval(self, response):
    for choice in response.choices:
        if hasattr(choice, "message") and choice.message:
            if choice.message.tool_calls:
                choice.message.tool_calls = None
            if choice.message.function_call:
                choice.message.function_call = None
            if choice.message.content is None:
                choice.message.content = ""
    return _orig_message_retrieval(self, response)
OpenAIClient.message_retrieval = _safe_message_retrieval


# Hydrodynamic coupling efficiency: fraction of laser energy delivered to shell KE
ETA_COUPLING = 0.015
DEFAULT_E_LASER = 3.0e6  # 3 MJ if the user doesn't specify

def _parse_laser_energy(message):
    """Extract the requested laser energy (in Joules) from the user's message.
    Recognizes values in MJ or kJ; defaults to 3 MJ if none found."""
    import re
    # Look for a number followed by MJ (with optional space, case-insensitive)
    m = re.search(r"([0-9]*\.?[0-9]+)\s*MJ", message, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1e6
    m = re.search(r"([0-9]*\.?[0-9]+)\s*kJ", message, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1e3
    return DEFAULT_E_LASER


def _parse_custom_headers(header_str):
    """Parse custom headers in 'key: value' format, separated by newlines or commas."""
    if not header_str:
        return None
    headers = {}
    for line in header_str.replace(",", "\n").split("\n"):
        line = line.strip()
        if ": " in line:
            k, v = line.split(": ", 1)
            headers[k.strip()] = v.strip()
    return headers or None


def _env_first(*names, default=None):
    """Return the first environment variable that is set (non-empty) among `names`, else
    `default`. Lets configuration come from the environment (e.g. exported in ~/.bashrc) and
    fall back to code defaults only when nothing is set."""
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


# --- Optional Anthropic backend ---------------------------------------------------------
# The reference deployment is an OpenAI-compatible endpoint (vLLM / AIMS). A frontier
# Anthropic model can be substituted to separate "the orchestration is broken" from "the
# open-weight model is struggling" -- useful when validating a campaign's physics claims.
#
# AutoGen 0.2's built-in AnthropicClient constructs `Anthropic(api_key=...)` with no way to
# pass base_url or extra headers, so it cannot reach a corporate gateway. This subclass
# rebuilds the underlying SDK client with both. Credentials and URLs come from the
# environment only -- nothing is hardcoded here.
try:
    from autogen.oai.anthropic import AnthropicClient as _BaseAnthropicClient
    from anthropic import Anthropic as _AnthropicSDK
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

if _HAS_ANTHROPIC:
    class ProxiedAnthropicClient(_BaseAnthropicClient):
        """AnthropicClient that honours a custom base_url and auth/proxy headers.

        Gateways in front of the Anthropic API commonly authenticate with their own header
        (e.g. Azure APIM's `Ocp-Apim-Subscription-Key`) rather than `x-api-key`, so the key
        is sent as a default header and a placeholder is handed to the SDK.
        """

        def __init__(self, config=None, **kwargs):
            # AutoGen calls custom clients as `cls(placeholder_config, **kwargs)`, passing the
            # config-list entry positionally; the base class only accepts **kwargs.
            if config:
                merged = dict(config)
                merged.update(kwargs)
                kwargs = merged
            base_url = kwargs.pop("base_url", None)
            headers = dict(kwargs.pop("default_headers", None) or {})
            auth_header = kwargs.pop("auth_header", None)
            key = kwargs.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
            if auth_header and key:
                headers[auth_header] = key
                # The SDK requires *some* api_key; the gateway ignores x-api-key entirely.
                kwargs["api_key"] = "unused"
            super().__init__(**kwargs)
            if base_url or headers:
                self._client = _AnthropicSDK(
                    api_key=kwargs.get("api_key") or key,
                    base_url=base_url,
                    default_headers=headers or None,
                    timeout=kwargs.get("timeout", 180),
                    max_retries=kwargs.get("max_retries", 3),
                )


def _build_anthropic_config():
    """Build an AutoGen config backed by the Anthropic Messages API, or None if not selected.

    Activated by ICF_BACKEND=anthropic. Reads (nothing is stored in this file):
      ICF_ANTHROPIC_BASE_URL / ANTHROPIC_BASE_URL   gateway base URL (optional; SDK default)
      ICF_ANTHROPIC_API_KEY  / ANTHROPIC_API_KEY    credential
      ICF_ANTHROPIC_MODEL    / ANTHROPIC_MODEL      model id
      ICF_ANTHROPIC_AUTH_HEADER                     header carrying the key (default x-api-key)
      ANTHROPIC_CUSTOM_HEADERS                      extra "k: v" headers, comma/newline separated
    """
    if (_env_first("ICF_BACKEND", default="") or "").lower() != "anthropic":
        return None
    if not _HAS_ANTHROPIC:
        raise RuntimeError("ICF_BACKEND=anthropic but the `anthropic` package is not installed.")

    api_key = _env_first("ICF_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ICF_BACKEND=anthropic requires ICF_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY.")
    model = _env_first("ICF_ANTHROPIC_MODEL", "ANTHROPIC_MODEL", default="claude-opus-4-1")
    # Some clients advertise a context-window variant by appending a bracketed suffix to the
    # model id (e.g. "Claude-Opus-5[1m]" for the 1M-token context). That suffix is a client-side
    # convention, not part of the deployment name: passing it through makes the gateway reject
    # the request with 400 'Deployment ... is not found'. Strip it.
    model = re.sub(r"\[[^\]]*\]\s*$", "", model).strip()
    base_url = _env_first("ICF_ANTHROPIC_BASE_URL", "ANTHROPIC_BASE_URL")
    auth_header = _env_first("ICF_ANTHROPIC_AUTH_HEADER", default="x-api-key")
    timeout = int(_env_first("ICF_TIMEOUT", "ICF_REQUEST_TIMEOUT", default="180"))

    entry = {
        "api_type": "anthropic",
        "model_client_cls": "ProxiedAnthropicClient",
        "model": model,
        "api_key": api_key,
        "max_tokens": int(_env_first("ICF_MAX_TOKENS", default="2048")),
        "timeout": timeout,
        "max_retries": 3,
        "price": [0, 0],
    }
    if base_url:
        entry["base_url"] = base_url
    if auth_header and auth_header.lower() != "x-api-key":
        entry["auth_header"] = auth_header
    hdrs = _parse_custom_headers(_env_first("ANTHROPIC_CUSTOM_HEADERS", default=""))
    if hdrs:
        entry["default_headers"] = hdrs

    # Extended-thinking models reject temperature != 1.0; keep 1.0 unless told otherwise.
    temp = float(_env_first("ICF_TEMPERATURE", default="1.0"))
    return {"config_list": [entry], "temperature": temp}


def _served_model(base_url, api_key):
    """The model id the endpoint serves, or None if it can't be asked or serves several."""
    import json
    import urllib.request
    req = urllib.request.Request(base_url.rstrip("/") + "/models",
                                 headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            ids = [m["id"] for m in json.load(r).get("data", [])]
    except Exception as e:
        print(f"Could not list the models at {base_url}: {e}")
        return None
    return ids[0] if len(ids) == 1 else None


def _build_config():
    """Build the SINGLE AutoGen LLM config used by ALL agents (one shared model, one
    OpenAI-compatible endpoint). Every value is taken from the environment if set, otherwise
    it falls back to the code default below.

    Set ICF_BACKEND=anthropic to use the Anthropic Messages API instead (see
    _build_anthropic_config); the default remains an OpenAI-compatible endpoint.

    Recognized env vars (first one that is set wins):
      base URL : ICF_BASE_URL, ICF_EXPLORER_BASE_URL
      model    : ICF_MODEL, ICF_EXPLORER_MODEL (default: the one model the endpoint serves)
      api key  : ICF_API_KEY, ICF_EXPLORER_API_KEY
      timeout  : ICF_TIMEOUT, ICF_REQUEST_TIMEOUT
      temp     : ICF_TEMPERATURE
      tokens   : ICF_MAX_TOKENS (default 2048; 8192 for Inkling)
      effort   : ICF_REASONING_EFFORT (default: the server's; medium for Inkling)
    """
    anthropic_cfg = _build_anthropic_config()
    if anthropic_cfg is not None:
        return anthropic_cfg

    base_url = _env_first("ICF_BASE_URL", "ICF_EXPLORER_BASE_URL",
                          default="http://localhost:8000/v1")
    api_key = _env_first("ICF_API_KEY", "ICF_EXPLORER_API_KEY", default="unused")
    # With no model named, use the one the endpoint serves, so pointing ICF_BASE_URL at another
    # node's server is all it takes.
    model = (_env_first("ICF_MODEL", "ICF_EXPLORER_MODEL")
             or _served_model(base_url, api_key) or "gptoss-20b-hedp")
    # A per-request timeout is essential for a REMOTE endpoint: without it, one stalled
    # generation (long-context request, dropped/half-open connection, server-side queueing)
    # blocks the OpenAI client indefinitely and freezes the whole campaign mid-round. On a
    # timeout the SDK retries, so a transient stall recovers instead of hanging forever.
    timeout = int(_env_first("ICF_TIMEOUT", "ICF_REQUEST_TIMEOUT", default="180"))
    # Inkling thinks before every answer (at its template's default effort of 0.9 when none is
    # sent) and the thinking counts against max_tokens, so it gets a lower default effort and a
    # larger token budget. Its server's reasoning parser returns the thinking separately.
    inkling = "inkling" in model.lower()
    max_tokens = int(_env_first("ICF_MAX_TOKENS", default="8192" if inkling else "2048"))
    effort = _env_first("ICF_REASONING_EFFORT", default="medium" if inkling else "")
    entry = {"model": model, "api_key": api_key, "base_url": base_url,
             "max_tokens": max_tokens, "timeout": timeout, "max_retries": 3, "price": [0, 0]}
    if effort:
        entry["reasoning_effort"] = effort

    # gpt-oss is calibrated to sample at temperature 1.0 / top_p 1.0 (OpenAI's stated
    # recommendation); running it much lower can degrade its reasoning. Default to 1.0 and
    # let ICF_TEMPERATURE override (dial down toward ~0.7 only if strict format-following,
    # e.g. the RUN_SIMULATION line, becomes unreliable).
    temp = float(_env_first("ICF_TEMPERATURE", default="1.0"))
    return {"config_list": [entry], "temperature": temp}


# One config, used by every agent and the group-chat manager.
config = _build_config()
# AutoGen 0.2 caches every LLM reply on disk (./.cache/<seed>) unless told otherwise, so a
# repeated request replayed the earlier campaign word for word, whatever the temperature. Off by
# default; set ICF_CACHE_SEED (e.g. 41) to turn the cache back on for reproducible replays.
_cache_seed = _env_first("ICF_CACHE_SEED", default="")
config["cache_seed"] = int(_cache_seed) if _cache_seed else None
_cfg0 = config["config_list"][0]
print(f"Model: {_cfg0['model']} | "
      f"Endpoint: {_cfg0.get('base_url', '(provider default)')} | "
      f"Backend: {_cfg0.get('api_type', 'openai')}")


def _config_with(temperature=None, **entry_overrides):
    """A copy of the shared config with a different temperature and/or per-request settings.

    Per-entry settings (max_tokens, ...) win over top-level ones in AutoGen, so they are written
    into every config_list entry; the shared `config` itself is never modified.
    """
    cfg = dict(config)
    cfg["config_list"] = [{**e, **entry_overrides} for e in config["config_list"]]
    if temperature is not None:
        cfg["temperature"] = temperature
    return cfg


def _register_custom_clients(*agents):
    """Bind the proxied Anthropic client to every agent that needs it.

    AutoGen only instantiates a `model_client_cls` client after the agent registers the
    class by name, so this must run once per agent (and the group-chat manager) after
    construction. A no-op on the OpenAI-compatible path.
    """
    if _cfg0.get("model_client_cls") != "ProxiedAnthropicClient":
        return
    for a in agents:
        if a is not None and getattr(a, "client", None) is not None:
            a.register_model_client(model_client_cls=ProxiedAnthropicClient)

# --- MCP client implementation ---
def call_mcp_icf_server(R0: float, v0: float, T0: float, M_sh: float, M_hs: float, delta: float, mode: float, roughness: float) -> str:
    """Synchronous wrapper to launch the MCP Client and ping the FastMCP Server."""
    async def _run_mcp_client():
        _dir = os.path.dirname(os.path.abspath(__file__))
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[os.path.join(_dir, "icf_mcp_server.py")]
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "run_icf_implosion",
                    arguments={
                        "R0": R0, "v0": v0, "T0": T0, "M_sh": M_sh, "M_hs": M_hs,
                        "delta": delta, "mode": mode, "roughness": roughness
                    }
                )
                return result.content[0].text

    return asyncio.run(_run_mcp_client())


# --- Per-campaign simulation result recorder ---
# Set to a list for the duration of a campaign; the simulation-execution code appends
# (params_dict, yield_MJ) for every run so the orchestrator can find the best design.
#
# THREAD-LOCAL, not a plain global: the Gradio app is shared (and launched with a public
# link), so two users can have campaigns in flight at once. Each campaign runs the group chat
# on its own thread, and a single global recorder would let one session's simulations land in
# another session's results. The simulation hook runs on the campaign's own thread, so a
# thread-local binds each record to the campaign that produced it.
_recorder_tls = threading.local()

def _get_recorder():
    return getattr(_recorder_tls, "active", None)

def _set_recorder(rec):
    _recorder_tls.active = rec

# The agents, and therefore their conversation histories, are module-level singletons, so two
# campaigns running concurrently would interleave turns into the same agent state. Serialize
# whole campaigns behind this lock; the UI still streams normally, later requests simply queue.
_campaign_lock = threading.Lock()

# The KE budget of the campaign currently running, bound per-thread alongside the recorder so
# the simulation hook can check every executed design against it. The energy constraint is the
# whole point of the optimization -- a design that quietly spends 11% more kinetic energy than
# the budget allows is not on the manifold and its yield is not comparable to the others -- and
# asking the PI to police it in prose proved unreliable.
KE_TOLERANCE = 0.05  # fractional deviation from the budget that is reported as a violation

def _get_ke_budget():
    return getattr(_recorder_tls, "ke_budget", None)

def _set_ke_budget(v):
    _recorder_tls.ke_budget = v

def _check_energy_budget(params):
    """Return a warning string if this design is off the campaign's energy manifold, else ''."""
    budget = _get_ke_budget()
    if not budget:
        return ""
    ke = 0.5 * params["M_sh"] * params["v0"] ** 2
    dev = (ke - budget) / budget
    if abs(dev) <= KE_TOLERANCE:
        return ""
    return (f"\n*** ENERGY BUDGET VIOLATION ***\n"
            f"This design has KE = 0.5 * {params['M_sh']:.4g} * ({params['v0']:.4g})^2 = "
            f"{ke:.0f} J, which is {dev*100:+.1f}% off the campaign budget of {budget:.0f} J "
            f"(tolerance +/-{KE_TOLERANCE*100:.0f}%). It is NOT on the constant-energy manifold, "
            f"so its yield is not comparable to the other designs in this campaign. "
            f"Instruct the Design_Explorer to rescale M_sh and/or v0 and re-simulate.\n")

def _parse_yield_mj(result_text):
    """Extract the fusion yield in MJ from an MCP result string; None if not found."""
    import re
    m = re.search(r"Total Fusion Yield:\s*([\d.]+)\s*MJ", result_text or "")
    return float(m.group(1)) if m else None

def _record_result(params, result_text):
    """Append a (params, yield_MJ) record to the active recorder, if one is set.

    A design off the campaign's energy budget is not recorded: the PI is told its yield does not
    count, and it must not win either. (A 1 MJ sweep point once "won" with 6.6 MJ from a design
    carrying 10 times the budget's kinetic energy.)"""
    rec = _get_recorder()
    if rec is not None and not _check_energy_budget(params):
        y = _parse_yield_mj(result_text)
        if y is not None:
            rec.append((dict(params), y))

# --- Robust (min-max) scoring ---
# The campaign solves max_x min_d Y(x, d): the Explorer picks the design vector x, the
# Adversary picks the defect vector d, and a design is only as good as its WORST outcome.
# Scoring by the best yield ever recorded collapses this to max_x max_d and rewards the
# Explorer for reporting its most favourable benign run, which is how the Adversary ended up
# with no influence on the outcome at all.
DESIGN_KEYS = ("R0", "v0", "T0", "M_sh", "M_hs")   # controlled by the Design_Explorer
DEFECT_KEYS = ("delta", "mode", "roughness")        # controlled by the Defect_Adversary

# A design scored on a single benign probe is not yet a robustness measurement. Require the
# Adversary to have attacked it at this many DISTINCT defect points before it can win.
MIN_PROBES_FOR_SCORE = 3

def _design_key(params):
    """Hashable identity of a design point, insensitive to the defect settings."""
    return tuple(round(float(params.get(k, 0.0)), 12) for k in DESIGN_KEYS)

def _defect_key(params):
    return tuple(round(float(params.get(k, 0.0)), 12) for k in DEFECT_KEYS)

def _group_by_design(recorder):
    """Collapse a recorder into {design_key: {"params", "probes", "worst", "best"}}."""
    groups = {}
    for params, y in recorder:
        g = groups.setdefault(_design_key(params), {"params": params, "probes": {}})
        # Keep the worst yield seen at each distinct defect point.
        dk = _defect_key(params)
        if dk not in g["probes"] or y < g["probes"][dk]:
            g["probes"][dk] = y
    for g in groups.values():
        ys = list(g["probes"].values())
        g["worst"] = min(ys) if ys else 0.0
        g["best"] = max(ys) if ys else 0.0
        g["n_probes"] = len(ys)
    return groups

def _robust_best(recorder, only=None):
    """Return (robust_yield, params, n_probes) for the best adequately-probed design.

    'Robust yield' is the minimum yield over every defect point the Adversary tried against
    that design. Designs probed fewer than MIN_PROBES_FOR_SCORE times are held back; if none
    qualify the requirement is relaxed rather than reporting nothing, so that an early or
    truncated campaign still produces an answer (flagged via n_probes by the caller).
    `only`, if given, is the set of design keys allowed to win (the stress-checked ones).
    """
    groups = {k: g for k, g in _group_by_design(recorder).items() if only is None or k in only}
    if not groups:
        return 0.0, None, 0
    qualified = [g for g in groups.values() if g["n_probes"] >= MIN_PROBES_FOR_SCORE]
    pool = qualified or list(groups.values())
    winner = max(pool, key=lambda g: g["worst"])
    return winner["worst"], winner["params"], winner["n_probes"]

# The Adversary does not always attack a design at the top of its budget: it may test the
# eventual winner once, with a mild defect, and spend the rest of the campaign on other designs.
# The robust yield was then the yield under that mild defect, not the design's worst case (a
# 2 MJ winner scored 7.6 MJ on one 1.5 um probe, and gives 0.03 MJ at the 3 um cap). So once the
# chat ends, the code runs the leading designs itself at the top of the budget, at the modes that
# entrain the most mix, and only a design checked this way can win.
STRESS_MODES = (2.0, 4.0, 8.0)
MAX_STRESS_DESIGNS = 4          # at most 3 simulations each

def _stress_calls(params):
    """The simulations that stress-check a design: each STRESS_MODE at the top of the budget."""
    design = {k: float(params[k]) for k in DESIGN_KEYS}
    return [dict(design, delta=DEFECT_CAPS["delta"], mode=mode, roughness=DEFECT_CAPS["roughness"])
            for mode in STRESS_MODES]

def _stress_check(recorder, run_sim):
    """Run the leading designs at the top of the defect budget, best worst case first, appending
    the runs to `recorder`. run_sim(params) returns the yield in MJ, or None.

    A design is checked only while it could still win: extra probes can only lower a worst
    case, so once a checked design's worst case beats every unchecked one, the rest are skipped.
    A design already run at every stress mode (one carried forward from an earlier sweep point)
    counts as checked at no cost and does not take one of the MAX_STRESS_DESIGNS slots.
    Returns (set of checked design keys, number of simulations run)."""
    checked, runs, best, slots = set(), 0, None, 0
    groups = _group_by_design(recorder)
    for key, g in sorted(groups.items(), key=lambda kv: kv[1]["worst"], reverse=True):
        if best is not None and g["worst"] <= best:
            break
        calls = [c for c in _stress_calls(g["params"]) if _defect_key(c) not in g["probes"]]
        if calls:
            if slots >= MAX_STRESS_DESIGNS:
                continue
            slots += 1
        checked.add(key)
        worst = g["worst"]
        for call in calls:
            y = run_sim(call)
            if y is not None:
                recorder.append((call, y))
                runs += 1
                worst = min(worst, y)
        best = worst if best is None else max(best, worst)
    return checked, runs

# --- Sweep carry-forward ---
# Each sweep point used to start from scratch, so its robust yield depended on which designs
# that campaign happened to try: in a 1-10 MJ sweep the 5 and 8 MJ points fell to 1.5 and 2.8 MJ
# between neighbours at 9.8 and 12-14 MJ, although a random scan finds designs keeping 17 and
# 24 MJ there. So the winners of the last MAX_CARRY points are rescaled to each new energy and
# stress-checked before the chat starts; they enter the record as the designs to beat. And a
# sweep campaign cannot end before the agents have tried SWEEP_MIN_DESIGNS designs of their own
# (the PI ended the 8 MJ campaign after 3).
MAX_CARRY = 2
SWEEP_MIN_DESIGNS = 5

def _rescale_design(params, e_laser):
    """The design in `params` moved onto the KE budget of `e_laser`: same R0, v0, T0 and M_hs,
    with M_sh scaled so that 0.5 * M_sh * v0^2 = ETA_COUPLING * e_laser. If that M_sh is outside
    the simulator's range (10-2000 ug), M_sh is clamped and v0 rescaled instead; None if v0 then
    leaves its range (100-800 km/s)."""
    d = {k: float(params[k]) for k in DESIGN_KEYS}
    ke = ETA_COUPLING * e_laser
    m_sh = 2.0 * ke / d["v0"] ** 2
    if not 1e-8 <= m_sh <= 2e-6:
        m_sh = min(max(m_sh, 1e-8), 2e-6)
        d["v0"] = -(2.0 * ke / m_sh) ** 0.5
        if not -8e5 <= d["v0"] <= -1e5:
            return None
    d["M_sh"] = m_sh
    return d

# --- Agent Definitions ---

user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=8,
    code_execution_config=False,
    system_message="You are the execution bridge. When the Principal_Investigator calls run_icf_implosion, you execute it and return the results. Do not modify parameters or interpret results."
)

_PI_SYSTEM_BASE = """\
You are the Principal Investigator leading an ICF capsule design campaign.

IMPORTANT: This is a DECELERATION-PHASE surrogate model. R0 is the shell radius at the START OF
STAGNATION (after the laser drive is off), NOT the initial fabricated capsule radius.
All parameters MUST be in SI units using scientific notation (e.g. 1.5e-4, not 150).

UNITS: All parameters are SI EXCEPT T0 which is in keV (not Kelvin, not eV).
  T0=1.0 means 1.0 keV. Do NOT convert to eV or Kelvin.

ENERGY BUDGET: The shell kinetic energy is constrained by the laser energy via a hydrodynamic
coupling efficiency of 1.5%: KE_budget = 0.015 * E_laser. The exact KE budget for THIS campaign
(based on the requested laser energy) is stated in the kickoff message below. Use THAT value.

KNOWN-GOOD REFERENCE (produces ~7 MJ yield at the 45 kJ / 3 MJ budget, no defects):
  R0=1.5e-4 m, v0=-4.5e5 m/s, T0=1.0 keV, M_sh=2.5e-7 kg, M_hs=5e-9 kg

PROTOCOL — follow this turn order strictly:
1. Address Design_Explorer by name and ask for capsule parameters. State the KE budget for this
   campaign explicitly. WAIT for their response.
2. After receiving the Explorer's parameters, VALIDATE the energy constraint:
   KE = 0.5 * M_sh * v0^2 must be within 5% of the campaign KE budget.
   If violated, ask the Explorer to revise. Do NOT proceed until KE matches the budget.
   Every simulation is also checked in code; if the results come back with an
   ENERGY BUDGET VIOLATION notice, that design is off the manifold and its yield does NOT
   count. Send the Explorer back to rescale M_sh and/or v0, and re-simulate.
3. Address Defect_Adversary by name and ask for defect parameters. WAIT for their response.
4. After receiving BOTH sets of parameters, request a simulation run.
   Do NOT propose parameters yourself — delegate to the specialists.
5. PROBE THE DESIGN AT LEAST 3 TIMES. This campaign optimizes the ROBUST objective
   max_x min_d Y(x,d): a design is scored by its WORST yield across the defect points tried
   against it, NOT by its best. One benign run is not a robustness measurement.
   Keep the 5 design parameters (R0, v0, T0, M_sh, M_hs) FIXED and ask the Adversary for a
   new defect vector (delta, mode, roughness) each time, then re-run. A design probed fewer
   than 3 times cannot win the campaign.
6. Evaluate results using the WORST yield seen for the design:
   - Worst-case yield > 1 MJ = robustly ignites
   - Stagnation pressure > 100 Gbar = good confinement
   - Hot-spot mix fraction > ~30% usually means burn is being quenched
   Report BOTH the nominal (best) and worst-case yield — the gap between them is the
   design's fragility, and a smaller gap is better even at slightly lower nominal yield.
7. Report results and start the next round if needed.

STOPPING CRITERIA:
  You must compare at least 3 DISTINCT designs (different R0/v0/T0/M_sh/M_hs), each probed at
  3+ defect points, before the campaign can converge. Probing one design many times is not a
  campaign — after each design is characterized, send the Explorer back for a new one that
  targets the weakness the Adversary exposed.
  Track the ROBUST (worst-case) yield from each round. Stop the campaign when EITHER:
  a) The robust-yield improvement between the current best and the previous best is < 5% for
     2 consecutive rounds (the design has converged — further iteration is unlikely to help).
  b) You reach the per-campaign round cap stated in the kickoff message (hard cap to
     prevent runaway optimization).
  At the end of each round, report:
     "Round N: worst-case = X.XX MJ over K probes (nominal X.XX MJ; best robust so far: Y.YY MJ, change: +Z.Z%)"
  When stopping, summarize the most ROBUST design found, its parameters, the worst defect
  vector found against it, and its nominal-vs-worst-case gap.
End your final summary with TERMINATE."""

_PI_TOOL_INSTR = """

When you have all 8 parameters and want to run the simulation, output EXACTLY this format:
RUN_SIMULATION: R0=<val>, v0=<val>, T0=<val>, M_sh=<val>, M_hs=<val>, delta=<val>, mode=<val>, roughness=<val>
The User_Proxy will parse this, run the simulation, and return results."""

principal_investigator = autogen.AssistantAgent(
    name="Principal_Investigator",
    llm_config=config,
    system_message=_PI_SYSTEM_BASE + _PI_TOOL_INSTR,
)

explorer_agent = autogen.AssistantAgent(
    name="Design_Explorer",
    llm_config=config,
    system_message="""\
You are the ICF Design Explorer. You propose capsule parameters to maximize fusion yield.
You CANNOT call tools or run simulations. You ONLY output parameter proposals for the
Principal_Investigator to simulate.

IMPORTANT: This is a DECELERATION-PHASE surrogate. R0 is the shell radius at the start of
stagnation (~100-200 um), NOT the initial fabricated capsule outer radius (~1 mm).

ALL VALUES MUST USE SCIENTIFIC NOTATION (e.g. 1.5e-4, not 150e-6 or 0.00015).

ENERGY CONSTRAINT (mandatory):
  KE = 0.5 * M_sh * v0^2 must match the campaign KE budget.
  The budget = 0.015 * E_laser; the Principal_Investigator will tell you the exact value in Joules.
  Show your KE calculation with every proposal and confirm it matches the stated budget.

PARAMETERS:
  R0   [m]    deceleration-phase radius.  Range: 8e-5 to 2.5e-4.   Example: 1.5e-4
  v0   [m/s]  implosion velocity (negative). Range: -7e5 to -3e5.  Example: -4.5e5
  T0   [keV]  initial hotspot temperature.   Range: 0.5 to 3.0.    Example: 1.0
               NOTE: T0 is in keV, NOT Kelvin and NOT eV. Just use the number (e.g. 1.2).
  M_sh [kg]   shell mass.                    Range: 5e-8 to 1e-6.  Example: 2.5e-7
               (that is 50 to 1000 micrograms; 2.5e-7 kg = 250 ug)
  M_hs [kg]   initial hotspot mass.          Range: 1e-9 to 5e-8.  Example: 5e-9
               (that is 1 to 50 micrograms; 5e-9 kg = 5 ug)
  Note: higher laser energy allows a higher KE budget, so M_sh and/or |v0| can be larger.

REFERENCE POINT (at the 45 kJ / 3 MJ budget, yields ~7 MJ):
  R0=1.5e-4, v0=-4.5e5, T0=1.0, M_sh=2.5e-7, M_hs=5e-9
  KE = 0.5 * 2.5e-7 * (4.5e5)^2 = 25312 J. Scale M_sh/v0 to match the actual campaign budget.

Reply with ONLY your 5 parameter values and KE check. Keep it brief.
Propose EXACTLY ONE design for the current round — do NOT enumerate multiple rounds or a
table of designs. Do NOT output the word TERMINATE; only the Principal_Investigator ends the
campaign."""
)

# The Defect_Adversary's budget: the largest defects a campaign applies. The simulator accepts
# delta up to 0.5 and roughness up to 6 um, but a capable model (Inkling) probes at the top of
# its budget, so the caps set how hard designs are hit. Each defect has its own cliff: roughness
# drives mix (>= 3.5 um alone collapses nearly every design the agents find at 2-4 MJ), and
# delta leaves the heavy piston's kinetic energy unconverted (delta = 0.3 alone collapses a
# typical 2 MJ design). At delta 0.3 and 3 um, robust designs still exist (a 300-design scan
# finds worst cases of 3.5 / 7 / 11 MJ at 2 / 3 / 4 MJ), and they are rarer at low energy (1% /
# 9% / 25% of designs keep > 1 MJ). In live sweeps the agents' 2 MJ designs collapse while
# their 3 and 4 MJ ones lose ~10-17%, so the two curves separate. Enforced in code, before the
# simulator is called.
DEFECT_CAPS = {"delta": float(_env_first("ICF_MAX_DELTA", default="0.3")),
               "roughness": float(_env_first("ICF_MAX_ROUGHNESS", default="3e-6"))}

def _cap_defects(call):
    """Cap the defects in `call` (in place) at DEFECT_CAPS; returns a note for each cap applied."""
    notes = ""
    for k, cap in DEFECT_CAPS.items():
        if call[k] > cap:
            notes += (f"  NOTE: {k}={call[k]:.4g} is beyond the defect budget; "
                      f"capped to {cap:.4g}\n")
            call[k] = cap
    return notes

adversary_agent = autogen.AssistantAgent(
    name="Defect_Adversary",
    llm_config=config,
    system_message=f"""\
You are the Defect_Adversary. Your ONLY job is to propose 3 defect parameter values.
Do NOT coordinate other agents. Do NOT discuss design parameters. Do NOT role-play as the PI.
You CANNOT call tools or run simulations.

When the Principal_Investigator asks you for defects, respond with ONLY:
  delta = <value>
  mode = <value>
  roughness = <value>
  Rationale: <one sentence>

PARAMETER BOUNDS (your defect budget, enforced in code; larger values are capped):
  delta     [0.0, {DEFECT_CAPS['delta']:g}]   mode-1 asymmetry.
  mode      [2, 40]       RT mode number.        Typical: 10-20.
  roughness [0, {DEFECT_CAPS['roughness']:g}]   surface roughness [m]. Typical: 1e-7 to 1e-6.

YOUR OBJECTIVE: MINIMIZE the fusion yield of the design currently on the table. The campaign
solves max_x min_d Y(x,d) — the Explorer maximizes over the design x, you minimize over the
defect vector d. A design is scored by its WORST outcome against your attacks, so finding a
genuinely damaging defect is how you do your job. Do not spread probes around for variety;
concentrate them where you expect the most damage.

HOW DEFECTS ACTUALLY DESTROY YIELD (use this — it is the physics in the model):
  - Roughness seeds the Rayleigh-Taylor perturbation that grows into the hot spot. With
    roughness = 0 there is NO mix and mode has no effect; only delta still acts.
  - mode controls how fast RT grows AND when it saturates. Bubbles stop growing near
    a ~ 0.1 * lambda with lambda = 2*pi*R/mode, so a high mode saturates early at a small
    amplitude and entrains LESS mass than a low mode, even though its raw amplification
    factor is far larger. Do not just max out the mode -- the reported amplification is a
    poor guide. Judge modes by the mix mass they actually inject; in practice low modes
    (near the bottom of the range) are the strongest attack, but scan to confirm it for
    THIS design, since the minimum can sit slightly off the boundary.
  - Mixed-in ablator is high-Z, so it multiplies radiative losses (brems ~ Z^2) and quenches
    burn in the contaminated volume. Watch the reported "Hot-Spot Mix Fraction".
  - delta splits the shell into a heavy and a light piston. The light side stops first, the
    heavy side's kinetic energy is left unconverted, and part of the shell misses the hot spot
    (the engaged mass falls as delta^2). The penalty is small for small delta but grows
    steeply: near the top of your budget, asymmetry alone can push a design off the ignition
    cliff, and it compounds with roughness. mode does not affect it.

There is an ignition cliff: below some mix fraction the design shrugs off the defect, and just
above it the yield collapses by orders of magnitude. Your goal is to find that cliff for THIS
design. Bracket it — if an attack barely dents the yield, escalate (up to your budget); if it
collapses the yield, back off to locate the threshold. A design that keeps its yield at the top
of your budget is robust; say so rather than proposing values beyond it.

USE THE FEEDBACK. Look at the yield and mix fraction your previous probes produced against
this design and move toward whatever hurt most. Report your reasoning in one sentence.

Do NOT output the word TERMINATE; only the Principal_Investigator ends the campaign."""
)

# --- Conversational PI ("chat" mode) ---
# This is a SEPARATE agent from `principal_investigator`: it has no tool registered and no
# campaign protocol. It answers every message that is not a request to run something: what the
# system can do and how it works, general ICF physics, and questions about results already collected.
_PI_ANALYST_SYSTEM = f"""\
You are the Principal Investigator of the Agentic RL Fusion Capsule Optimizer, a multi-agent system
for inertial confinement fusion (ICF) capsule design. You are talking directly with a researcher.
Answer whatever they ask: what this system can do and how it works, questions about the results
collected this session, or general ICF physics.

WHAT THE SYSTEM CAN DO (the user starts each of these by typing a request like the example):
  - Single-point design: "optimize a capsule for a 3 MJ laser". The agents design a capsule whose
    shell kinetic energy fits the laser budget and stress-test it against defects.
  - Laser-energy sweep: "sweep yield vs laser energy from 1 to 5 MJ with 6 points". One campaign per
    energy, plotted as the best design's yield with no defects and its worst case under defects. A
    sweep request must give a laser-energy range in MJ.
    Only laser energy can be swept; there is no sweep over defects or other parameters.
  - Media: "show me a movie of the implosion". For the best design found, renders a PNG of the
    time-history diagnostics (radius, areal density, temperature) and an MP4 movie. The movie is a
    schematic 2D cross-section (400 um x 400 um) drawn from the 0D solution, not a 2D hydro
    simulation: the left and right halves of the shell sit at the two piston radii (the mode-1
    asymmetry), the shell is a ring rippled at the RT mode number with amplitude
    roughness*exp(RT growth exponent), and shell and hot spot have fixed illustrative densities.
    It runs from the start of deceleration (radius R0) to about 20 time steps past stagnation,
    time-stamped in ns.
  - Debrief: after a run, ask about the results (best design, robustness, trade-offs, comparisons).

HOW IT WORKS:
  - Agents: the Principal_Investigator runs the campaign, enforces the energy budget and decides when
    to stop. The Design_Explorer proposes the 5 design parameters (R0, v0, T0, M_sh, M_hs). The
    Defect_Adversary proposes the defects that hurt yield most (delta = mode-1 shell asymmetry,
    mode = Rayleigh-Taylor mode number, roughness = surface roughness). An execution proxy runs the
    simulations.
  - Simulations run on a physics surrogate served over MCP. It is a 0D model of the DECELERATION
    phase only (from the start of stagnation through burn), solved as a stiff 10-component ODE system
    with the Radau integrator, under 1 s per run. It combines:
      * asymmetric dual-piston stagnation mechanics with mode-1 asymmetry (Hurricane et al., Phys.
        Plasmas 27, 062704 (2020));
      * Spitzer electron thermal ablation and marginal-ignition physics (Betti et al., Phys. Plasmas
        9, 2277 (2002));
      * the NRL Plasma Formulary Gamow DT reactivity, <sigma v> = 3.68e-12 * T^(-2/3) *
        exp(-19.94 * T^(-1/3)) cm^3/s with T in keV (taken as 0 below 0.5 keV), plus alpha heating
        and bremsstrahlung;
      * linear Rayleigh-Taylor growth with Haan saturation of the bubbles (a ~ 0.1*lambda), so high
        modes saturate early and low modes entrain more material;
      * RT mix: cold ablator entrained into the hot spot quenches the burn by the clean fraction
        (1 - f_mix) and raises bremsstrahlung by (1 + Z_ENH*f_mix);
      * a mode-1 mass defect: the two pistons carry (1+delta)/2 and (1-delta)/2 of the engaged shell
        mass M_sh*(1 - C_DELTA_LOSS*delta^2), so the light side stops first and the heavy side's
        kinetic energy is left unconverted. Large delta alone can collapse the yield.
    Shell and hot-spot mass are conserved (ablated shell mass becomes hot-spot fuel). The mix
    coefficients are calibrated to NIF trends (Ma et al., PRL 111, 085004 (2013); Zylstra/Casey et
    al., Phys. Plasmas 27, 092709 (2020)), so results are a parameterized sensitivity study, not a
    quantitative prediction. NOT modeled: the laser, hohlraum, absorption and laser-plasma
    instabilities, the acceleration phase, shock timing, 2D/3D effects, and mode-1 jetting. R0 is
    the shell radius at the start of stagnation, not the fabricated capsule radius. Outputs include
    stagnation pressure, fusion yield, burn fraction, RT amplification and hot-spot mix fraction
    and mass.
  - Parameter bounds enforced by the simulator: R0 30-500 um, |v0| 100-800 km/s (v0 < 0, inward),
    T0 0.1-10 keV, M_sh 10-2000 ug, M_hs 1-100 ug, delta 0-0.5, mode 2-40, roughness 0-6 um.
    In a campaign the Defect_Adversary's budget is smaller: delta <= {DEFECT_CAPS['delta']:g} and
    roughness <= {DEFECT_CAPS['roughness'] * 1e6:g} um (larger proposals are capped), so a design's
    worst case is its yield under defects up to that size.
  - Energy budget: shell KE = 0.5*M_sh*v0^2 must equal 1.5% of E_laser (the hydro coupling efficiency)
    to within 5%; e.g. at 3 MJ the shell KE must be 45 kJ +/- 2.25 kJ.
  - The objective is robust: a design is scored by its WORST yield over at least 3 defect probes
    (max_x min_d Y), and a campaign compares at least 3 distinct designs. When the agents finish,
    the code itself re-runs the leading designs at the top of the defect budget (modes 2, 4 and
    8); only a design checked this way can win, and these runs are in the record as defect probes.
  - A campaign runs in design rounds (at most 10 for a single design or a sweep point) and stops
    early once the best yield improves by less than 5% over two consecutive rounds (after at least
    3 rounds), or sooner if the campaign's Principal_Investigator agent ends it; the end-of-campaign
    line says which, and warns if the winning design was probed at fewer than 3 defect points.
    A sweep plots, at each energy, the best design's yield with no defects and its robust
    (worst-case) yield under defects. In a sweep, the winners of the previous {MAX_CARRY} points
    are carried forward: rescaled to the new energy (same R0, v0, T0, M_hs; M_sh scaled to the KE
    budget), run by the code at the top of the defect budget before the agents start, and entered
    as the designs to beat. A sweep point cannot end before the agents have tried
    {SWEEP_MIN_DESIGNS} designs of their own.
    A movie request reuses the best design found so far unless the user also asks to optimize.
  - Units: SI throughout except T0, which is in keV.

RULES:
  - You cannot run simulations or start campaigns in this conversation. If the user wants one, give
    them the exact phrasing to type. The four request types above are the ONLY things the system
    can run; never offer or invent any other command or capability. The user cannot set design or
    defect parameters directly, and cannot sweep anything but laser energy: the Design_Explorer
    and the Defect_Adversary choose all parameters themselves during a campaign. If asked to set
    or sweep a parameter, say so plainly.
  - For this session's results, reason ONLY from the campaign record you are given and quote its
    numbers. Never invent simulation results. If nothing has been run yet, say so. Each run in the
    record is tagged [defect probe] or [no defects]. The record holds ONLY the parameters and
    yields it lists: stagnation pressure, areal density, temperature, burn fraction and mix were
    not recorded, so if asked for them say they are not in the record; do not estimate them.
  - For general ICF questions, answer from your own knowledge, and say when an answer goes beyond
    what the surrogate model represents. Do not guess at implementation details of the surrogate
    or of this system beyond what is described here; say you are not sure instead.
  - Never fabricate numbers, constants, fit coefficients, citations or experimental results. If you
    do not know an exact value (e.g. NIF shot data or a fit's coefficients), say so and point to the
    source instead of writing placeholder values.
  - You are an AI agent (a language model) in the role of the Principal Investigator; never claim to
    be human. The person you are talking to is the user, not an agent.
    When describing the system, "you" means the user; never call any of the agents "you".
  - Quote parameters in the record's SI units (m, m/s, kg; T0 in keV). If you convert, convert
    carefully: 1e-6 m = 1 um, 1e-9 m = 1 nm.
  - Be concise and quantitative."""

# The campaign agents keep the shared temperature (gpt-oss is calibrated for 1.0), but the chat
# agent answers factual questions, where lower-temperature sampling invents fewer details.
CHAT_TEMPERATURE = float(_env_first("ICF_CHAT_TEMPERATURE", default="0.4"))

pi_analyst = autogen.AssistantAgent(
    name="Principal_Investigator",
    llm_config=_config_with(temperature=CHAT_TEMPERATURE),
    system_message=_PI_ANALYST_SYSTEM,
)

# --- Intent router ---
# Classifies each message as OPTIMIZE / SWEEP / MEDIA / CHAT. It sees only the message, runs at
# temperature 0 and short reasoning, and falls back to the keyword rules if the reply is unusable.
_ROUTER_SYSTEM = """\
You route messages sent to an ICF capsule-design assistant. Classify the user's message into exactly
one category and reply with ONLY that word:

OPTIMIZE  the user asks to run an optimization campaign that designs a capsule for a laser energy
          (e.g. "optimize a capsule for a 3 MJ laser", "design a target for 5 MJ", "run it at 2 MJ").
SWEEP     the user asks to run a laser-energy sweep, i.e. campaigns over a range of laser energies
          (e.g. "sweep yield vs laser energy from 1 to 5 MJ with 6 points").
MEDIA     the user asks to render or show the implosion movie, animation or diagnostic plots,
          including together with an optimization (e.g. "show me a movie of the implosion",
          "render the diagnostics", "optimize a design at 4 MJ and show me a movie of it").
CHAT      anything else: greetings, questions about what the assistant can do or how it works,
          general physics questions, and questions about results that were already produced
          (e.g. "what can you do?", "what is a sweep?", "why did the best design win?").

Choose OPTIMIZE, SWEEP or MEDIA only when the user clearly asks for that action to be carried out
now. Questions that merely mention these things are CHAT."""

ROUTER_MODE = (_env_first("ICF_ROUTER", default="llm") or "llm").lower()
_router_overrides = {"max_tokens": 512}
if _cfg0.get("api_type", "openai") == "openai":
    _router_overrides["reasoning_effort"] = "low"   # gpt-oss: keep the classification fast
if "inkling" in _cfg0["model"].lower():
    # Inkling's thinking counts against max_tokens and runs a few hundred tokens even at "low".
    _router_overrides["max_tokens"] = 2048
router_agent = autogen.AssistantAgent(
    name="Router",
    llm_config=_config_with(temperature=0.0, **_router_overrides),
    system_message=_ROUTER_SYSTEM,
)

# Bind the proxied Anthropic client (no-op on the OpenAI-compatible path). The group-chat
# manager is created per campaign and registers itself in run_campaign.
_register_custom_clients(principal_investigator, explorer_agent, adversary_agent,
                         pi_analyst, router_agent, user_proxy)

# Open-weight models are unreliable at formal tool-calling, so the PI requests a simulation
# with a plain-text RUN_SIMULATION directive that a hook parses and executes against the MCP
# server. The other hooks tolerate null/malformed content and bound the context window.

def _sanitize_content(sender, message, recipient, silent):
    if isinstance(message, dict):
        content = message.get("content")
        if content is None:
            reasoning = message.pop("reasoning", None) or message.pop("thinking", None)
            message["content"] = reasoning if reasoning else "(no response)"
    elif message is None:
        message = "(no response)"
    return message

for _agent in [principal_investigator, explorer_agent, adversary_agent, user_proxy]:
    _agent.register_hook("process_message_before_send", _sanitize_content)

# Bound the context an agent sees so a long campaign never
# overflows the model's context window. Keep the first message (which carries the
# energy-budget instructions) plus the most recent KEEP_RECENT messages.
KEEP_RECENT = 12

def _trim_history(messages):
    if not messages or len(messages) <= KEEP_RECENT + 1:
        return messages
    return [messages[0]] + messages[-KEEP_RECENT:]

for _agent in [principal_investigator, explorer_agent, adversary_agent]:
    _agent.register_hook("process_all_messages_before_reply", _trim_history)

# A campaign turn may legitimately request several designs at once, so EVERY RUN_SIMULATION
# line is executed — not just the first. Capping the count keeps one runaway turn (models
# occasionally emit a dozen speculative designs) from stalling the campaign.
MAX_SIMS_PER_TURN = 8

# The results banner is emitted ONLY by the hook below, from real engine output. A model that
# has seen a few of these in its context will sometimes write one itself -- a complete,
# correctly-formatted, entirely invented results block -- and then reason from the numbers it
# just made up. Observed in a Claude Opus 5 campaign: three fabricated blocks, one of which
# reported 1.578 MJ for a design the engine deterministically puts at 0.013 MJ, with a
# corrupted trailer ("Residual Kinetic Energy: 942.34 Piano") that gave it away. Such text is
# never recorded by _record_result, so scoring stays clean, but the other agents cannot tell it
# from real output. Strip any banner the model authored before appending genuine results.
_FAKE_RESULT_RE = re.compile(r"^[ \t]*-{2,}\s*SIMULATION(?:\s+RESULTS)?[^\n]*-{2,}[ \t]*$",
                             re.M | re.I)

def _strip_fabricated_results(content):
    """Remove model-authored simulation-result blocks. Returns (clean_content, n_removed).

    Everything from the first self-authored banner to the end of the message is dropped. The
    RUN_SIMULATION directive always precedes its results block, so the prefix that is kept
    still carries the request the hook needs to execute.
    """
    marks = list(_FAKE_RESULT_RE.finditer(content))
    if not marks:
        return content, 0
    return (content[:marks[0].start()].rstrip()
            + f"\n\n[NOTE: {len(marks)} simulation-result block(s) written by the agent itself "
              "were removed here. Only output produced by the physics engine is admissible; "
              "the authoritative results for this turn follow below.]\n"), len(marks)

# gpt-oss writes the directive in many shapes: the one-line form the prompt asks for, the same
# line wrapped after a comma, YAML or JSON blocks ("R0: 1.45e-4", nested under "defect:"), and
# markdown ("**RUN_SIMULATION:** R0=..."). A one-line regex dropped everything after a wrap,
# reported "Missing parameters" and left the agents debugging the parser instead of designing.
# So a directive is the text after the keyword up to the next blank line or code fence, and each
# parameter is the first "key = value" or "key: value" for that key inside it.
_SIM_KEYS = ["R0", "v0", "T0", "M_sh", "M_hs", "delta", "mode", "roughness"]
_SIM_PARAM_RE = re.compile(
    r"(?<![A-Za-z0-9_])[\"'*`]*(R0|v0|T0|M_sh|M_hs|delta|mode|roughness)[\"'*`]*\s*[=:]\s*[\"'*`]*"
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")

def _parse_sim_directives(content):
    """Return one {param: value} dict per RUN_SIMULATION directive in `content`.

    A mention of the keyword with no parameters after it (prose) is not a directive.
    """
    directives = []
    starts = [m.end() for m in re.finditer(r"RUN_SIMULATION", content)]
    for i, start in enumerate(starts):
        end = starts[i + 1] - len("RUN_SIMULATION") if i + 1 < len(starts) else len(content)
        body = re.split(r"\n[ \t]*\n|```", content[start:end], maxsplit=1)[0]
        params = {}
        for key, value in _SIM_PARAM_RE.findall(body):
            params.setdefault(key, float(value))
        if params:
            directives.append(params)
    return directives

def _auto_execute_simulation(sender, message, recipient, silent):
    content = (message.get("content") or "") if isinstance(message, dict) else str(message or "")
    content, _n_fake = _strip_fabricated_results(content)
    # Execute ALL directives, not just the first: stopping at one silently discarded the rest
    # while the PI went on to reason as though every design had been simulated.
    directives = _parse_sim_directives(content)
    if not directives:
        return message

    required = _SIM_KEYS
    total = len(directives)
    blocks = []
    for i, params in enumerate(directives[:MAX_SIMS_PER_TURN], 1):
        label = f" ({i} of {total})" if total > 1 else ""
        if all(k in params for k in required):
            call = {k: params[k] for k in required}
            capped = _cap_defects(call)
            result = call_mcp_icf_server(**call)
            _record_result(call, result)
            blocks.append(f"--- SIMULATION RESULTS{label} ---\n" + capped + result
                          + _check_energy_budget(call))
        else:
            blocks.append(f"--- SIMULATION{label} ---\nERROR: Missing parameters. "
                          f"Got: {list(params.keys())}. Need: {required}. Re-issue it as one line: "
                          "RUN_SIMULATION: R0=<m>, v0=<m/s>, T0=<keV>, M_sh=<kg>, M_hs=<kg>, "
                          f"delta=<0-{DEFECT_CAPS['delta']:g}>, mode=<2-40>, "
                          f"roughness=<0-{DEFECT_CAPS['roughness']:g} m>")
    if total > MAX_SIMS_PER_TURN:
        blocks.append(f"--- NOTE ---\nOnly the first {MAX_SIMS_PER_TURN} of {total} requested "
                      f"simulations were run this turn. Re-request the remainder in the next "
                      f"round; do NOT assume results for designs that were not simulated.")

    content += "\n\n" + "\n\n".join(blocks)
    if isinstance(message, dict):
        message["content"] = content
    else:
        message = content
    return message

principal_investigator.register_hook("process_message_before_send", _auto_execute_simulation)

# --- Campaign runner ---
# Deterministic speaker cycle (open-weight models are unreliable at choosing the next speaker
# themselves): PI -> Explorer -> PI -> Adversary, repeating.
#
# The cycle MUST NOT end on the same agent it begins with. A five-element
# [PI, Explorer, PI, Adversary, PI] list wraps from its last entry straight back to its first,
# giving PI -> PI: the PI both asked for defects and answered itself, so every subsequent turn
# landed one slot out of phase and the specialists drifted into each other's roles (the
# Adversary proposing capsule designs, the Explorer proposing defects). Four elements make the
# repeat seamless: ... -> Adversary -> PI -> Explorer -> ...
#
# Module-level so run_campaign can size its message budget from the true round length.
def _turn_cycle_agents():
    return [principal_investigator, explorer_agent,
            principal_investigator, adversary_agent]

MSGS_PER_ROUND = 4  # len(_turn_cycle_agents()); one full speaker cycle

# Robust scoring costs several speaker cycles per DESIGN. The Adversary must land
# MIN_PROBES_FOR_SCORE probes against one design before it is scorable, and it gets one turn
# per cycle, so a design iteration is roughly MIN_PROBES_FOR_SCORE cycles -- not one. Sizing
# the message budget as max_rounds * MSGS_PER_ROUND therefore bought only max_rounds/3 designs:
# campaign 3 asked for 6 rounds, explored 2 designs, and hit the budget before MIN_DESIGNS=3
# could ever be satisfied, so convergence was unreachable by construction. Scale the budget by
# the probe depth so `max_rounds` keeps meaning "design iterations".
CYCLES_PER_DESIGN_ROUND = MIN_PROBES_FOR_SCORE

def _build_groupchat():
    """Construct a fresh GroupChat + manager with per-campaign speaker state."""
    agents = [user_proxy, principal_investigator, explorer_agent, adversary_agent]
    _turn_cycle = _turn_cycle_agents()
    _cycle_state = {"idx": -1}

    def _deterministic_speaker(last_speaker, groupchat):
        if last_speaker == user_proxy:
            _cycle_state["idx"] = 0
            return principal_investigator
        _cycle_state["idx"] = (_cycle_state["idx"] + 1) % len(_turn_cycle)
        return _turn_cycle[_cycle_state["idx"]]

    groupchat = autogen.GroupChat(
        agents=agents, messages=[], max_round=30,
        speaker_selection_method=_deterministic_speaker,
    )
    return groupchat


def run_campaign(user_message, e_laser, max_rounds, result_holder, seed_designs=None,
                 min_designs=0):
    """Run one full multi-agent optimization campaign at a given laser energy.

    Generator: yields the growing transcript string as agents speak. On completion,
    populates result_holder with {best_yield, best_params, results}.

    `seed_designs` (sweeps): designs carried forward from earlier points, already on this
    energy's KE budget. They are stress-checked before the chat and enter the record as the
    designs to beat. `min_designs`: the campaign cannot end, by convergence or by the PI's
    TERMINATE, before the agents have tried this many designs of their own.
    """
    groupchat = _build_groupchat()
    # AutoGen's max_round counts individual messages (turns), NOT design iterations. One
    # speaker cycle is PI -> Explorer -> PI -> Adversary (4 messages), and a design iteration
    # spans CYCLES_PER_DESIGN_ROUND of those cycles because the Adversary needs that many turns
    # to probe the design to scorable depth. The simulation results ride along inside the PI's
    # own turn. Convert the requested number of DESIGN rounds into a message budget, with
    # headroom for the kickoff and the final summary.
    msgs_per_round = MSGS_PER_ROUND * CYCLES_PER_DESIGN_ROUND
    groupchat.max_round = 3 + max_rounds * msgs_per_round

    # Per-campaign record of (params, yield) for every simulation actually executed. Bound to
    # the chat thread inside _run_chat below (that is where the simulation hook fires).
    recorder = []

    # Designs carried forward from earlier sweep points: run them at the top of the defect
    # budget now, so they enter the record already stress-checked, as the designs to beat.
    seed_keys, seed_lines = set(), []
    if seed_designs:
        yield (f"_Re-running {len(seed_designs)} design(s) carried forward from earlier sweep "
               f"points at the top of the defect budget..._")
    for d in seed_designs or []:
        ys = []
        for call in _stress_calls(d):
            y = _parse_yield_mj(call_mcp_icf_server(**call))
            if y is not None:
                recorder.append((call, y))
                ys.append(y)
        if ys:
            seed_keys.add(_design_key(d))
            seed_lines.append(f"  - {_fmt_params(d)}: worst case {min(ys):.2f} MJ")

    def _own_designs():
        """The agents' own designs this campaign, i.e. not carried forward."""
        return {k: g for k, g in _group_by_design(recorder).items() if k not in seed_keys}

    # --- Deterministic stopping (does NOT rely on the LLM emitting TERMINATE) ---
    # Stop when the best recorded yield has plateaued: < 5% improvement over CONV_PATIENCE
    # consecutive DESIGN ROUNDS, computed in code from the recorded yields.
    #
    # This is measured per ROUND, not per simulation. The running-best sequence is monotone
    # non-decreasing, so consecutive non-improving *simulations* are the common case, not
    # evidence of convergence -- keying off them ended essentially every campaign at the
    # MIN_SIMS floor (3 sims) regardless of whether the design had actually settled, which is
    # why the reported campaigns never reached their round budget. Requiring the plateau to
    # persist across whole rounds, and requiring the swarm to have explored a minimum number
    # of rounds first, lets a campaign iterate as documented.
    CONV_THRESH = 0.05
    CONV_PATIENCE = 2      # consecutive plateaued rounds required to stop
    MIN_ROUNDS = 3         # never converge before this many design rounds have been recorded
    # Robust scoring means one design now occupies several simulations (the Adversary probes
    # it repeatedly), so a round count alone no longer implies the swarm explored anything:
    # three probes of a SINGLE design would otherwise satisfy MIN_ROUNDS and end the campaign
    # after one Explorer proposal. Convergence additionally requires this many distinct,
    # adequately-probed designs to have been compared.
    MIN_DESIGNS = 3
    _round_marks = []      # (msg_count, best-so-far yield) at the close of each design round
    # pi_terminated: the PI ended the campaign itself; held: its early TERMINATEs refused
    _stop = {"pi_terminated": False, "held": set()}

    def _best_so_far():
        # Convergence tracks the ROBUST objective (worst case over each design's probes), not
        # the best benign yield -- otherwise a campaign can "converge" on a design the
        # Adversary has never successfully attacked.
        robust, _params, _n = _robust_best(recorder)
        return robust

    def _note_round_boundary():
        """Record the running-best at the close of a design round.

        Idempotent per round: AutoGen may evaluate is_termination_msg more than once for the
        same message, and double-counting a boundary would let a single plateaued round
        satisfy a two-round patience window.
        """
        if not recorder:
            return
        n_msgs = len(groupchat.messages)
        if _round_marks and _round_marks[-1][0] == n_msgs:
            return
        _round_marks.append((n_msgs, _best_so_far()))

    def _converged():
        if len(_round_marks) < MIN_ROUNDS or len(_round_marks) < CONV_PATIENCE + 1:
            return False
        # Require a genuine comparison across the agents' own designs, each probed enough to
        # have a meaningful worst case, before declaring the search finished.
        own = _own_designs()
        scorable = [g for g in own.values() if g["n_probes"] >= MIN_PROBES_FOR_SCORE]
        if len(scorable) < MIN_DESIGNS or len(own) < min_designs:
            return False
        def imp(cur, prev):
            return (cur - prev) / prev if prev > 0 else 1.0
        recent = [y for _n, y in _round_marks[-(CONV_PATIENCE + 1):]]
        return all(imp(recent[i + 1], recent[i]) < CONV_THRESH for i in range(len(recent) - 1))

    def _is_termination(msg):
        # The campaign ends only on a Principal_Investigator turn, when EITHER the design has
        # converged (deterministic, code-based) OR the PI explicitly signs off with TERMINATE.
        # Restricting to the PI avoids the Explorer/Adversary ending the chat by echoing the
        # keyword or dumping all rounds at once; the code-based convergence avoids depending on
        # the LLM following the stopping protocol (which the larger models often don't). The
        # trailing-markdown strip handles "**TERMINATE**" / "TERMINATE.".
        if not isinstance(msg, dict):
            return False
        # The Adversary speaks last in the cycle, so its turn closes a design round.
        if msg.get("name") == adversary_agent.name:
            _note_round_boundary()
        if msg.get("name") != principal_investigator.name:
            return False
        if _converged():
            return True
        content = msg.get("content")
        if not isinstance(content, str):
            return False
        if content.rstrip().rstrip("*_`~.!: \t\r\n").upper().endswith("TERMINATE"):
            n_own = len(_own_designs())
            if n_own < min_designs:
                _hold_open(content, n_own)
                return False
            _stop["pi_terminated"] = True
            return True
        return False

    def _hold_open(content, n_own):
        """Refuse an early TERMINATE: tell every agent the campaign goes on. The Explorer speaks
        next in the cycle, so the note asks it for a new design."""
        if content in _stop["held"]:     # the same message checked twice
            return
        _stop["held"].add(content)
        note = {"role": "user", "content": (
            f"NOT FINISHED: only {n_own} design(s) of your own have been tried, and this sweep "
            f"point needs at least {min_designs} before it can end, so the TERMINATE is ignored. "
            f"Design_Explorer, propose a NEW design (different R0/v0/T0/M_sh/M_hs, on the KE "
            f"budget) that targets the weakness the Defect_Adversary exposed.")}
        groupchat.append(note, user_proxy)
        for agent in groupchat.agents:
            if agent is not user_proxy:
                manager.send(note, agent, request_reply=False, silent=True)

    manager = autogen.GroupChatManager(
        groupchat=groupchat, llm_config=config, is_termination_msg=_is_termination)
    _register_custom_clients(manager)

    ke_budget = ETA_COUPLING * e_laser
    budget_note = (
        f"CAMPAIGN ENERGY BUDGET: Requested laser energy = {e_laser/1e6:.2f} MJ. "
        f"With {ETA_COUPLING*100:.1f}% hydrodynamic coupling, the shell kinetic energy budget is "
        f"KE = {ke_budget:.0f} J ({ke_budget/1e3:.1f} kJ). "
        f"All capsule designs must satisfy 0.5 * M_sh * v0^2 ~= {ke_budget:.0f} J. "
        f"ROUND BUDGET: run up to {max_rounds} design rounds; iterate the design across "
        f"multiple rounds and only output TERMINATE once the yield has converged per the "
        f"stopping criteria (do not stop after a single round)."
    )
    if min_designs:
        budget_note += (
            f"\n\nThis campaign is one point of a laser-energy sweep: compare at least "
            f"{min_designs} distinct designs of your own before you TERMINATE (an earlier "
            f"TERMINATE is ignored).")
    if seed_lines:
        budget_note += (
            f"\n\nINCUMBENT DESIGNS: the best designs from the earlier sweep points, rescaled to "
            f"this budget (same R0, v0, T0 and M_hs; M_sh scaled to the KE budget). The code has "
            f"already run each at the top of the defect budget (delta={DEFECT_CAPS['delta']:g}, "
            f"roughness={DEFECT_CAPS['roughness']:g} m, modes "
            f"{', '.join(f'{m:g}' for m in STRESS_MODES)}):\n" + "\n".join(seed_lines) +
            f"\nThe best of these worst cases, {_robust_best(recorder)[0]:.2f} MJ, is the robust "
            f"yield to beat. Refine these designs or try new ones; they do not count toward "
            f"your own designs.")

    chat_error = []
    done = threading.Event()

    def _run_chat():
        # Bind the recorder and the energy budget to THIS thread: the RUN_SIMULATION hook fires
        # here, so this is where _record_result and _check_energy_budget look for them.
        _set_recorder(recorder)
        _set_ke_budget(ke_budget)
        try:
            user_proxy.initiate_chat(
                manager,
                message=f"USER REQUEST: {user_message}\n\n{budget_note}\n\nPrincipal_Investigator, please coordinate the swarm to find a robust design at this energy budget."
            )
        except Exception as e:
            chat_error.append(str(e))
        finally:
            _set_recorder(None)
            _set_ke_budget(None)
            done.set()

    # Serialize campaigns: the agents (and their histories) are module-level singletons, so
    # two concurrent campaigns would corrupt each other's conversation state. Held for the
    # whole campaign and released in the finally below (explicit rather than `with`, so the
    # streaming loop underneath does not need re-indenting).
    _campaign_lock.acquire()
    try:
        thread = threading.Thread(target=_run_chat, daemon=True)
        thread.start()
    except BaseException:
        _campaign_lock.release()
        raise

    seen = 0
    accumulated = ("**Carried forward from earlier sweep points** (worst case at the top of the "
                   "defect budget):\n" + "\n".join(seed_lines)) if seed_lines else ""
    # Keepalive: during a model turn no new messages appear, so without periodic output the
    # streaming connection to the browser can drop and the UI "freezes" mid-run even though the
    # backend keeps going. Re-emit the transcript with a small rotating marker every few seconds
    # so the stream stays alive and the user can see the agents are still working.
    KEEPALIVE_SECS = 4.0
    last_emit = time.time()
    spin = 0
    # try/finally so an abandoned generator (browser closed mid-campaign, which raises
    # GeneratorExit at the yield) still releases the campaign lock instead of deadlocking
    # every later request.
    try:
        while not done.is_set():
            msgs = groupchat.messages
            if len(msgs) > seen:
                for msg in msgs[seen:]:
                    name = msg.get("name", "System")
                    content = msg.get("content", "")
                    if content:
                        accumulated += f"\n\n---\n**{name}:**\n{content}"
                        yield accumulated.strip()
                seen = len(msgs)
                last_emit = time.time()
            else:
                time.sleep(0.5)
                if time.time() - last_emit >= KEEPALIVE_SECS:
                    spin = (spin + 1) % 3
                    yield (accumulated + f"\n\n---\n_agents working{'.' * (spin + 1)}_").strip()
                    last_emit = time.time()

        for msg in groupchat.messages[seen:]:
            name = msg.get("name", "System")
            content = msg.get("content", "")
            if content:
                accumulated += f"\n\n---\n**{name}:**\n{content}"
                yield accumulated.strip()

        thread.join(timeout=5)
    finally:
        _campaign_lock.release()

    if chat_error:
        accumulated += f"\n\n---\n**ERROR:** {chat_error[0]}"
        yield accumulated.strip()

    # Stress check: run the leading designs at the top of the defect budget (see _stress_check).
    checked, stress_runs = set(), 0
    if recorder:
        yield (accumulated + "\n\n---\n_Checking the leading design(s) at the top of the defect "
               "budget..._").strip()
        checked, stress_runs = _stress_check(
            recorder, lambda call: _parse_yield_mj(call_mcp_icf_server(**call)))

    # Aggregate statistics over every simulation run this campaign.
    import statistics
    yields = [y for _params, y in recorder]
    mean_yield = statistics.fmean(yields) if yields else 0.0
    std_yield = statistics.stdev(yields) if len(yields) > 1 else 0.0

    # Nominal best: the highest single yield anywhere in the campaign. Kept for reference and
    # for comparison against the robust figure -- the gap between them is the whole point of
    # running an adversary.
    nominal_yield, nominal_params = 0.0, None
    for params, y in recorder:
        if y > nominal_yield:
            nominal_yield, nominal_params = y, params

    # Robust best: the design whose WORST case under adversarial attack is highest. This is
    # the campaign's actual answer to max_x min_d Y(x, d). Only stress-checked designs compete.
    best_yield, best_params, best_probes = _robust_best(recorder, only=checked or None)
    n_designs = len(_group_by_design(recorder))

    # Always show a clear end-of-campaign line, even if the LLM never produced a clean summary.
    if not chat_error:
        if _converged():
            reason = "converged"
        elif _stop["pi_terminated"]:
            reason = "the PI ended it"
        elif len(groupchat.messages) >= groupchat.max_round:
            reason = "reached the round budget"
        else:
            reason = "the conversation stopped"
        under_probed = (best_probes < MIN_PROBES_FOR_SCORE)
        caveat = (f" (WARNING: winning design was probed at only {best_probes} defect "
                  f"point(s); {MIN_PROBES_FOR_SCORE} required for a robustness claim)"
                  if under_probed else "")
        carried = (f", {len(seed_keys)} of them carried forward from earlier sweep points"
                   if seed_keys else "")
        accumulated += (
            f"\n\n---\n**Campaign complete** ({reason}). "
            f"Ran {len(recorder)} simulation(s) over {n_designs} design(s){carried}.\n\n"
            f"- **Robust yield (max-min objective): {best_yield:.2f} MJ** "
            f"— worst case over {best_probes} adversarial probe(s){caveat}\n"
            f"- Nominal best (single most favourable run): {nominal_yield:.2f} MJ\n"
            f"- Adversarial degradation: "
            f"{((nominal_yield - best_yield) / nominal_yield * 100) if nominal_yield > 0 else 0.0:.1f}%\n"
            f"- Stress check: {len(checked)} leading design(s) re-run by the code at the top of the "
            f"defect budget (delta={DEFECT_CAPS['delta']:g}, roughness={DEFECT_CAPS['roughness']:g} m, "
            f"modes {', '.join(f'{m:g}' for m in STRESS_MODES)}): {stress_runs} simulation(s), "
            f"included above"
        )
        yield accumulated.strip()

    result_holder["best_yield"] = best_yield          # robust (max-min) objective
    result_holder["best_params"] = best_params
    result_holder["best_probes"] = best_probes
    result_holder["nominal_yield"] = nominal_yield    # best single run, for comparison
    result_holder["nominal_params"] = nominal_params
    result_holder["results"] = recorder
    result_holder["yields"] = yields
    result_holder["mean_yield"] = mean_yield
    result_holder["std_yield"] = std_yield
    result_holder["n_carried"] = len(seed_keys)
    result_holder["winner_carried"] = best_params is not None and _design_key(best_params) in seed_keys


# --- Request routing helpers ---
# Trigger words that mark a message as a request to RUN something (a sweep or media render).
# Module-level so _detect_mode and _is_run_request share one source of truth.
SWEEP_KEYWORDS = ["sweep", "scan", "as a function of", "vs laser", "versus laser",
                  "curve", "parameter study", "yield vs"]
MEDIA_KEYWORDS = ["movie", "animation", "animate", "video", "render", "show the implosion",
                  "show me the implosion", "show the graph", "show the plot", "diagnostic",
                  "time histor", "density field"]

RUN_KEYWORDS = ["optimize", "optimise", "simulate", "run a", "run the sim", "run another",
                "design a", "new campaign", "make a movie"]
# Verbs that make a question an instruction ("can you sweep ...?", "what happens if you run ...?").
ACTION_KEYWORDS = RUN_KEYWORDS + ["run ", "sweep", "scan", "show me", "show the", "render",
                                  "animate", "generate", "plot", "make a"]
QUESTION_STARTS = ("what", "why", "how", "which", "who", "when", "where", "explain", "describe",
                   "tell me", "is ", "are ", "does ", "do ", "should ", "could you explain",
                   "can you explain", "can you tell")
# Questions about what something IS are answered even if they name an action ("what is a sweep?").
DEFINITION_STARTS = ("what is a", "what's a", "what are", "what does", "define", "explain", "describe")

# A laser-energy range such as "1 to 5 MJ" or "2-10 MJ". Shared by the router and _sweep_energies.
ENERGY_RANGE_RE = re.compile(
    r"([0-9]*\.?[0-9]+)\s*(?:to|-|–|—|and)\s*([0-9]*\.?[0-9]+)\s*MJ", re.IGNORECASE)

def _is_run_request(message):
    """True if the message asks for a new single-point campaign."""
    import re
    m = message.lower()
    if any(k in m for k in RUN_KEYWORDS):
        return True
    # A request naming a laser energy together with an action word is a run request.
    has_energy = bool(re.search(r"[0-9.]+\s*(MJ|kJ)", message, re.IGNORECASE))
    return has_energy and any(k in m for k in ["optimize", "simulate", "run", "design", "capsule", "laser"])

def _detect_mode(message):
    """Route a message to "sweep", "media", "single" (run a campaign) or "chat" (answer it)."""
    m = message.lower().strip()
    if m.startswith(DEFINITION_STARTS) and not any(k in m for k in RUN_KEYWORDS):
        return "chat"
    is_question = m.endswith("?") or m.startswith(QUESTION_STARTS)
    if is_question and not any(k in m for k in ACTION_KEYWORDS):
        return "chat"
    if any(k in m for k in SWEEP_KEYWORDS):
        # A sweep without a laser-energy range is not runnable; let the PI explain the phrasing.
        return "sweep" if ENERGY_RANGE_RE.search(message) else "chat"
    if any(k in m for k in MEDIA_KEYWORDS):
        return "media"
    if _is_run_request(message):
        return "single"
    return "chat"

_ROUTER_LABELS = {"OPTIMIZE": "single", "SWEEP": "sweep", "MEDIA": "media", "CHAT": "chat"}

def _classify_with_llm(message):
    """Ask the router LLM for the message's intent. Returns a mode, or None if unusable."""
    try:
        reply = router_agent.generate_reply(messages=[{"role": "user", "content": message}])
    except Exception as e:
        print(f"[router] LLM classification failed, using keyword rules: {e}")
        return None
    if isinstance(reply, dict):
        reply = reply.get("content")
    words = re.findall(r"[A-Z]+", (reply or "").upper())
    labels = [w for w in words if w in _ROUTER_LABELS]
    return _ROUTER_LABELS[labels[0]] if labels else None

def _route(message):
    """Pick the mode for a message: the LLM's intent if available, else the keyword rules.

    Code-level guards apply either way: a sweep needs a laser-energy range, and a run request
    that also asks for the movie or plots goes to media mode, which runs the campaign first.
    """
    mode = _classify_with_llm(message) if ROUTER_MODE == "llm" else None
    if mode is None:
        mode = _detect_mode(message)
    if mode == "single" and any(k in message.lower() for k in MEDIA_KEYWORDS):
        mode = "media"
    if mode == "sweep" and not ENERGY_RANGE_RE.search(message):
        mode = "chat"
    return mode

def _parse_num_points(message, default=5, cap=12):
    import re
    m = re.search(r"(\d+)\s*(?:-?\s*point|points|samples|sample|values|energies)", message, re.IGNORECASE)
    if m:
        return max(2, min(cap, int(m.group(1))))
    return default

def _parse_energy_range(message):
    """Return (lo_J, hi_J) for a sweep. Defaults to 1-10 MJ."""
    import re
    m = re.search(r"(?:from|between)?\s*([0-9]*\.?[0-9]+)\s*(?:to|-|and|–|—)\s*([0-9]*\.?[0-9]+)\s*MJ",
                  message, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1e6, float(m.group(2)) * 1e6
    return 1.0e6, 10.0e6

def _parse_step_mj(message):
    """Return a step size in MJ if the user specified one (e.g. 'increments of 1',
    'in steps of 0.5', 'every 2 MJ'), else None."""
    import re
    m = re.search(r"(?:increments?|steps?|every|spacing|interval)\s*(?:of\s*)?([0-9]*\.?[0-9]+)\s*(?:MJ)?",
                  message, re.IGNORECASE)
    if m:
        try:
            v = float(m.group(1))
            return v if v > 0 else None
        except ValueError:
            return None
    return None

def _segment_energies(lo, hi, step_mj, default_n):
    """Energies (J) for a single range. Uses step if given, else default_n points."""
    if hi < lo:
        lo, hi = hi, lo
    if step_mj:
        step = step_mj * 1e6
        n = int(round((hi - lo) / step)) + 1
        n = max(2, n)
        energies = [lo + step * i for i in range(n)]
        if energies[-1] < hi - 1e-6:
            energies.append(hi)
        return energies
    if default_n <= 1:
        return [lo]
    return [lo + (hi - lo) * i / (default_n - 1) for i in range(default_n)]


def _sweep_energies(message, cap=30):
    """Build the list of laser energies (Joules) for a sweep from the user's request.

    Supports MULTIPLE range segments, each with its own optional step size, e.g.
    "from 1 to 2 MJ in increments of 0.1 and 2 to 10 MJ in increments of 1".
    Each "X to Y MJ" range is paired with the nearest following step phrase (if any)
    before the next range. Segments are concatenated and de-duplicated. Falls back to
    a single range / point-count / default. Capped to `cap` campaigns total.
    """
    import re
    range_re = ENERGY_RANGE_RE
    step_re = re.compile(
        r"(?:increments?|steps?|every|spacing|interval)\s*(?:of\s*)?([0-9]*\.?[0-9]+)",
        re.IGNORECASE)

    ranges = list(range_re.finditer(message))
    default_n = _parse_num_points(message)

    if not ranges:
        lo, hi = _parse_energy_range(message)
        energies = _segment_energies(lo, hi, _parse_step_mj(message), default_n)
    else:
        energies = []
        for i, rm in enumerate(ranges):
            lo = float(rm.group(1)) * 1e6
            hi = float(rm.group(2)) * 1e6
            # Find a step phrase that occurs after this range but before the next one.
            seg_end = ranges[i + 1].start() if i + 1 < len(ranges) else len(message)
            sm = step_re.search(message, rm.end(), seg_end)
            step_mj = float(sm.group(1)) if sm else None
            energies.extend(_segment_energies(lo, hi, step_mj, default_n))

    # De-duplicate (shared endpoints between adjacent segments) while preserving order.
    seen, deduped = set(), []
    for e in energies:
        key = round(e / 1e5)  # 0.1 MJ resolution
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    return deduped[:cap]


# --- Media output dir ---
def _new_media_dir():
    import uuid
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media_output", uuid.uuid4().hex[:8])
    os.makedirs(d, exist_ok=True)
    return d

def _defect_free_yield(params):
    """Yield in MJ of the design in `params` with no defects (delta = roughness = 0, so the mode
    has no effect)."""
    design = {k: float(params[k]) for k in DESIGN_KEYS}
    res = call_mcp_icf_server(**design, delta=0.0, mode=2.0, roughness=0.0)
    return _parse_yield_mj(res) or 0.0

def _plot_sweep(points, outdir, probes=None):
    """points: list of (E_laser_MJ, defect_free_MJ, worst_case_MJ or None); probes: the number of
    defect probes behind each worst case (marked on the plot if too few). Returns PNG path.

    Both curves are for the same design, the best one at that energy (the robust winner): its
    yield with no defects, and its worst yield over the defects the Adversary tried against it.
    The gap between them is what the defects cost that design. (The best single run was no
    reference: the Adversary probes at the top of its budget, so that run had defects too.)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [p[0] for p in points]
    best = [p[1] for p in points]
    worst = [p[2] if p[2] is not None else float("nan") for p in points]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(xs, worst, best, color="gray", alpha=0.2, linewidth=0)
    ax.plot(xs, best, "o-", color="steelblue", linewidth=2, markersize=7,
            label="best design, no defects")
    ax.plot(xs, worst, "s-", color="crimson", linewidth=2, markersize=7,
            label="best design, worst case under defects (robust)")
    for x, w, n in zip(xs, worst, probes or []):
        if n < MIN_PROBES_FOR_SCORE and w == w:
            ax.annotate(f"only {n} probe{'s' if n != 1 else ''}", (x, w), textcoords="offset points",
                        xytext=(8, -3), ha="left", fontsize=8, color="crimson")
    ax.set_xlabel("Laser Energy (MJ)")
    ax.set_ylabel("Fusion Yield (MJ)")
    ax.set_ylim(bottom=0)
    ax.set_title("Agent-Optimized Fusion Yield vs. Laser Energy\n(best design, without and with defects)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    png = os.path.join(outdir, "yield_vs_laser.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


# --- Gradio Blocks UI ---
# These are DESIGN ROUNDS (full PI -> Explorer -> PI -> Adversary -> PI cycles). run_campaign
# converts them into AutoGen's message-count budget. A campaign also stops early when the PI
# emits TERMINATE (yield converged), so these are ceilings, not fixed counts. Sweeps run one
# campaign per energy point with the same ceiling as a single design: with only 4 rounds, a
# point that started badly could end with nothing that ignited (0 MJ at 3 MJ in one sweep).
SWEEP_MAX_ROUNDS = 10
SINGLE_MAX_ROUNDS = 10
MEDIA_MAX_ROUNDS = 6

def _fallback_design(e_laser):
    """A physically reasonable design at the given energy budget, used when an agent
    campaign produces no simulation (keeps a sweep curve populated). Scales shell mass
    to the budget at the reference velocity."""
    ke_budget = ETA_COUPLING * e_laser
    v0 = -4.5e5
    M_sh = 2.0 * ke_budget / (v0 ** 2)
    M_sh = max(5e-8, min(2e-6, M_sh))
    return {"R0": 1.5e-4, "v0": v0, "T0": 1.0, "M_sh": M_sh, "M_hs": 5e-9,
            "delta": 0.04, "mode": 12.0, "roughness": 5e-7}

# --- Conversational ("chat") helpers ---
def _fmt_params(params):
    return ", ".join(f"{k}={v:.3g}" for k, v in params.items())


_RECORD_LEGEND = (
    "Parameter key: R0 = shell radius at the start of stagnation [m], v0 = implosion velocity [m/s], "
    "T0 = initial hot-spot temperature [keV], M_sh = shell mass [kg], M_hs = hot-spot mass [kg]. "
    "Defects: delta = mode-1 (left/right) mass asymmetry, 0-0.5: the two halves of the shell carry "
    "(1+delta)/2 and (1-delta)/2 of its mass. mode = Legendre mode number l (2-40) of the "
    "Rayleigh-Taylor perturbation seeded by the surface roughness (wavelength 2*pi*R/l); it is "
    "unrelated to delta. roughness = initial surface roughness amplitude [m]. Only the parameters "
    "and yields below were recorded. Campaigns are listed in the order they ran.")

def _format_campaigns(state):
    """Serialize every campaign and media render this session into a text block for the PI."""
    campaigns, media = state.get("campaigns", []), state.get("media", [])
    if not campaigns and not media:
        return "(no campaign results recorded yet)"
    lines = [_RECORD_LEGEND, ""]
    for i, c in enumerate(campaigns, 1):
        latest = " (most recent)" if i == len(campaigns) else ""
        lines.append(f"### Campaign {i}{latest}: {c['label']}")
        lines.append(f"User request: {c['user_message']}")
        e_best = c.get("best_e_laser", c["e_laser"])
        ke = ETA_COUPLING * e_best
        if c.get("sweep_points"):
            es = [p[0] for p in c["sweep_points"]]
            lines.append(f"Laser energies: {min(es):.2f}-{max(es):.2f} MJ ({len(es)} points). "
                         f"Best design found at {e_best/1e6:.2f} MJ | KE budget there: {ke:.0f} J "
                         f"({ke/1e3:.1f} kJ)")
        else:
            lines.append(f"Laser energy: {e_best/1e6:.2f} MJ | KE budget: {ke:.0f} J ({ke/1e3:.1f} kJ)")
        bp = c.get("best_params")
        if bp:
            g = _group_by_design(c.get("results", [])).get(_design_key(bp))
            if g:
                lines.append(f"Best yield: {c.get('best_yield', 0):.3f} MJ = the best design's worst "
                             f"yield over the {g['n_probes']} defect point(s) tried against it "
                             f"(its most favourable run gave {g['best']:.3f} MJ)")
            else:
                lines.append(f"Best yield: {c.get('best_yield', 0):.3f} MJ")
            lines.append(f"Best design (defects shown are from its first run): {_fmt_params(bp)}")
        scope = " (all laser energies together)" if c.get("sweep_points") else ""
        lines.append(f"Runs: {len(c.get('results', []))} | mean yield{scope} "
                     f"{c.get('mean_yield', 0):.3f} ± {c.get('std_yield', 0):.3f} MJ")
        run_e = c.get("run_e_laser") or []
        for j, (params, y) in enumerate(c.get("results", [])):
            probe = any(params.get(k, 0) for k in ("delta", "roughness"))
            tag = "[defect probe]" if probe else "[no defects]"
            at = f" at {run_e[j]/1e6:.2f} MJ laser" if j < len(run_e) else ""
            lines.append(f"  - yield {y:.3f} MJ{at} {tag}  <-  {_fmt_params(params)}")
        if c.get("sweep_points"):
            pts = "; ".join(
                f"{e:.2f} MJ: no defects {clean:.2f} MJ, robust "
                + (f"{rob:.2f} MJ" if rob is not None else "n/a (no agent design)")
                for e, clean, rob in c["sweep_points"])
            lines.append(f"Sweep points (laser: both for that point's best design; no defects = "
                         f"its yield with delta = roughness = 0, robust = its worst-case yield "
                         f"under the Adversary's defects): {pts}")
        if c.get("media"):
            lines.append(f"Rendered media: {c['media']}")
        lines.append("")
    for i, m in enumerate(media, 1):
        lines.append(f"### Media render {i}: implosion movie (MP4, schematic 2D cross-section from the "
                     f"start of deceleration through stagnation) and time-history plot (PNG: radius, "
                     f"areal density, temperature) "
                     f"shown to the user for the design with yield {m['yield']:.3f} MJ at "
                     f"{m['e_laser']/1e6:.2f} MJ laser: {_fmt_params(m['design'])}")
    return "\n".join(lines)


def chat_with_pi(state):
    """Ask the conversational PI to answer state['chat'] (which ends with the user's message),
    given the session's campaign record (possibly empty) as context. Multi-turn: prior Q&A is
    replayed."""
    context = _format_campaigns(state)
    messages = [
        {"role": "user",
         "content": f"Campaign record for this session so far:\n\n{context}"},
        {"role": "assistant",
         "content": "Understood. I have the session's campaign record. How can I help?"},
    ]
    messages += [{"role": m["role"], "content": m["content"]} for m in state.get("chat", [])]
    try:
        reply = pi_analyst.generate_reply(messages=messages)
    except Exception as e:
        return f"The Principal Investigator could not answer: {e}"
    if isinstance(reply, dict):
        reply = reply.get("content")
    return reply or "(no response)"


def _record_campaign(state, label, user_message, e_laser, holder, sweep_points=None, media=None,
                     best_e_laser=None):
    """Append one campaign's structured results to the session state."""
    state.setdefault("campaigns", []).append({
        "label": label,
        "user_message": user_message,
        "e_laser": e_laser,
        "best_e_laser": best_e_laser if best_e_laser is not None else e_laser,
        "best_yield": holder.get("best_yield", 0.0),
        "best_params": holder.get("best_params"),
        "results": holder.get("results", []),
        "yields": holder.get("yields", []),
        "mean_yield": holder.get("mean_yield", 0.0),
        "std_yield": holder.get("std_yield", 0.0),
        "run_e_laser": holder.get("run_e_laser"),
        "sweep_points": sweep_points,
        "media": media,
    })


def _sweep_holder(sweep_best, sweep_results, default_e, run_e_laser=None):
    """Summarize a whole sweep as one campaign record. Returns (holder, best_e_laser).

    `sweep_best` holds (robust_yield, e_laser, best_params) per point. The sweep's best design is
    the point with the highest ROBUST yield, like every other campaign, and it keeps the laser
    energy it was found at, so a later movie request labels it correctly. `run_e_laser` gives
    the laser energy of each run in `sweep_results`, so the PI can tell the points apart.
    """
    import statistics
    best_y, best_e, best_p = max(sweep_best, key=lambda b: b[0]) if sweep_best else (0.0, default_e, None)
    yields = [y for _p, y in sweep_results]
    return {
        "best_yield": best_y, "best_params": best_p, "results": sweep_results, "yields": yields,
        "mean_yield": statistics.fmean(yields) if yields else 0.0,
        "std_yield": statistics.stdev(yields) if len(yields) > 1 else 0.0,
        "run_e_laser": run_e_laser,
    }, best_e


def handle_request(user_message, chat_history, state):
    """Main entry point. Generator yielding (chatbot_messages, image, video).

    `state` is a session-scoped dict {"campaigns": [...], "chat": [...]} that is mutated in
    place to persist results and the analysis conversation across turns.
    """
    state.setdefault("campaigns", [])
    state.setdefault("chat", [])
    if not (user_message or "").strip():   # blank input: nothing to do
        yield list(chat_history or []), gr.update(), gr.update()
        return
    history = list(chat_history or [])
    history.append({"role": "user", "content": user_message})

    def _assistant(text):
        return history + [{"role": "assistant", "content": text}]

    mode = _route(user_message)
    print(f"[router] {mode:6s} <- {user_message[:80]!r}")

    # --- Conversation: anything that is not a request to run something goes to the PI ---
    if mode == "chat":
        state["chat"].append({"role": "user", "content": user_message})
        yield _assistant("_Principal_Investigator is thinking…_"), gr.update(), gr.update()
        answer = chat_with_pi(state)
        state["chat"].append({"role": "assistant", "content": answer})
        yield _assistant(answer), gr.update(), gr.update()
        return

    if mode == "single":
        e_laser = _parse_laser_energy(user_message)
        holder = {}
        for transcript in run_campaign(user_message, e_laser, SINGLE_MAX_ROUNDS, holder):
            yield _assistant(transcript), gr.update(), gr.update()
        _record_campaign(state, f"single @ {e_laser/1e6:.2f} MJ", user_message, e_laser, holder)
        return

    if mode == "sweep":
        energies = _sweep_energies(user_message)
        n = len(energies)
        lo, hi = energies[0], energies[-1]
        log = (f"# Laser-Energy Sweep\n\nRunning **{n}** agent campaigns from "
               f"**{lo/1e6:.2f} MJ** to **{hi/1e6:.2f} MJ** "
               f"({SWEEP_MAX_ROUNDS} rounds and at least {SWEEP_MIN_DESIGNS} designs each; the "
               f"best designs of the last {MAX_CARRY} points are carried forward to the next). "
               f"This will take several minutes.\n")
        yield _assistant(log), gr.update(), gr.update()

        points = []
        probes = []               # defect probes behind each point's worst case
        sweep_best = []          # (robust_yield, e_laser, best_params) per point, for the PI debrief
        sweep_results = []        # flattened (params, yield) over the whole sweep
        run_e = []                # laser energy of each entry in sweep_results
        carry = []                # winners of the last MAX_CARRY points, most recent first
        for i, e_laser in enumerate(energies):
            log += f"\n### Point {i+1}/{n}: {e_laser/1e6:.2f} MJ laser\n"
            yield _assistant(log), gr.update(), gr.update()
            holder = {}
            sub_msg = f"optimize an ICF capsule for a {e_laser/1e6:.2f} MJ laser"
            seeds = {}
            for p in carry:
                d = _rescale_design(p, e_laser)
                if d is not None:
                    seeds.setdefault(_design_key(d), d)
            last = ""
            for transcript in run_campaign(sub_msg, e_laser, SWEEP_MAX_ROUNDS, holder,
                                           seed_designs=list(seeds.values()),
                                           min_designs=SWEEP_MIN_DESIGNS):
                last = transcript
                yield _assistant(log + "\n" + last), gr.update(), gr.update()
            if holder.get("best_params"):
                sweep_best.append((holder.get("best_yield", 0.0), e_laser, holder["best_params"]))
                # A carried design that won again replaces its own ancestor in the list.
                won = holder["best_params"]
                carry = [won] + [p for p in carry if _design_key(_rescale_design(p, e_laser) or p)
                                 != _design_key(won)]
                carry = carry[:MAX_CARRY]
            sweep_results.extend(holder.get("results", []))
            run_e.extend([e_laser] * len(holder.get("results", [])))
            # Robust yield: None if the agents found no design.
            robust_y = holder.get("best_yield", 0.0) if holder.get("best_params") else None
            nsim = len(holder.get("results", []))
            note = ""
            if holder.get("best_params"):
                clean_y = _defect_free_yield(holder["best_params"])
            else:
                # Agent campaign produced no simulation — fall back to a direct run of a
                # reasonable design so the sweep curve stays populated (no worst case).
                clean_y = _defect_free_yield(_fallback_design(e_laser))
                note = " _(fallback direct simulation)_"
            points.append((e_laser / 1e6, clean_y, robust_y))
            probes.append(holder.get("best_probes", 0))
            worst = f"{robust_y:.2f} MJ" if robust_y is not None else "n/a"
            if robust_y is not None and probes[-1] < MIN_PROBES_FOR_SCORE:
                worst += f" (only {probes[-1]} defect probe(s))"
            if holder.get("winner_carried"):
                note += " _(a design carried forward from an earlier point)_"
            log += (f"\n**Point {i+1} result: best design {clean_y:.2f} MJ with no defects, "
                    f"{worst} worst case under defects ({nsim} sims)**{note}\n")
            yield _assistant(log), gr.update(), gr.update()

        outdir = _new_media_dir()
        png = _plot_sweep(points, outdir, probes)
        summary = ("\n\n## Sweep Complete\n\nBoth yields are for the best design at each energy."
                   "\n\n| Laser Energy (MJ) | No Defects (MJ) "
                   "| Worst Case Under Defects (MJ) | Defect Probes |\n|---|---|---|---|\n")
        for (e_mj, clean_y, robust_y), n_p in zip(points, probes):
            worst = f"{robust_y:.2f}" if robust_y is not None else "n/a"
            flag = (f" (fewer than {MIN_PROBES_FOR_SCORE})"
                    if robust_y is not None and n_p < MIN_PROBES_FOR_SCORE else "")
            summary += f"| {e_mj:.2f} | {clean_y:.2f} | {worst} | {n_p}{flag} |\n"
        log += summary

        # Record the whole sweep as one campaign for the PI debrief.
        sweep_holder, best_e = _sweep_holder(sweep_best, sweep_results, hi, run_e)
        _record_campaign(
            state, f"sweep {lo/1e6:.2f}-{hi/1e6:.2f} MJ ({n} pts)",
            user_message, hi, sweep_holder, best_e_laser=best_e,
            sweep_points=points)

        yield _assistant(log), gr.update(value=png), gr.update()
        return

    if mode == "media":
        import icf_viz
        e_laser = _parse_laser_energy(user_message)

        # A media request is usually a FOLLOW-UP ("...now show the movie"), so by default reuse
        # the best design already found this session instead of re-running the whole optimization
        # (which is slow and can disagree with the earlier result). Only run a fresh campaign if
        # the user explicitly asks to optimize/run, or if no design has been found yet.
        wants_new_run = any(k in user_message.lower() for k in
                            ["optimize", "simulate", "run a", "run the sim", "new campaign", "design a"])
        best, best_yield, best_e = None, 0.0, e_laser
        for c in state.get("campaigns", []):
            bp = c.get("best_params")
            if bp and c.get("best_yield", 0.0) >= best_yield:
                best, best_yield, best_e = bp, c.get("best_yield", 0.0), c.get("best_e_laser", e_laser)

        if best is not None and not wants_new_run:
            e_laser = best_e
            log = (f"# Simulation Media\n\nReusing the best design found earlier this session "
                   f"(**{best_yield:.2f} MJ** yield at {e_laser/1e6:.2f} MJ laser) and rendering its "
                   f"implosion movie and diagnostic plots. _No new optimization is run — ask me to "
                   f"\"optimize\" if you want a fresh campaign._\n\n"
                   f"**Design:** " + _fmt_params(best) + "\n")
            yield _assistant(log), gr.update(), gr.update()
        else:
            reason = "you asked for a new optimization" if wants_new_run else "no design has been found yet this session"
            log = (f"# Simulation Media\n\nRunning an agent campaign at {e_laser/1e6:.2f} MJ "
                   f"({reason}), then rendering its implosion movie and diagnostic plots.\n")
            yield _assistant(log), gr.update(), gr.update()

            holder = {}
            sub_msg = f"optimize an ICF capsule for a {e_laser/1e6:.2f} MJ laser"
            for transcript in run_campaign(sub_msg, e_laser, MEDIA_MAX_ROUNDS, holder):
                yield _assistant(log + "\n" + transcript), gr.update(), gr.update()

            best, best_yield = holder.get("best_params"), holder.get("best_yield", 0.0)
            if not best:
                best = _fallback_design(e_laser)
                holder["best_params"] = best  # so the PI debrief reflects what was actually rendered
                log += ("\n\n_(the agent campaign produced no recorded simulation, so the agents' "
                        "narrative yield is not a real result — rendering a physically reasonable "
                        "fallback design instead)_\n")
            else:
                log += f"\n\n## Best design found (recorded yield {holder.get('best_yield', 0):.2f} MJ)\n"
            _record_campaign(state, f"media @ {e_laser/1e6:.2f} MJ", user_message, e_laser, holder)
            yield _assistant(log), gr.update(), gr.update()

        log += "\nRendering implosion movie and diagnostics...\n"
        yield _assistant(log), gr.update(), gr.update()

        outdir = _new_media_dir()
        try:
            png = icf_viz.render_diagnostics(best, outdir)
            yield _assistant(log + "\n_Diagnostics ready, rendering movie..._"), gr.update(value=png), gr.update()
            mp4 = icf_viz.render_implosion_movie(best, outdir)
            state.setdefault("media", []).append(
                {"e_laser": e_laser, "design": dict(best), "yield": best_yield})
            log += "\n\n**Media rendering complete.**"
            yield _assistant(log), gr.update(value=png), gr.update(value=mp4)
        except Exception as e:
            log += f"\n\n**Media generation failed:** {e}"
            yield _assistant(log), gr.update(), gr.update()
        return


def build_theme():
    # Gradio 6 takes the theme in launch(), not in the Blocks constructor.
    return gr.themes.Origin(
        primary_hue="green",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    )


def build_ui():
    _here = os.path.dirname(os.path.abspath(__file__))
    # The white logo is legible on the dark theme (amd_logo.png is the README's).
    _logo = os.path.join(_here, "amd_logo_white.png")
    avatars = (None, _logo if os.path.exists(_logo) else None)
    # Force dark mode as the default by adding the theme param on first load.
    _force_dark = """
    function refresh() {
        const url = new URL(window.location);
        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
    }
    """
    with gr.Blocks(title="Agentic RL Fusion Capsule Optimizer") as demo:
        # Session-scoped store of all campaign results + the PI analysis conversation.
        results_state = gr.State({"campaigns": [], "chat": []})
        with gr.Row(equal_height=True):
            with gr.Column(scale=2, min_width=120):
                if os.path.exists(_logo):
                    gr.Image(value=_logo, height=64, show_label=False, container=False,
                             interactive=False)
            with gr.Column(scale=6):
                gr.Markdown(
                    "<div style='text-align:right; padding-top:18px;'>"
                    "<span style='display:inline-block; background:#2a2f36; color:#ffffff; "
                    "font-size:1.8em; font-weight:600; padding:6px 12px; border-radius:14px; "
                    "border:1px solid #3c4450;'>powered by LUX and the AIMS Ecosystem</span></div>"
                )
        gr.Markdown("# Agentic RL Fusion Capsule Optimizer\n"
                    "Multi-agent system for robust fusion yield assessment. "
                    "Ask for a single point design (*'optimize a capsule for a 3 MJ laser energy'*), "
                    "a **sweep** (e.g. *'sweep yield vs laser energy "
                    "from 1 to 5 MJ with 6 points'*), or **media** "
                    "(e.g. *'show me a movie of the implosion'*). "
                    "Or just **chat with the Principal Investigator**: ask what the system can do, "
                    "how it works, general ICF questions, or, after a run, about the results "
                    "(e.g. *'why did the best design win?'*, *'how robust is it to defects?'*, "
                    "*'compare the 3 MJ and 5 MJ campaigns'*).")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=600, label="Agent Conversation",
                                     avatar_images=avatars, render_markdown=True)
                with gr.Row():
                    txt = gr.Textbox(placeholder="Message the fusion design swarm…",
                                     scale=8, show_label=False, container=False,
                                     lines=1, max_lines=6, autofocus=True)
                    submit = gr.Button("Send", variant="primary", scale=1, min_width=80)
            with gr.Column(scale=2):
                image = gr.Image(label="Plots", type="filepath")
                video = gr.Video(label="Dynamics")
                gr.Markdown(
                    "<div style='text-align:right; font-size:0.75em; color:#888; "
                    "line-height:1.4;'>"
                    "ICF core physics model served via MCP derived from:<br>"
                    "O. A. Hurricane <i>et al.</i>, \"An analytic asymmetric-piston model "
                    "for the impact of mode-1 shell asymmetry on ICF implosions,\" "
                    "<i>Phys. Plasmas</i> <b>27</b>, 062704 (2020).<br>"
                    "R. Betti <i>et al.</i>, \"Deceleration phase of inertial confinement "
                    "fusion implosions,\" <i>Phys. Plasmas</i> <b>9</b>, 2277–2286 (2002)."
                    "<br><br>"
                    "Multi-agent architecture inspired by LLNL MADA project:<br>"
                    "M. H. Shachar <i>et al.</i>, \"Multiagent design assistant for the "
                    "simulation of inertial fusion energy,\" "
                    "<a href='https://doi.org/10.1103/c6jc-fnkm'><i>Phys. Rev. E</i> "
                    "<b>114</b>, 025303 (2026)</a>."
                    "</div>"
                )

        def _submit(message, history, state):
            state = state or {"campaigns": [], "chat": []}
            for chatbot_msgs, img, vid in handle_request(message, history, state):
                yield chatbot_msgs, img, vid, state

        submit.click(_submit, [txt, chatbot, results_state],
                     [chatbot, image, video, results_state]).then(
            lambda: "", None, txt)
        txt.submit(_submit, [txt, chatbot, results_state],
                   [chatbot, image, video, results_state]).then(
            lambda: "", None, txt)
        # Force dark mode as the default on page load.
        demo.load(None, None, None, js=_force_dark)
    return demo


if __name__ == "__main__":
    demo = build_ui()
    # Several apps can share a node, so unless ICF_PORT is set, Gradio takes the first free port
    # from 7860 up at the moment it binds. (Picking it earlier, in the job script, raced: two
    # jobs starting together on one node both chose 7860.)
    port = _env_first("ICF_PORT")
    demo.launch(server_name="0.0.0.0", server_port=int(port) if port else None,
                share=True, theme=build_theme(), prevent_thread_lock=True)
    node, port = socket.gethostname().split(".")[0], demo.server_port
    print(f"UI: http://{node}:{port}. From your laptop:\n"
          f"  ssh -N -J <user>@home.ccs.ornl.gov -L {port}:{node}:{port} "
          f"<user>@login1.lux.olcf.ornl.gov\n  then browse to http://localhost:{port}", flush=True)
    demo.block_thread()
