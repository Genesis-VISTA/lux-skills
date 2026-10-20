#!/usr/bin/env python3
"""
Guard the `self.drafter` dereferences that only the last PP rank can satisfy.

Why this patch exists
---------------------
vLLM builds the speculative-decoding drafter on exactly one rank
(vllm/v1/worker/gpu_model_runner.py, in GPUModelRunner.__init__):

    # NOTE(Jiayi): currently we put the entire draft model on
    # the last PP rank. This is not ideal if there are many
    # layers in the draft model.
    if self.speculative_config and get_pp_group().is_last_rank:
        self.drafter: (...)

so on a non-last PP rank the attribute is never assigned. Five later sites
read it back behind a guard that tests `self.speculative_config` -- which is
true on *every* rank, because it is the engine-wide config -- without also
testing that this rank actually has a drafter. With pipeline parallelism the
non-last ranks hit a bare AttributeError:

    RuntimeError: Worker failed with error
    ''GPUModelRunner' object has no attribute 'drafter''

raised out of determine_available_memory() -> profile_run() -> _dummy_run(),
i.e. during engine startup, before a single token is served. Observed with
TP8 PP2 across two Lux nodes: only Worker_PP1_TP0 logged "Loading drafter
model...", and the PP0 ranks died.

This is specific to the eagle-family methods, which is what makes it bite us:
`SpeculativeConfig.use_eagle()` returns True for "eagle", "eagle3", **"mtp"**
and the other eagle-style methods (config/speculative.py:1906). MTP is our
default, so "MTP on" and "PP>1" could not both be true before this.

Single-node runs are unaffected and stay bit-identical: with PP=1 every rank
*is* the last rank, `hasattr(self, "drafter")` is True everywhere, and each
guard evaluates exactly as it did before.

The five sites
--------------
Reachable on a non-last PP rank whenever use_eagle() is true:

  ~2601  _build_attn_metadata: picks spec_decode_common_attn_metadata by
         asking `isinstance(self.drafter, ...)` / `self.drafter.kv_cache_gid`
  ~2616  the first per-group proposer attn metadata branch (isinstance)
  ~2620  the Gemma4Proposer per-group block table branch
  ~6188  _dummy_run: `assert isinstance(self.drafter, ...)` then
         `self.drafter.dummy_run(...)`
  ~7164  initialize_kv_cache: `self.drafter.initialize_attn_backend(...)`
  ~7217  _check_and_update_cudagraph_mode: initialize_cudagraph_keys(...)

Skipping all of these on a non-last rank is correct rather than merely safe.
The draft model lives entirely on the last PP rank, so its collectives run
inside *that* rank's TP group -- with TP8/PP2 that is the eight GPUs of the
second node. The first node's ranks are in a different TP group and were
never participants, so there is no collective for them to miss and no hang
to trade the crash for. They have no drafter weights, no drafter KV cache and
no drafter cudagraphs to set up.

The one site deliberately left alone is ~7460
(`uses_extract_hidden_states()`), which has the same shape but is
unreachable for mtp. Patching it would widen the anchor surface for no gain;
if upstream ever routes MTP through it, this script will keep applying
cleanly and that site will need its own guard.

Why hasattr and not `self.drafter = None`
-----------------------------------------
Assigning None on the other ranks looks tidier and is worse. Upstream already
branches on presence in two places -- `if hasattr(self, "drafter")` before
load_model (~5345) and `getattr(self, "drafter", None)` in get_draft_model
(~3332) -- so "absent" is the established encoding for "this rank has no
drafter". A None would make the load_model guard fire on ranks that have
nothing to load, and the `assert isinstance(self.drafter, EagleProposer|...)`
at ~6188 would still fail, just as an AssertionError instead. Matching
upstream's own idiom keeps this patch a guard rather than a redesign.

Upstream status
---------------
Present on the pinned build 0.2.1.dev32+gbc0f47cd0. Worth reporting: the
NOTE at the drafter construction shows PP was a considered case, so the
missing guards read as an oversight rather than an unsupported combination.
"""
import argparse
import importlib.util
import os
import re
import sys

TARGET = "vllm/v1/worker/gpu_model_runner.py"

# Inserted into the first patched guard so a second run is a no-op, and so the
# reason is legible to anyone reading the patched file in a container.
MARKER = "zenith: drafter exists only on the last PP rank"

# ---- site 1: _build_attn_metadata, the spec_decode_common_attn_metadata pick.
OLD_1 = '''            if self.speculative_config and spec_decode_common_attn_metadata is None:
'''
NEW_1 = '''            # {marker} (see
            # patches/spec_decode_drafter_pp_guard.py). Without the hasattr
            # this raises AttributeError on every rank but the last.
            if (
                self.speculative_config
                and hasattr(self, "drafter")
                and spec_decode_common_attn_metadata is None
            ):
'''.format(marker=MARKER)

# ---- sites 2 and 3: the two per-group proposer branches, immediately below.
# These are isinstance() tests, so getattr(..., None) is enough -- None is not
# an instance of either proposer and the branch falls through. Site 2 tests a
# model-specific MTP proposer class; its name is read from the target file
# (SITE_2_RE) and substituted for {cls}.
SITE_2_RE = re.compile(
    r"^            if self\.speculative_config and "
    r"isinstance\(self\.drafter, (\w+MTPProposer)\):$",
    re.MULTILINE,
)
OLD_2 = '''            if self.speculative_config and isinstance(self.drafter, {cls}):
'''
NEW_2 = '''            if self.speculative_config and isinstance(
                getattr(self, "drafter", None), {cls}
            ):
'''

OLD_3 = '''            elif self.speculative_config and isinstance(self.drafter, Gemma4Proposer):
'''
NEW_3 = '''            elif self.speculative_config and isinstance(
                getattr(self, "drafter", None), Gemma4Proposer
            ):
'''

# ---- site 4: _dummy_run. 12-space indent, three clauses. The indent and the
# clause count are what make this anchor distinct from site 6, which has the
# same three clauses at 8 spaces.
OLD_4 = '''            if self.speculative_config and (
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_draft_model()
                or self.speculative_config.uses_extract_hidden_states()
            ):
'''
NEW_4 = '''            if (
                self.speculative_config
                and hasattr(self, "drafter")
                and (
                    self.speculative_config.use_eagle()
                    or self.speculative_config.uses_draft_model()
                    or self.speculative_config.uses_extract_hidden_states()
                )
            ):
'''

# ---- site 5: initialize_kv_cache. 8-space indent, two clauses.
OLD_5 = '''        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_draft_model()
        ):
'''
NEW_5 = '''        if (
            self.speculative_config
            and hasattr(self, "drafter")
            and (
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_draft_model()
            )
        ):
'''

# ---- site 6: _check_and_update_cudagraph_mode. 8-space indent, three clauses.
OLD_6 = '''        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_draft_model()
            or self.speculative_config.uses_extract_hidden_states()
        ):
'''
NEW_6 = '''        if (
            self.speculative_config
            and hasattr(self, "drafter")
            and (
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_draft_model()
                or self.speculative_config.uses_extract_hidden_states()
            )
        ):
'''

# Each anchor must appear exactly this many times. Checking the count rather
# than just presence is the point: if upstream adds a seventh site with the
# same shape, an unchecked replace() would silently fix one and leave the other
# to crash at startup on the next PP run.
SITES = [
    ("spec_decode_common_attn_metadata pick", OLD_1, NEW_1, 1),
    ("first per-group proposer branch", OLD_2, NEW_2, 1),
    ("Gemma4Proposer per-group branch", OLD_3, NEW_3, 1),
    ("_dummy_run drafter dummy_run", OLD_4, NEW_4, 1),
    ("initialize_kv_cache drafter attn backend", OLD_5, NEW_5, 1),
    ("cudagraph mode drafter keys", OLD_6, NEW_6, 1),
]


def find_target():
    """Locate the model runner inside whichever site-packages vLLM lives in.

    Deliberately does NOT `import vllm`: importing it executes
    vllm/platforms/rocm.py, which calls torch.cuda.get_device_properties() at
    module scope and dies in a build container with no GPU exposed.
    find_spec() resolves the package location from the filesystem without
    running __init__.py.
    """
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        sys.exit("vllm is not installed; run this inside the container")
    root = os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))
    path = os.path.join(root, TARGET)
    if not os.path.exists(path):
        sys.exit(f"expected {path} to exist; vLLM layout changed")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report status without writing")
    args = ap.parse_args()

    path = find_target()
    with open(path) as fh:
        src = fh.read()

    if MARKER in src:
        print(f"already patched: {path}")
        return 0

    # Confirm the bug is still the bug before rewriting anything. If upstream
    # has moved the drafter off the last-PP-rank-only construction, these
    # guards are no longer needed and silently adding them would hide the
    # change rather than benefit from it.
    if "if self.speculative_config and get_pp_group().is_last_rank:" not in src:
        print(f"ERROR: the last-PP-rank drafter construction is gone from {path}",
              file=sys.stderr)
        print("upstream may have fixed this; re-derive or retire the patch",
              file=sys.stderr)
        return 1

    m = SITE_2_RE.search(src)
    cls = m.group(1) if m else "<MTP proposer not found>"
    sites = [(name, old.format(cls=cls), new.format(cls=cls), want)
             for name, old, new, want in SITES]

    missing = []
    for name, old, _new, want in sites:
        got = src.count(old)
        if got != want:
            missing.append(f"  {name}: expected {want} occurrence(s), found {got}")
    if missing:
        print(f"ERROR: drafter guard anchors do not match {path}", file=sys.stderr)
        print("\n".join(missing), file=sys.stderr)
        print("vLLM reformatted these conditions; re-derive the patch",
              file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {path}  ({len(SITES)} drafter guards)")
        return 0

    out = src
    for _name, old, new, want in sites:
        out = out.replace(old, new, want)

    with open(path, "w") as fh:
        fh.write(out)
    print(f"patched: {path}  ({len(SITES)} drafter guards for PP>1 + spec decode)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
