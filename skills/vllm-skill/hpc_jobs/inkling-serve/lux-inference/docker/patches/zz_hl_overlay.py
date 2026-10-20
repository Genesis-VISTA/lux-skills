"""Hyperloom kernel phase: put this tree's vllm overlay into the installed vllm.

hl_overlay/vllm (next to this file) is the image's installed vllm after the other
patches here, as the hlk worker exported it, plus whatever kernel rewrites this
tree carries; hl_overlay/BASE.sha256 lists the export as it was ("<sha256>  vllm/<path>").
A file may have more than one line: its rewrite was merged onto each listed
export (fa4_rel_attention.py: the zenith_v6 image's, and the one in a gfx950
image built from this tree with the MI250X fix-ups baked in), and any of them
counts as "the exported one" below.
Named zz_ so PATCH_FIXUP runs it last: the other patches are the source of truth,
and the overlay only carries the rewrites on top of them.

Per overlay file:
  - same bytes as its export (not rewritten): left alone, whatever is installed;
  - rewritten, and the installed file is the exported one: copied over it (logged);
  - rewritten, and the installed file is not the exported one (the image or a patch
    changed it since the export): FAIL, the rewrite was made against other code;
  - not in the export (a file the rewrite added): copied in if nothing is installed
    there, FAIL if something is;
  - in the export but not installed: FAIL, the overlay is for another vllm.
A FAIL exits 1; the hl tree's run_container.sh makes that fatal for the serve, so a
candidate never runs the base kernels under a rewrite's name. Idempotent: a second
run finds the rewrites installed and copies nothing.

ZENITH_HL_OVERLAY=partial: a rewrite whose installed file has changed since the
export (or, for a file the rewrite added, already exists with other contents) is
skipped with a SKIP line instead of failing; the other rewrites are still copied.
This is for an image whose overlay files match none of the listed exports (say a
vLLM bump), until the rewrites are merged onto it. Any other value behaves as
before; run_container.sh does not run this script at all with ZENITH_HL_OVERLAY=0.
"""
import hashlib
import importlib.util
import os
import shutil
import sys
from pathlib import Path

here = Path(__file__).resolve().parent / "hl_overlay"
partial = os.environ.get("ZENITH_HL_OVERLAY", "1") == "partial"
overlay, manifest = here / "vllm", here / "BASE.sha256"
if not overlay.is_dir():
    print(f"zz_hl_overlay: no {overlay}; nothing to do")
    sys.exit(0)
if not manifest.is_file():
    print(f"zz_hl_overlay: FAIL -- {manifest} is missing; cannot tell rewrites from the base", file=sys.stderr)
    sys.exit(1)
spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print("zz_hl_overlay: FAIL -- vllm is not installed", file=sys.stderr)
    sys.exit(1)
inst = Path(spec.origin).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


base = {}
for line in manifest.read_text().splitlines():
    h, _, rel = line.partition("  ")
    if rel.startswith("vllm/"):
        base.setdefault(rel[len("vllm/"):], set()).add(h)
copy, fail, skip = [], [], []
for d, dirs, files in os.walk(overlay):
    dirs[:] = [x for x in dirs if x != "__pycache__"]
    for f in files:
        src = Path(d) / f
        rel = str(src.relative_to(overlay))
        dst = inst / rel
        h = sha(src)
        if h in base.get(rel, ()):
            continue  # not rewritten
        have = sha(dst) if dst.is_file() else None
        if have == h:
            continue  # rewrite already installed
        if rel in base and have is None:
            fail.append(f"{rel}: in the export, not installed")
        elif rel in base and have not in base[rel]:
            if partial:
                skip.append(f"{rel} (installed file changed since the export; ZENITH_HL_OVERLAY=partial)")
            else:
                fail.append(f"{rel}: installed file is not the exported one (changed since the export)")
        elif rel not in base and have is not None:
            if partial:
                skip.append(f"{rel} (new in the overlay, but the installed vllm has a different one; ZENITH_HL_OVERLAY=partial)")
            else:
                fail.append(f"{rel}: new in the overlay, but the installed vllm has a different one")
        else:
            copy.append((src, dst, "new" if rel not in base else "rewritten"))
missing = sorted(r for r in base if not (overlay / r).is_file())
if fail:
    for x in fail:
        print(f"zz_hl_overlay: FAIL -- {x}", file=sys.stderr)
    print(f"zz_hl_overlay: FAIL -- {len(fail)} file(s); nothing copied", file=sys.stderr)
    sys.exit(1)
for x in skip:
    print(f"zz_hl_overlay: SKIP {x}")
for src, dst, why in copy:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)  # a new mtime, so stale bytecode is not used
    print(f"zz_hl_overlay: {why} {dst} ({sha(dst)[:12]})")
if missing:
    print(f"zz_hl_overlay: {len(missing)} exported file(s) deleted in the overlay stay installed: {missing[:5]}")
print(f"zz_hl_overlay: {len(copy)} file(s) from the overlay" + (f", {len(skip)} skipped" if skip else ""))
