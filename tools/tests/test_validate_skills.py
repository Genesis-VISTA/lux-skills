"""Tests for tools/validate_skills.py.

Plain pytest functions; every fixture is built in ``tmp_path`` by
``make_skill`` (no fixture files).  The rule table ``CASES`` gives one
positive case per (rule id, severity) pair the validator can emit, and
``test_every_registered_rule_has_a_case`` fails when a rule is added to the
validator without a test.  ``test_clean_skill_is_silent_for`` is the negative
case for every rule at once; targeted negatives follow for the hazards that
look similar to a compliant file (quoted colons, CRLF, ``>-`` scalars, ...).
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import validate_skills as vs  # noqa: E402  (imported via its path on purpose)

VALIDATOR = TOOLS_DIR / "validate_skills.py"
ALL = vs.PROFILES

DESC = "Summarise climate model output files. Use when the user asks for NetCDF summaries."
BODY = "# Clean skill\n\nRead the input file and write a short summary.\n"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_skill(
    tmp_path: Path,
    dirname: str,
    frontmatter_text: str,
    body: str = BODY,
    filename: str = "SKILL.md",
    *,
    raw: bytes | str | None = None,
    parent: Path | None = None,
) -> Path:
    """Create ``<parent or tmp_path>/<dirname>/<filename>`` and return the skill directory.

    ``frontmatter_text`` is the YAML between the ``---`` delimiters.  ``raw``
    replaces the whole file content (bytes for BOM / encoding fixtures).
    """

    skill_dir = (parent or tmp_path) / dirname
    skill_dir.mkdir(parents=True, exist_ok=True)
    if raw is None:
        content = f"---\n{frontmatter_text}\n---\n{body}".encode("utf-8")
    elif isinstance(raw, str):
        content = raw.encode("utf-8")
    else:
        content = raw
    (skill_dir / filename).write_bytes(content)
    return skill_dir


def fm(name: str, description: str = DESC, extra: str = "") -> str:
    """Frontmatter text for a skill called ``name``; ``extra`` lines are appended verbatim."""

    text = f"name: {name}\ndescription: {description}"
    return f"{text}\n{extra}" if extra else text


def validate(paths: Path | Sequence[Path], profiles: Sequence[str] = ALL, *, full_catalog: bool = True) -> list[vs.Finding]:
    """Discover, load and run the profiles the way ``main`` does (catalog checks included)."""

    inputs = [paths] if isinstance(paths, (str, Path)) else list(paths)
    discovery = vs.discover_skills(inputs)
    names = {entry.name for entry in discovery.entries}
    contexts = [vs.load_skill(entry, catalog_names=names) for entry in discovery.entries]
    return vs.run_profiles(contexts, profiles, catalog=True, full_catalog=full_catalog, discovery=discovery)


def rule_ids(findings: Sequence[vs.Finding], profile: str | None = None) -> set[str]:
    return {f.rule_id for f in findings if profile is None or f.profile == profile}


def severities(findings: Sequence[vs.Finding], rule_id: str) -> set[str]:
    return {f.severity for f in findings if f.rule_id == rule_id}


def run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "NO_COLOR": "1"},
    )


def git(*args: str, cwd: Path) -> str:
    proc = subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


Builder = Callable[[Path], Path]


def S(dirname: str, frontmatter_text: str, body: str = BODY, filename: str = "SKILL.md", **kwargs) -> Builder:
    """Builder for a single skill; validation runs on the skill directory."""

    return lambda tmp_path: make_skill(tmp_path, dirname, frontmatter_text, body, filename, **kwargs)


def R(*builders: Builder) -> Builder:
    """Builder for a collection: every builder runs under tmp_path and the root is validated."""

    def build(tmp_path: Path) -> Path:
        for builder in builders:
            builder(tmp_path)
        return tmp_path

    return build


def OY(name: str, text: str, filename: str = "openai.yaml", dirname: str = "agents") -> Builder:
    """Builder for a skill with an ``agents/openai.yaml`` sidecar."""

    def build(tmp_path: Path) -> Path:
        skill_dir = make_skill(tmp_path, name, fm(name))
        (skill_dir / dirname).mkdir()
        (skill_dir / dirname / filename).write_text(text, encoding="utf-8")
        return skill_dir

    return build


def with_files(builder: Builder, files: dict[str, str]) -> Builder:
    """Add extra files (relative paths) to the skill a builder creates."""

    def build(tmp_path: Path) -> Path:
        skill_dir = builder(tmp_path)
        for rel, text in files.items():
            target = skill_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        return skill_dir

    return build


def dangling_symlink(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "dangling-skill"
    skill_dir.mkdir()
    os.symlink("missing.md", skill_dir / "SKILL.md")
    return skill_dir


def symlinked_file(tmp_path: Path) -> Path:
    """``real-skill`` plus ``link-skill`` whose SKILL.md is a symlink to the real file."""

    real = make_skill(tmp_path, "real-skill", fm("real-skill"))
    link = tmp_path / "link-skill"
    link.mkdir()
    os.symlink(real / "SKILL.md", link / "SKILL.md")
    return tmp_path


def skill_md_is_a_directory(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "dir-skill"
    (skill_dir / "SKILL.md").mkdir(parents=True)
    return skill_dir


def nested_skill_md(tmp_path: Path) -> Path:
    outer = make_skill(tmp_path, "outer-skill", fm("outer-skill"))
    make_skill(outer, "references", "name: inner\ndescription: nested file")
    return tmp_path


def case_variant_only(tmp_path: Path) -> Path:
    make_skill(tmp_path, "variant-skill", fm("variant-skill"), filename="Skill.md")
    return tmp_path


def uppercase_extension_only(tmp_path: Path) -> Path:
    make_skill(tmp_path, "upper-ext", fm("upper-ext"), filename="SKILL.MD")
    return tmp_path


def hidden_skill_in_a_collection(tmp_path: Path) -> Path:
    make_skill(tmp_path, ".hidden-skill", fm("hidden-skill"))
    make_skill(tmp_path, "visible-skill", fm("visible-skill"))
    return tmp_path


def directory_without_a_skill_file(tmp_path: Path) -> Path:
    target = tmp_path / "no-skill-here"
    target.mkdir()
    (target / "README.md").write_text("# not a skill\n", encoding="utf-8")
    return target


def many_directories(tmp_path: Path) -> Path:
    skill_dir = make_skill(tmp_path, "many-skill", fm("many-skill"))
    for i in range(vs.CODEX_MAX_DIRS_PER_ROOT):
        (skill_dir / "data" / str(i)).mkdir(parents=True)
    return skill_dir


def plugin_manifest(tmp_path: Path) -> Path:
    skill_dir = make_skill(tmp_path, "plugin-skill", fm("plugin-skill-" + "x" * 25))
    (skill_dir / ".claude-plugin").mkdir()
    (skill_dir / ".claude-plugin" / "plugin.json").write_text(json.dumps({"name": "p" * 100}), encoding="utf-8")
    return skill_dir


def budget_catalog(tmp_path: Path) -> Path:
    for i in range(12):
        name = f"budget-{i:02d}"
        make_skill(tmp_path, name, fm(name, "Use when " + "x" * 1490))
    return tmp_path


def not_a_skill_file(tmp_path: Path) -> Path:
    path = tmp_path / "README.md"
    path.write_text("# not a skill\n", encoding="utf-8")
    return path


BRACES_40 = "{" + ",".join(str(i) for i in range(40)) + "}"
BIG_BODY = "# Title\n" + "word " * 5000 + "\n"
NBSP = "\xa0"
LIGATURE_NAME = "ﬁle-skill"  # NFKC -> file-skill
CAFE = "café"


@dataclasses.dataclass(frozen=True)
class Case:
    rule: str
    severity: str
    build: Builder


CASES: list[Case] = [
    # spec ----------------------------------------------------------------
    Case("spec-discovery-skill-md-required", "error", directory_without_a_skill_file),
    Case("spec-discovery-skill-md-required", "warning", nested_skill_md),
    Case("spec-discovery-case-insensitive-filesystem", "error", case_variant_only),
    Case("spec-discovery-exact-filename-for-clients", "warning", S("lower-skill", fm("lower-skill"), filename="skill.md")),
    Case("spec-discovery-name-collisions", "error", R(
        lambda t: make_skill(t, "dup-skill", fm("dup-skill"), parent=t / "domain-a"),
        lambda t: make_skill(t, "dup-skill", fm("dup-skill"), parent=t / "domain-b"),
    )),
    Case("spec-discovery-symlink-dir-name", "error", dangling_symlink),
    Case("spec-fm-must-start-with-dashes", "error", S("bom-skill", "", raw=b"\xef\xbb\xbf" + f"---\n{fm('bom-skill')}\n---\n{BODY}".encode())),
    Case("spec-fm-closing-delimiter", "error", S("no-close", "", raw=f"---\n{fm('no-close')}\n# body\n")),
    Case("spec-fm-triple-dash-truncates", "error", S("dash-skill", fm("dash-skill", "part one --- part two. Use when needed"))),
    Case("spec-fm-delimiter-leniency-vs-clients", "warning", S("oc-skill", "", raw=f"--- # note\n{fm('oc-skill')}\n---\n{BODY}")),
    Case("spec-fm-must-be-mapping", "error", S("seq-skill", "", raw=f"---\n- a\n- b\n---\n{BODY}")),
    Case("spec-fm-no-flow-style", "error", S("flow-skill", fm("flow-skill", extra="metadata: {author: x}"))),
    Case("spec-fm-no-anchors-aliases", "error", S("anchor-skill", fm("anchor-skill", f"&d {DESC}", "license: *d"))),
    Case("spec-fm-no-tags", "error", S("tag-skill", fm("tag-skill", f"!!str {DESC}"))),
    Case("spec-fm-no-duplicate-keys", "error", S("dup-key", fm("dup-key", extra=f"description: {DESC}"))),
    Case("spec-fm-tabs-structural", "error", S("tab-skill", fm("tab-skill", extra="metadata:\n\tauthor: x"))),
    Case("spec-fm-tabs-structural", "warning", S("tabq-skill", fm("tabq-skill", '"a\tb use when asked for summaries of files"'))),
    Case("spec-fm-scalars-are-strings", "warning", S("num-desc", fm("num-desc", "12345"))),
    Case("spec-fm-plain-scalar-reserved-starts", "error", S("pct-skill", fm("pct-skill", extra="compatibility: %x"))),
    Case("spec-fm-plain-scalar-reserved-starts", "warning", S("hash-skill", fm("hash-skill", "Use this for #1 priority files when asked"))),
    Case("spec-fm-quoted-scalars", "error", S("quote-skill", fm("quote-skill", '"say "hi" now when asked"'))),
    Case("spec-fm-block-scalars-and-multiline", "info", S("folded-skill", fm("folded-skill", ">\n  Summarise climate output.\n  Use when asked for summaries."))),
    Case("spec-fm-control-characters-crash", "error", S("ctrl-skill", fm("ctrl-skill", DESC + "\x01"))),
    Case("spec-fm-unicode-line-separators", "warning", S("nbsp-skill", fm("nbsp-skill", DESC + NBSP + "x"))),
    Case("spec-fm-utf8-required", "error", S("utf-skill", "", raw=b"---\nname: utf-skill\ndescription: caf\xe9 use when\n---\nbody\n")),
    Case("spec-fm-keys-case-and-hyphen-exact", "error", S("case-key", f"name: case-key\nDescription: {DESC}")),
    Case("spec-fm-sequence-indent-absorbed", "error", S("absorb-skill", fm("absorb-skill", extra="allowed-tools:\n  - Read\n   - Bash"))),
    Case("spec-fm-merge-keys", "warning", S("merge-skill", fm("merge-skill", extra="metadata:\n  <<:\n    a: b"))),
    Case("spec-keys-allowed-set", "error", S("dmi-skill", fm("dmi-skill", extra="disable-model-invocation: true"))),
    Case("spec-keys-required-name-description", "error", S("no-desc", "name: no-desc")),
    Case("spec-keys-extensions-belong-in-metadata", "info", S("dmi-skill", fm("dmi-skill", extra="disable-model-invocation: true"))),
    Case("spec-keys-empty-optional-values", "warning", S("empty-lic", fm("empty-lic", extra="license:"))),
    Case("spec-name-nonempty-string", "error", S("empty-name", f'name: ""\ndescription: {DESC}')),
    Case("spec-name-length-1-64", "error", S("n" * 65, fm("n" * 65))),
    Case("spec-name-lowercase", "error", S("Upper-Skill", fm("Upper-Skill"))),
    Case("spec-name-charset-unicode-alnum-hyphen", "error", S("under_score", fm("under_score"))),
    Case("spec-name-charset-unicode-alnum-hyphen", "warning", S(CAFE, fm(CAFE))),
    Case("spec-name-no-edge-hyphens", "error", S("foo-", fm("foo-"))),
    Case("spec-name-no-consecutive-hyphens", "error", S("foo--bar", fm("foo--bar"))),
    Case("spec-name-must-match-directory", "error", S("dir-name", fm("other-name"))),
    Case("spec-name-nfkc-normalization", "warning", S("file-skill", fm(LIGATURE_NAME))),
    Case("spec-desc-nonempty-string", "error", S("empty-desc", "name: empty-desc\ndescription:")),
    Case("spec-desc-max-1024", "error", S("long-desc", fm("long-desc", "x" * 1025))),
    Case("spec-desc-block-scalar-trailing-newline", "error", S("block-1024", fm("block-1024", ">\n  " + "u" * 1024))),
    Case("spec-desc-content-guidance", "warning", S("vague-skill", fm("vague-skill", "Helps with PDFs."))),
    Case("spec-desc-unquoted-colon-space", "error", S("colon-skill", fm("colon-skill", "Use when: the user wants summaries"))),
    Case("spec-compat-string-max-500", "error", S("compat-501", fm("compat-501", extra="compatibility: " + "c" * 501))),
    Case("spec-compat-string-max-500", "warning", S("compat-empty", fm("compat-empty", extra="compatibility:"))),
    Case("spec-license-freeform", "warning", S("list-lic", fm("list-lic", extra="license:\n  - MIT"))),
    Case("spec-metadata-string-map-coercion", "error", S("meta-text", fm("meta-text", extra="metadata: text"))),
    Case("spec-allowed-tools-space-separated-string", "warning", S("comma-tools", fm("comma-tools", extra="allowed-tools: Bash, Read"))),
    Case("spec-body-unrestricted-may-be-empty", "warning", S("empty-body", fm("empty-body"), body="")),
    Case("spec-body-under-500-lines", "warning", S("long-body", fm("long-body"), body="# T\n" + "line\n" * 499)),
    Case("spec-body-under-5000-tokens", "warning", S("big-body", fm("big-body"), body=BIG_BODY)),
    Case("spec-files-relative-paths-from-root", "warning", S("missing-ref", fm("missing-ref"), body="See [x](references/missing.md).\n")),
    Case("spec-files-one-level-deep", "info", with_files(S("deep-ref", fm("deep-ref"), body="See [x](references/sub/x.md).\n"), {"references/sub/x.md": "x"})),
    Case("spec-sem-body-tokens-not-interpreted", "info", S("args-skill", fm("args-skill"), body="Use $ARGUMENTS here.\n")),
    # claude-code ---------------------------------------------------------
    Case("claude-code-discovery-entry-layout", "error", S("lower-skill", fm("lower-skill"), filename="skill.md")),
    Case("claude-code-discovery-dot-dirs", "warning", hidden_skill_in_a_collection),
    Case("claude-code-discovery-reserved-synced", "error", S("synced", fm("synced"))),
    Case("claude-code-discovery-size-limit", "error", S("huge-skill", fm("huge-skill"), body="# T\n" + "w" * (vs.CLAUDE_SIZE_LIMIT + 100) + "\n")),
    Case("claude-code-discovery-symlinks", "error", dangling_symlink),
    Case("claude-code-discovery-symlinks", "warning", symlinked_file),
    Case("claude-code-fm-opening-delimiter", "error", S("no-fm", "", raw="# Just a body\nno frontmatter\n")),
    Case("claude-code-fm-closing-delimiter-first-occurrence", "error", S("dash-skill", fm("dash-skill", "part one --- part two. Use when needed"))),
    Case("claude-code-fm-missing-closing-delimiter", "error", S("no-close", "", raw=f"---\n{fm('no-close')}\n# body\n")),
    Case("claude-code-fm-yaml-parser-and-repair", "warning", S("colon-skill", fm("colon-skill", "Use when: the user wants summaries"))),
    Case("claude-code-fm-yaml-parser-and-repair", "error", S("unterm-skill", fm("unterm-skill", '"unterminated use when'))),
    Case("claude-code-fm-mapping-required", "error", S("seq-skill", "", raw=f"---\n- a\n- b\n---\n{BODY}")),
    Case("claude-code-fm-anchor-stripped", "error", S("anchor-skill", fm("anchor-skill", f"&d {DESC}", "license: *d"))),
    Case("claude-code-fm-anchor-stripped", "warning", S("merge-skill", fm("merge-skill", extra="metadata:\n  <<:\n    a: b"))),
    Case("claude-code-fm-duplicate-keys-last-wins", "error", S("dup-key", fm("dup-key", extra=f"description: {DESC}"))),
    Case("claude-code-fm-keys-exact-case", "warning", S("case-key", f"name: case-key\nDescription: {DESC}")),
    Case("claude-code-fm-unknown-keys-ignored", "info", S("req-skill", fm("req-skill", extra="requires: x"))),
    Case("claude-code-fm-inline-comment-truncation", "error", S("hash-skill", fm("hash-skill", "Use this for #1 priority files when asked"))),
    Case("claude-code-fm-crlf-bom-tabs", "warning", S("stray-cr", "", raw=f"---\nname: stray-cr\ndescription: Use when the value\rcontinues.\n---\n{BODY}")),
    Case("claude-code-name-from-directory", "warning", S("dir-name", fm("other-name"))),
    Case("claude-code-name-from-directory", "error", R(S("alpha", fm("alpha")), S("beta", fm("alpha")))),
    Case("claude-code-name-case-sensitive-lookup", "warning", S("Upper-Skill", fm("Upper-Skill"))),
    Case("claude-code-name-no-format-validation", "error", S("bad(name)", fm("bad(name)"))),
    Case("claude-code-name-no-format-validation", "warning", S("under_score", fm("under_score"))),
    Case("claude-code-name-type", "warning", S("123", fm("123"))),
    Case("claude-code-name-precedence", "warning", S("doctor", fm("doctor"))),
    Case("claude-code-description-optional-fallback", "warning", S("no-desc", "name: no-desc")),
    Case("claude-code-description-empty-value", "warning", S("empty-desc", "name: empty-desc\ndescription:")),
    Case("claude-code-description-type", "error", S("list-desc", "name: list-desc\ndescription:\n  - a")),
    Case("claude-code-description-type", "warning", S("num-desc", fm("num-desc", "12345"))),
    Case("claude-code-description-listing-cap", "warning", S("cap-skill", fm("cap-skill", "d" * 1537))),
    Case("claude-code-description-listing-budget", "warning", budget_catalog),
    Case("claude-code-key-disable-model-invocation", "error", S("dmi-maybe", fm("dmi-maybe", extra="disable-model-invocation: maybe"))),
    Case("claude-code-key-user-invocable", "error", S("ui-maybe", fm("ui-maybe", extra="user-invocable: maybe"))),
    Case("claude-code-key-allowed-tools", "error", S("tools-num", fm("tools-num", extra="allowed-tools: 12"))),
    Case("claude-code-key-allowed-tools", "warning", S("tools-star", fm("tools-star", extra='allowed-tools: "*"'))),
    Case("claude-code-key-allowed-tools", "info", S("tools-unknown", fm("tools-unknown", extra="allowed-tools: Frob"))),
    Case("claude-code-key-disallowed-tools", "error", S("dtools-num", fm("dtools-num", extra="disallowed-tools: 12"))),
    Case("claude-code-key-disallowed-tools", "warning", S("dtools-star", fm("dtools-star", extra='disallowed-tools: "*"'))),
    Case("claude-code-key-disallowed-tools", "info", S("dtools-unknown", fm("dtools-unknown", extra="disallowed-tools: Frob"))),
    Case("claude-code-key-shell", "warning", S("shell-skill", fm("shell-skill", extra="shell: zsh"))),
    Case("claude-code-key-effort", "warning", S("effort-skill", fm("effort-skill", extra="effort: extreme"))),
    Case("claude-code-key-context-agent-background", "warning", S("agent-skill", fm("agent-skill", extra="agent: Explore"))),
    Case("claude-code-key-model", "warning", S("model-skill", fm("model-skill", extra="model:"))),
    Case("claude-code-key-paths", "error", S("paths-num", fm("paths-num", extra="paths: 12"))),
    Case("claude-code-key-paths", "warning", S("paths-mixed", fm("paths-mixed", extra='paths: "**, src/*.py"'))),
    Case("claude-code-key-paths", "info", S("paths-str", fm("paths-str", extra='paths: "src/**/*.py"'))),
    Case("claude-code-key-hooks", "error", S("hooks-top", fm("hooks-top", extra="PreToolUse: x"))),
    Case("claude-code-key-metadata-license-compatibility", "warning", S("meta-reuse", fm("meta-reuse", extra="metadata:\n  paths: x"))),
    Case("claude-code-key-scalar-coercions", "warning", S("version-list", fm("version-list", extra="version:\n  - 1"))),
    Case("claude-code-body-substitutions", "warning", S("plugin-var", fm("plugin-var"), body="Read ${CLAUDE_PLUGIN_ROOT}/x.\n")),
    Case("claude-code-body-substitutions", "info", S("unused-arg", fm("unused-arg", extra="arguments: source target"), body="Use $source only.\n")),
    Case("claude-code-body-inline-shell", "warning", S("shell-body", fm("shell-body"), body="Run !`git status` first.\n")),
    Case("claude-code-body-500-lines", "warning", S("long-body", fm("long-body"), body="# T\n" + "line\n" * 499)),
    Case("claude-code-body-compaction-budget", "warning", S("big-body", fm("big-body"), body=BIG_BODY)),
    Case("claude-code-files-supporting", "warning", S("missing-ref", fm("missing-ref"), body="See [x](references/missing.md).\n")),
    # codex ---------------------------------------------------------------
    Case("codex-disc-no-claude-skills-dir", "warning", S("claude-dir", fm("claude-dir"), body="See .claude/skills/x.\n")),
    Case("codex-disc-skill-filename-case-sensitive", "error", S("lower-skill", fm("lower-skill"), filename="skill.md")),
    Case("codex-disc-every-skill-md-is-a-skill", "warning", nested_skill_md),
    Case("codex-disc-hidden-dirs-pruned", "error", hidden_skill_in_a_collection),
    Case("codex-disc-symlinked-skill-md-ignored", "error", symlinked_file),
    Case("codex-disc-scan-truncation-limits", "error", many_directories),
    Case("codex-disc-plugin-manifest-namespacing", "warning", plugin_manifest),
    Case("codex-disc-plugin-manifest-namespacing", "error", plugin_manifest),
    Case("codex-disc-dedupe-by-path-only", "warning", R(S("one-skill", fm("shared")), S("two-skill", fm("shared")))),
    Case("codex-files-unreadable-or-directory", "error", skill_md_is_a_directory),
    Case("codex-fm-opening-delimiter", "error", S("bom-skill", "", raw=b"\xef\xbb\xbf" + f"---\n{fm('bom-skill')}\n---\n{BODY}".encode())),
    Case("codex-fm-closing-delimiter-and-nonempty-block", "error", S("empty-fm", "", raw=f"---\n---\n{BODY}")),
    Case("codex-fm-triple-dash-line-truncates", "error", S("dash-block", fm("dash-block", "|\n  first para use when\n  ---\n  second"))),
    Case("codex-fm-cr-only-rejected", "error", S("cr-skill", "", raw=f"---\n{fm('cr-skill')}\n---\n{BODY}".replace("\n", "\r"))),
    Case("codex-fm-must-be-yaml-mapping", "error", S("seq-skill", "", raw=f"---\n- a\n- b\n---\n{BODY}")),
    Case("codex-fm-yaml-dialect", "error", S("trailing-colon", f"name: trailing-colon:\ndescription: {DESC}")),
    Case("codex-fm-duplicate-keys-typed-fields", "error", S("dup-key", fm("dup-key", extra=f"description: {DESC}"))),
    Case("codex-fm-duplicate-keys-typed-fields", "warning", S("dup-lic", fm("dup-lic", extra="license: MIT\nlicense: MIT"))),
    Case("codex-fm-anchors-tags-stripped", "warning", S("anchor-skill", fm("anchor-skill", f"&d {DESC}", "license: *d"))),
    Case("codex-fm-colon-repair", "info", S("colon-skill", fm("colon-skill", "Use when: the user wants summaries"))),
    Case("codex-fm-flow-collections-coerced", "warning", S("flow-name", f"name: {{a: b}}\ndescription: {DESC}")),
    Case("codex-keys-ignored-values-must-parse", "error", S("glob-skill", fm("glob-skill", extra="paths: **/*.py"))),
    Case("codex-files-utf8-required", "error", S("utf-skill", "", raw=b"---\nname: utf-skill\ndescription: caf\xe9 use when\n---\nbody\n")),
    Case("codex-name-fallback-to-directory", "warning", S("nameless", f"description: {DESC}")),
    Case("codex-name-max-64-chars", "error", S("n" * 65, fm("n" * 65))),
    Case("codex-name-whitespace-collapsed", "warning", S("space-name", fm('"space  name"'))),
    Case("codex-name-scalar-coercion", "warning", S("123", fm("123"))),
    Case("codex-name-scalar-coercion", "error", S("list-name", f"name:\n  - a\ndescription: {DESC}")),
    Case("codex-name-charset-unrestricted", "warning", S("Upper-Skill", fm("Upper-Skill"))),
    Case("codex-name-directory-match-not-required", "warning", S("dir-name", fm("other-name"))),
    Case("codex-name-mentionable-charset", "warning", S(CAFE, fm(CAFE))),
    Case("codex-name-env-var-blocklist", "warning", S("path", fm("path"))),
    Case("codex-name-colon-space", "warning", S("space-name", fm('"space name"'))),
    Case("codex-desc-required-nonempty", "error", S("no-desc", "name: no-desc")),
    Case("codex-desc-must-be-scalar", "error", S("list-desc", "name: list-desc\ndescription:\n  - a")),
    Case("codex-desc-must-be-scalar", "warning", S("num-desc", fm("num-desc", "12345"))),
    Case("codex-desc-collapsed-to-one-line", "info", S("literal-skill", fm("literal-skill", "|\n  Summarise climate output.\n  Use when asked for summaries."))),
    Case("codex-desc-no-load-limit-but-1024-in-catalog", "warning", S("long-desc", fm("long-desc", "x" * 1025))),
    Case("codex-desc-catalog-budget", "warning", budget_catalog),
    Case("codex-desc-angle-brackets", "warning", S("angle-skill", fm("angle-skill", "Use when handling <pdf> files"))),
    Case("codex-desc-todo-placeholder", "warning", S("todo-skill", fm("todo-skill", '"[TODO: write this] use when"'))),
    Case("codex-keys-only-name-description-short-description", "warning", S("paths-str", fm("paths-str", extra='paths: "src/**/*.py"'))),
    Case("codex-keys-only-name-description-short-description", "info", S("req-skill", fm("req-skill", extra="requires: x"))),
    Case("codex-key-disable-model-invocation-ignored", "warning", S("dmi-skill", fm("dmi-skill", extra="disable-model-invocation: true"))),
    Case("codex-key-disable-model-invocation-ignored", "info", S("dmi-false", fm("dmi-false", extra="disable-model-invocation: false"))),
    Case("codex-key-metadata-short-description", "info", S("short-skill", fm("short-skill", extra="metadata:\n  short-description: brief"))),
    Case("codex-key-metadata-short-description", "warning", S("short-us", fm("short-us", extra="metadata:\n  short_description: brief"))),
    Case("codex-key-metadata-must-be-mapping", "error", S("meta-text", fm("meta-text", extra="metadata: text"))),
    Case("codex-body-explicit-injection-no-cap", "warning", S("args-skill", fm("args-skill"), body="Use $ARGUMENTS here.\n")),
    Case("codex-body-explicit-injection-no-cap", "info", S("big-body", fm("big-body"), body=BIG_BODY)),
    Case("codex-body-size-recommendations", "warning", S("long-body", fm("long-body"), body="# T\n" + "line\n" * 499)),
    Case("codex-files-openai-yaml-fail-open", "warning", OY("oy-bad", "interface: [\n")),
    Case("codex-files-openai-yaml-exact-case", "warning", OY("oy-case", "interface:\n  display_name: x\n", filename="OpenAI.yaml")),
    Case("codex-files-openai-yaml-allow-implicit-invocation", "info", OY("oy-policy", "policy:\n  allow_implicit_invocation: false\n")),
    Case("codex-files-openai-yaml-allow-implicit-invocation", "warning", OY("oy-policy-str", 'policy:\n  allow_implicit_invocation: "no"\n')),
    Case("codex-files-openai-yaml-interface-constraints", "warning", OY("oy-iface", "interface:\n  display_name: " + "d" * 65 + "\n")),
    Case("codex-files-openai-yaml-dependencies", "warning", OY("oy-deps", "dependencies:\n  tools:\n    - type: mcp\n")),
    # antigravity ---------------------------------------------------------
    Case("antigravity-workspace-root", "warning", S("claude-dir", fm("claude-dir"), body="See .claude/skills/x.\n")),
    Case("antigravity-skill-file-name", "warning", S("lower-skill", fm("lower-skill"), filename="skill.md")),
    Case("antigravity-symlinks-followed", "error", dangling_symlink),
    Case("antigravity-dot-dirs-discovered", "warning", hidden_skill_in_a_collection),
    Case("antigravity-skill-file-name", "error", uppercase_extension_only),
    Case("antigravity-frontmatter-delimiters", "error", S("dash-skill", fm("dash-skill", "part one --- part two. Use when needed"))),
    Case("antigravity-frontmatter-delimiters", "warning", S("closer-text", "", raw=f"---\n{fm('closer-text')}\n--- end\n{BODY}")),
    Case("antigravity-fm-indented-block-dropped", "error", S("indented", "", raw=f"---\n  name: indented\n  description: {DESC}\n---\n{BODY}")),
    Case("antigravity-yaml-parser-go-yaml-v3", "error", S("at-skill", fm("at-skill", extra="license: @x"))),
    Case("antigravity-yaml-duplicate-keys", "error", S("dup-key", fm("dup-key", extra=f"description: {DESC}"))),
    Case("antigravity-yaml-tabs", "error", S("tab-skill", fm("tab-skill", extra="metadata:\n\tauthor: x"))),
    Case("antigravity-yaml-type-errors", "error", S("list-desc", "name: list-desc\ndescription:\n  - a")),
    Case("antigravity-desc-plain-scalar-hazards", "error", S("colon-skill", fm("colon-skill", "Use when: the user wants summaries"))),
    Case("antigravity-desc-plain-scalar-hazards", "warning", S("hash-skill", fm("hash-skill", "Use this for #1 priority files when asked"))),
    Case("antigravity-files-utf8-required", "error", S("nul-skill", "", raw=f"---\n{fm('nul-skill')}\n---\nbody\x00\n")),
    Case("antigravity-unknown-keys-ignored", "warning", S("ui-skill", fm("ui-skill", extra="user-invocable: true"))),
    Case("antigravity-unknown-keys-ignored", "info", S("req-skill", fm("req-skill", extra="requires: x"))),
    Case("antigravity-keys-case-sensitive", "error", S("case-key", f"name: case-key\nDescription: {DESC}")),
    Case("antigravity-keys-case-sensitive", "warning", S("us-key", fm("us-key", extra="disable_model_invocation: true"))),
    Case("antigravity-name-optional-defaults-to-file-stem", "error", S("nameless", f"description: {DESC}")),
    Case("antigravity-name-not-validated", "warning", S("Upper-Skill", fm("Upper-Skill"))),
    Case("antigravity-name-whitespace-preserved", "warning", S("space-name", fm('"space name"'))),
    Case("antigravity-name-need-not-match-directory", "warning", S("dir-name", fm("other-name"))),
    Case("antigravity-name-dedup", "error", R(S("n1", f"description: {DESC}"), S("n2", f"description: {DESC}"))),
    Case("antigravity-slash-command", "warning", S(CAFE, fm(CAFE))),
    Case("antigravity-description-optional-degrades", "error", S("no-desc", "name: no-desc")),
    Case("antigravity-description-scalar-coercion", "warning", S("num-desc", fm("num-desc", "12345"))),
    Case("antigravity-description-no-length-limit", "warning", S("long-desc", fm("long-desc", "x" * 1025))),
    Case("antigravity-desc-literal-block-newlines", "warning", S("literal-skill", fm("literal-skill", "|\n  Summarise climate output.\n  Use when asked for summaries."))),
    Case("antigravity-trigger-fields", "info", S("vague-skill", fm("vague-skill", "Helps with PDFs."))),
    Case("antigravity-disable-model-invocation", "error", S("dmi-mixed", fm("dmi-mixed", extra="disable-model-invocation: yEs"))),
    Case("antigravity-disable-model-invocation", "error", S("dmi-map", fm("dmi-map", extra="disable-model-invocation:\n  a: b"))),
    Case("antigravity-disable-slash-command", "error", S("dsc-seq", fm("dsc-seq", extra="disable-slash-command:\n  - a"))),
    Case("antigravity-disable-model-invocation", "info", S("dmi-skill", fm("dmi-skill", extra="disable-model-invocation: true"))),
    Case("antigravity-disable-slash-command", "error", S("dsc-num", fm("dsc-num", extra="disable-slash-command: 1"))),
    Case("antigravity-disable-slash-command", "warning", S("dsc-skill", fm("dsc-skill", extra="disable-slash-command: true"))),
    Case("antigravity-metadata-mapping", "error", S("meta-icon", fm("meta-icon", extra="metadata:\n  icon:\n    - a"))),
    Case("antigravity-hidden-key-not-for-skills", "warning", S("hidden-key", fm("hidden-key", extra="hidden: true"))),
    Case("antigravity-body-verbatim", "warning", S("args-skill", fm("args-skill"), body="Use $ARGUMENTS here.\n")),
    Case("antigravity-body-verbatim", "info", S("tools-body", fm("tools-body"), body="Use the Skill tool.\n")),
    Case("antigravity-body-size-guidance", "warning", S("long-body", fm("long-body"), body="# T\n" + "line\n" * 499)),
    Case("antigravity-supporting-files", "warning", S("missing-ref", fm("missing-ref"), body="See [x](references/missing.md).\n")),
    # cursor --------------------------------------------------------------
    Case("cursor-discovery-filename-case", "error", S("lower-skill", fm("lower-skill"), filename="skill.md")),
    Case("cursor-discovery-walk", "error", S("build", fm("build"))),
    Case("cursor-discovery-walk", "warning", nested_skill_md),
    Case("cursor-discovery-hidden-and-codex-system-prune", "warning", S("skill-creator", fm("skill-creator"))),
    Case("cursor-discovery-symlinks", "error", dangling_symlink),
    Case("cursor-discovery-symlinks", "warning", symlinked_file),
    Case("cursor-semantics-third-party-dirs-default-on", "warning", S("claude-dir", fm("claude-dir"), body="See .claude/skills/x.\n")),
    Case("cursor-frontmatter-parser", "error", skill_md_is_a_directory),
    Case("cursor-frontmatter-no-leading-whitespace", "error", S("no-fm", "", raw="# Just a body\nno frontmatter\n")),
    Case("cursor-frontmatter-four-dashes-not-frontmatter", "error", S("four-dash", "", raw=f"----\n{fm('four-dash')}\n---\n{BODY}")),
    Case("cursor-frontmatter-four-dashes-not-frontmatter", "warning", S("closer-text", "", raw=f"---\n{fm('closer-text')}\n--- end\n{BODY}")),
    Case("cursor-frontmatter-missing-closing", "error", S("no-close", "", raw=f"---\n{fm('no-close')}\n# body\n")),
    Case("cursor-frontmatter-first-closing-anywhere", "error", S("doc-end", fm("doc-end", extra="..."))),
    Case("cursor-frontmatter-cr-only", "error", S("cr-skill", "", raw=f"---\n{fm('cr-skill')}\n---\n{BODY}".replace("\n", "\r"))),
    Case("cursor-frontmatter-language-tag", "error", S("toml-skill", "", raw=f"---toml\n{fm('toml-skill')}\n---\n{BODY}")),
    Case("cursor-frontmatter-duplicate-keys", "error", S("dup-key", fm("dup-key", extra=f"description: {DESC}"))),
    Case("cursor-frontmatter-yaml-exception-effect", "error", S("local-tag", fm("local-tag", f"!foo {DESC}"))),
    Case("cursor-frontmatter-must-be-mapping", "error", S("seq-skill", "", raw=f"---\n- a\n- b\n---\n{BODY}")),
    Case("cursor-frontmatter-tabs", "error", S("tab-skill", fm("tab-skill", extra="metadata:\n\tauthor: x"))),
    Case("cursor-frontmatter-tabs", "warning", S("tabq-skill", fm("tabq-skill", '"a\tb use when asked for summaries of files"'))),
    Case("cursor-yaml-boolean-semantics", "warning", S("dmi-yes", fm("dmi-yes", extra="disable-model-invocation: yes"))),
    Case("cursor-keys-exact-spelling", "error", S("case-key", f"name: case-key\nDescription: {DESC}")),
    Case("cursor-keys-exact-spelling", "warning", S("us-key", fm("us-key", extra="disable_model_invocation: true"))),
    Case("cursor-description-required-string", "error", S("no-desc", "name: no-desc")),
    Case("cursor-description-truncated-1536", "warning", S("cap-skill", fm("cap-skill", "d" * 1537))),
    Case("cursor-description-prompt-budget", "info", S("long-desc", fm("long-desc", "x" * 1025))),
    Case("cursor-description-prompt-budget", "warning", budget_catalog),
    Case("cursor-description-yaml-hazards", "error", S("colon-skill", fm("colon-skill", "Use when: the user wants summaries"))),
    Case("cursor-description-yaml-hazards", "warning", S("hash-skill", fm("hash-skill", "Use this for #1 priority files when asked"))),
    Case("cursor-name-ignored-by-loader", "warning", S("dir-name", fm("other-name"))),
    Case("cursor-key-disable-model-invocation", "info", S("dmi-skill", fm("dmi-skill", extra="disable-model-invocation: true"))),
    Case("cursor-key-paths-globs", "warning", S("paths-num", fm("paths-num", extra="paths: 12"))),
    Case("cursor-key-paths-globs", "info", S("paths-str", fm("paths-str", extra='paths: "src/**/*.py"'))),
    Case("cursor-key-paths-matching", "warning", S("paths-dot", fm("paths-dot", extra='paths: "./src/*.py"'))),
    Case("cursor-key-alwaysApply-turns-skill-into-rule", "warning", S("always-skill", fm("always-skill", extra="alwaysApply: true"))),
    Case("cursor-key-environments", "error", S("env-bad", fm("env-bad", extra="environments: locl"))),
    Case("cursor-key-environments", "warning", S("env-cloud", fm("env-cloud", extra="environments: cloud"))),
    Case("cursor-key-metadata-surfaces-scopedTo", "error", S("surf-bad", fm("surf-bad", extra="metadata:\n  surfaces: web"))),
    Case("cursor-key-metadata-surfaces-scopedTo", "warning", S("surf-cli", fm("surf-cli", extra="metadata:\n  surfaces: cli"))),
    Case("cursor-key-metadata-surfaces-scopedTo", "info", S("scoped", fm("scoped", extra="metadata:\n  scopedTo: a@b.c"))),
    Case("cursor-key-icon-color", "warning", S("color-skill", fm("color-skill", extra="color: pink"))),
    Case("cursor-keys-unknown-ignored", "warning", S("ui-skill", fm("ui-skill", extra="user-invocable: true"))),
    Case("cursor-keys-unknown-ignored", "info", S("req-skill", fm("req-skill", extra="requires: x"))),
    Case("cursor-body-model-visible-form", "warning", S("empty-body", fm("empty-body"), body="")),
    Case("cursor-body-model-visible-form", "info", S("tools-body", fm("tools-body"), body="Use the Skill tool.\n")),
    Case("cursor-body-size", "warning", S("body-100k", fm("body-100k"), body="# T\n" + "w" * (vs.CURSOR_ATTACH_CAP + 100) + "\n")),
]

# (rule, severity) pairs covered by dedicated tests below rather than by CASES.
EXTRA_COVERED: set[tuple[str, str]] = set()


# --------------------------------------------------------------------------- #
# Rule coverage: one positive case per (rule id, severity), one negative for all
# --------------------------------------------------------------------------- #


def test_every_registered_rule_has_a_case() -> None:
    registered = {(rid, sev) for rid, info in vs.RULES.items() for sev in info.severities}
    covered = {(c.rule, c.severity) for c in CASES} | EXTRA_COVERED
    assert registered - covered == set()
    unknown = {c.rule for c in CASES} - set(vs.RULES)
    assert unknown == set()


def test_case_severities_are_declared() -> None:
    for case in CASES:
        assert case.severity in vs.RULES[case.rule].severities, case


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c.rule}[{c.severity}]")
def test_rule_fires(case: Case, tmp_path: Path) -> None:
    target = case.build(tmp_path)
    profile = vs.RULES[case.rule].profile
    findings = validate(target, [profile])
    matching = [f for f in findings if f.rule_id == case.rule]
    assert matching, f"{case.rule} did not fire; got {sorted(rule_ids(findings))}"
    assert case.severity in {f.severity for f in matching}, {f.severity for f in matching}
    for finding in matching:
        assert finding.profile == profile
        assert finding.message.startswith(case.rule + ": ")
        assert finding.hint


@pytest.fixture(scope="module")
def clean_findings(tmp_path_factory: pytest.TempPathFactory) -> list[vs.Finding]:
    root = tmp_path_factory.mktemp("clean")
    skill_dir = make_skill(root, "clean-skill", fm("clean-skill"))
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "run.py").write_text("print('ok')\n", encoding="utf-8")
    (skill_dir / "SKILL.md").write_text(
        f"---\n{fm('clean-skill', extra='license: MIT')}\n---\n{BODY}\nRun `scripts/run.py` when asked.\n",
        encoding="utf-8",
    )
    return validate(skill_dir)


def test_clean_skill_has_no_findings(clean_findings: list[vs.Finding]) -> None:
    assert clean_findings == []


@pytest.mark.parametrize("rule_id", sorted(vs.RULES))
def test_clean_skill_is_silent_for(rule_id: str, clean_findings: list[vs.Finding]) -> None:
    assert rule_id not in rule_ids(clean_findings)


def test_antigravity_uppercase_extension_is_an_error(tmp_path: Path) -> None:
    """SKILL.MD is never discovered, so the profile branch needs a hand-built entry.

    Discovery reports the same directory through ``_discovery_findings``; that
    path is covered by ``test_case_variant_is_reported_by_every_profile``.
    """

    skill_dir = make_skill(tmp_path, "upper-ext", fm("upper-ext"), filename="SKILL.MD")
    entry = vs.SkillEntry(dir=skill_dir, skill_md=skill_dir / "SKILL.MD", filename="SKILL.MD", root=tmp_path)
    findings = vs.run_profiles([vs.load_skill(entry)], ["antigravity"], catalog=False)
    assert severities(findings, "antigravity-skill-file-name") == {"error"}


def test_case_variant_is_reported_by_every_profile(tmp_path: Path) -> None:
    """A directory whose only skill file is SKILL.MD must fail on every profile, not only spec."""

    target = uppercase_extension_only(tmp_path)
    for profile, rule_id in (
        ("spec", "spec-discovery-case-insensitive-filesystem"),
        ("claude-code", "claude-code-discovery-entry-layout"),
        ("codex", "codex-disc-skill-filename-case-sensitive"),
        ("antigravity", "antigravity-skill-file-name"),
        ("cursor", "cursor-discovery-filename-case"),
    ):
        findings = validate(target, [profile])
        assert rule_ids(findings) == {rule_id}, (profile, rule_ids(findings))
        assert severities(findings, rule_id) == {"error"}
    proc = run_cli(str(target / "upper-ext"), "--profile", "antigravity")
    assert proc.returncode == 1, proc.stdout
    assert "antigravity-skill-file-name" in proc.stdout


def test_lowercase_stem_variant_is_a_warning_for_antigravity(tmp_path: Path) -> None:
    make_skill(tmp_path, "variant-skill", fm("variant-skill"), filename="Skill.md")
    findings = validate(tmp_path, ["antigravity"])
    assert severities(findings, "antigravity-skill-file-name") == {"warning"}


def test_directory_without_a_skill_file_is_an_error(tmp_path: Path) -> None:
    """skills-ref rejects it with 'Missing required file: SKILL.md'; a silent pass hides a typo."""

    target = directory_without_a_skill_file(tmp_path)
    findings = validate(target, ["spec"])
    assert [(f.rule_id, f.severity) for f in findings] == [("spec-discovery-skill-md-required", "error")]
    proc = run_cli(str(target), "--profile", "spec")
    assert proc.returncode == 1
    assert "0 skills checked: 1 errors" in proc.stdout
    # a directory holding only SKILLS.md is the same miss
    (target / "SKILL.md.bak").write_text(f"---\n{fm('x')}\n---\n{BODY}", encoding="utf-8")
    assert rule_ids(validate(target, ["spec"])) == {"spec-discovery-skill-md-required"}
    # ... but an explicit path to a file that is not a skill file still is one too
    assert rule_ids(validate(not_a_skill_file(tmp_path), ["spec"])) == {"spec-discovery-skill-md-required"}


def test_hidden_skill_directory_is_reported_by_the_walk(tmp_path: Path) -> None:
    """The catalog walk skips dot-dirs, so the miss has to be reported instead."""

    target = hidden_skill_in_a_collection(tmp_path)
    discovery = vs.discover_skills([target])
    assert [d.name for d in discovery.hidden_skills] == [".hidden-skill"]
    assert [e.name for e in discovery.entries] == ["visible-skill"]
    findings = validate(target)
    hidden = [f for f in findings if f.skill == ".hidden-skill"]
    assert {(f.profile, f.rule_id, f.severity) for f in hidden} == {
        ("claude-code", "claude-code-discovery-dot-dirs", "warning"),
        ("codex", "codex-disc-hidden-dirs-pruned", "error"),
        ("antigravity", "antigravity-dot-dirs-discovered", "warning"),
        ("cursor", "cursor-discovery-walk", "error"),
    }


@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="root can read a mode-000 file")
def test_unreadable_skill_md_is_an_error_everywhere(tmp_path: Path) -> None:
    skill_dir = make_skill(tmp_path, "locked-skill", fm("locked-skill"))
    os.chmod(skill_dir / "SKILL.md", 0)
    try:
        findings = validate(skill_dir)
    finally:
        os.chmod(skill_dir / "SKILL.md", 0o644)
    assert {f.profile for f in findings if f.severity == "error"} == set(ALL)
    assert "codex-files-unreadable-or-directory" in rule_ids(findings)
    assert "spec-discovery-skill-md-required" in rule_ids(findings)


# --------------------------------------------------------------------------- #
# Targeted negatives (look-alikes of the hazards must stay silent)
# --------------------------------------------------------------------------- #


def test_crlf_is_accepted_everywhere(tmp_path: Path) -> None:
    raw = f"---\r\n{fm('crlf-skill')}\r\n---\r\n# T\r\nbody\r\n"
    findings = validate(make_skill(tmp_path, "crlf-skill", "", raw=raw))
    assert findings == []


def test_quoted_colon_and_hash_are_fine(tmp_path: Path) -> None:
    findings = validate(make_skill(tmp_path, "q-skill", fm("q-skill", '"Use when: the user wants #1 summaries of files"')))
    assert findings == []


def test_strip_chomped_block_scalar_has_no_trailing_newline_note(tmp_path: Path) -> None:
    findings = validate(make_skill(tmp_path, "strip-skill", fm("strip-skill", f">-\n  {DESC}")))
    assert findings == []


def test_exact_limits_are_not_exceeded(tmp_path: Path) -> None:
    name = "n" * 64
    body = "# T\n" + "line\n" * 494  # 500 lines in total: 5 frontmatter lines + heading + 494
    description = "Use when " + "x" * 1015  # exactly 1024 code points
    findings = validate(make_skill(tmp_path, name, fm(name, description, "compatibility: " + "c" * 500), body=body))
    assert [f for f in findings if f.severity != "info"] == []
    # the only note left is Cursor's "over 480 characters" budget advisory
    assert rule_ids(findings) == {"cursor-description-prompt-budget"}


def test_boolean_spellings_only_flag_the_right_harnesses(tmp_path: Path) -> None:
    findings = validate(make_skill(tmp_path, "yes-skill", fm("yes-skill", extra="disable-model-invocation: yes")))
    assert "claude-code-key-disable-model-invocation" not in rule_ids(findings)
    assert severities(findings, "antigravity-disable-model-invocation") == {"info"}
    assert "cursor-yaml-boolean-semantics" in rule_ids(findings)
    assert "spec-keys-allowed-set" in rule_ids(findings)


def test_quoted_glob_and_true_boolean_are_fine_outside_spec(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, "glob-skill", fm("glob-skill", extra='paths: "**/*.py"\ndisable-model-invocation: true'))
    findings = validate(skill)
    assert not [f for f in findings if f.severity == "error" and f.profile != "spec"]
    assert "codex-keys-ignored-values-must-parse" not in rule_ids(findings)


def test_existing_references_are_silent(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, "ref-skill", fm("ref-skill"), body="See [ref](references/guide.md) and scripts/run.py.\n")
    (skill / "references").mkdir()
    (skill / "references" / "guide.md").write_text("g", encoding="utf-8")
    (skill / "scripts").mkdir()
    (skill / "scripts" / "run.py").write_text("", encoding="utf-8")
    assert validate(skill) == []


@pytest.mark.parametrize(
    "value",
    ["true", "123", "null", "2024-01-01", "0x1f", "1:30", "yes"],
    ids=lambda v: v.replace(":", "_"),
)
def test_spec_reads_plain_scalars_as_text(tmp_path: Path, value: str) -> None:
    """strictyaml does no implicit typing, so `description: true` is the string 'true'."""

    skill = make_skill(tmp_path, "typed-desc", fm("typed-desc", value))
    findings = validate(skill, ["spec"])
    assert [f for f in findings if f.severity == "error"] == []
    assert "spec-desc-nonempty-string" not in rule_ids(findings)
    assert "spec-fm-scalars-are-strings" in rule_ids(findings)
    if re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", value):  # the ones that are also legal names
        name_skill = make_skill(tmp_path, value, f"name: {value}\ndescription: {DESC}")
        assert [f for f in validate(name_skill, ["spec"]) if f.severity == "error"] == []


def test_spec_comment_only_value_is_still_empty(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, "hash-desc", "name: hash-desc\ndescription: # note")
    assert severities(validate(skill, ["spec"]), "spec-desc-nonempty-string") == {"error"}


@pytest.mark.parametrize("value", ["", " 1.0", " yes", " ~", " 2024-01-01"])
def test_empty_or_untyped_compatibility_is_a_warning(tmp_path: Path, value: str) -> None:
    """skills-ref accepts every plain scalar here; only a sequence/mapping is an error."""

    skill = make_skill(tmp_path, "compat-skill", fm("compat-skill", extra=f"compatibility:{value}"))
    findings = validate(skill, ["spec"])
    assert [f for f in findings if f.severity == "error"] == []
    if not value.strip():
        assert severities(findings, "spec-compat-string-max-500") == {"warning"}
        assert "spec-keys-empty-optional-values" not in rule_ids(findings)


def test_compatibility_sequence_is_still_an_error(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, "compat-seq", fm("compat-seq", extra="compatibility:\n  - a"))
    assert severities(validate(skill, ["spec"]), "spec-compat-string-max-500") == {"error"}


def test_indented_closing_delimiter_is_not_a_missing_delimiter(tmp_path: Path) -> None:
    """skills-ref splits on the '---' substring, so an indented closer closes the block."""

    skill = make_skill(tmp_path, "indent-close", "", raw=f"---\n{fm('indent-close')}\n  ---\n\n{BODY}")
    findings = validate(skill, ["spec"])
    assert [f for f in findings if f.severity == "error"] == []
    assert rule_ids(findings) == {"spec-fm-delimiter-leniency-vs-clients"}


def test_substring_split_inside_a_value_still_truncates(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, "cut-value", "", raw=f"---\nname: cut-value\ndescription: {DESC}---tail\n\n{BODY}")
    assert severities(validate(skill, ["spec"]), "spec-fm-triple-dash-truncates") == {"error"}


def test_opening_delimiter_without_a_newline_keeps_the_first_key(tmp_path: Path) -> None:
    """skills-ref parses content.split('---', 2)[1], which includes '---name: foo'."""

    skill = make_skill(tmp_path, "run-on", "", raw=f"---name: run-on\ndescription: {DESC}\n---\n{BODY}")
    findings = validate(skill, ["spec"])
    assert [f for f in findings if f.severity == "error"] == []
    assert rule_ids(findings) == {"spec-fm-delimiter-leniency-vs-clients"}


def test_tab_inside_a_block_scalar_is_not_indentation(tmp_path: Path) -> None:
    """A tab in block-scalar content parses on the first pass in every harness."""

    skill = make_skill(tmp_path, "block-tab", fm("block-tab", extra="metadata:\n  snippet: |\n    build:\n    \tmake all"))
    findings = validate(skill)
    assert [f for f in findings if f.severity == "error"] == []
    assert severities(findings, "spec-fm-tabs-structural") == {"warning"}
    assert "claude-code-fm-crlf-bom-tabs" not in rule_ids(findings)
    assert "codex-fm-yaml-dialect" not in rule_ids(findings)
    assert "antigravity-yaml-tabs" not in rule_ids(findings)
    assert severities(findings, "cursor-frontmatter-tabs") == {"warning"}


def test_cr_only_line_endings_have_no_frontmatter_for_claude_code(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, "cr-skill", "", raw=f"---\n{fm('cr-skill')}\n---\n{BODY}".replace("\n", "\r"))
    findings = validate(skill)
    assert severities(findings, "claude-code-fm-opening-delimiter") == {"error"}
    assert "claude-code-fm-crlf-bom-tabs" not in rule_ids(findings)
    assert severities(findings, "codex-fm-cr-only-rejected") == {"error"}
    assert severities(findings, "cursor-frontmatter-cr-only") == {"error"}


def test_foreign_plugin_manifest_is_not_a_claude_dual_identity(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, "codex-manifest", fm("codex-manifest"))
    (skill / ".codex-plugin").mkdir()
    (skill / ".codex-plugin" / "plugin.json").write_text(json.dumps({"name": "p"}), encoding="utf-8")
    findings = validate(skill, ["claude-code"])
    assert "claude-code-files-supporting" not in rule_ids(findings)
    assert severities(validate(skill, ["codex"]), "codex-disc-plugin-manifest-namespacing") == {"warning"}


def test_plugin_manifest_above_the_skill_does_not_namespace_it(tmp_path: Path) -> None:
    """Only the skill directory is installed, so an ancestor manifest never reaches Codex."""

    (tmp_path / ".claude-plugin").mkdir()
    (tmp_path / ".claude-plugin" / "plugin.json").write_text(json.dumps({"name": "outer"}), encoding="utf-8")
    skill = make_skill(tmp_path, "nested-skill", fm("nested-skill"))
    for target in (tmp_path, skill):
        assert "codex-disc-plugin-manifest-namespacing" not in rule_ids(validate(target, ["codex"]))


def test_synced_only_matches_the_reserved_spelling(tmp_path: Path) -> None:
    assert "claude-code-discovery-reserved-synced" not in rule_ids(
        validate(make_skill(tmp_path, "sync_ed", fm("sync_ed")), ["claude-code"]))
    assert severities(validate(make_skill(tmp_path, "SyNced", fm("SyNced")), ["claude-code"]),
                      "claude-code-discovery-reserved-synced") == {"error"}


def test_empty_flow_mapping_says_what_it_is(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, "empty-map", "", raw=f"---\n{{}}\n---\n{BODY}")
    findings = [f for f in validate(skill, ["claude-code"]) if f.rule_id == "claude-code-fm-mapping-required"]
    assert len(findings) == 1 and "empty mapping" in findings[0].message
    assert "not a mapping" not in findings[0].message


def test_hidden_skills_are_left_out_of_the_codex_catalog_budget(tmp_path: Path) -> None:
    for i in range(12):
        name = f"budget-{i:02d}"
        skill = make_skill(tmp_path, name, fm(name, "Use when " + "x" * 1490))
        (skill / "agents").mkdir()
        (skill / "agents" / "openai.yaml").write_text("policy:\n  allow_implicit_invocation: false\n", encoding="utf-8")
    assert "codex-desc-catalog-budget" not in rule_ids(validate(tmp_path, ["codex"]))


def test_symlinked_skill_md_is_reported_at_the_link(tmp_path: Path) -> None:
    skill = tmp_path / "link-skill"
    skill.mkdir()
    (skill / "SKILL.md.src").write_text(f"---\n{fm('link-skill')}\n---\n{BODY}", encoding="utf-8")
    os.symlink(skill / "SKILL.md.src", skill / "SKILL.md")
    findings = [f for f in validate(skill, ["codex"]) if f.rule_id == "codex-disc-symlinked-skill-md-ignored"]
    assert findings and all(f.file.endswith("link-skill/SKILL.md") for f in findings), [f.file for f in findings]


def test_nested_skill_md_below_the_codex_depth_limit(tmp_path: Path) -> None:
    outer = make_skill(tmp_path, "deep-skill", fm("deep-skill"))
    make_skill(outer / "l2" / "l3" / "l4" / "l5" / "l6", "l7", "name: deep\ndescription: nested")
    findings = [f for f in validate(tmp_path, ["codex"]) if f.rule_id == "codex-disc-every-skill-md-is-a-skill"]
    assert len(findings) == 1 and "silently skips it" in findings[0].message
    shallow = make_skill(tmp_path, "shallow-skill", fm("shallow-skill"))
    make_skill(shallow, "references", "name: inner\ndescription: nested")
    inner = [f for f in validate(shallow, ["codex"]) if f.rule_id == "codex-disc-every-skill-md-is-a-skill"]
    assert len(inner) == 1 and "separate skill" in inner[0].message


def test_openai_yml_spelling_is_reported(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, "yml-ext", fm("yml-ext"))
    (skill / "agents").mkdir()
    (skill / "agents" / "openai.yml").write_text("policy:\n  allow_implicit_invocation: false\n", encoding="utf-8")
    assert severities(validate(skill, ["codex"]), "codex-files-openai-yaml-exact-case") == {"warning"}


@pytest.mark.parametrize("key", ["disable-model-invocation", "disable-slash-command"])
@pytest.mark.parametrize("block", ["\n  a: b", "\n  - a"])
def test_antigravity_block_booleans_drop_the_skill(tmp_path: Path, key: str, block: str) -> None:
    rule = f"antigravity-{key}"
    skill = make_skill(tmp_path, "block-bool", fm("block-bool", extra=f"{key}:{block}"))
    assert severities(validate(skill, ["antigravity"]), rule) == {"error"}


def test_antigravity_tolerates_a_tab_that_is_not_indentation(tmp_path: Path) -> None:
    """go-yaml only rejects tabs used for indentation; PyYAML rejects them anywhere."""

    skill = make_skill(tmp_path, "tab-sep", "", raw=f"---\nname: tab-sep\ndescription:\t{DESC}\n---\n{BODY}")
    assert rule_ids(validate(skill, ["antigravity"])) == set()
    indented = make_skill(tmp_path, "tab-ind", fm("tab-ind", extra="metadata:\n\tauthor: x"))
    assert severities(validate(indented, ["antigravity"]), "antigravity-yaml-tabs") == {"error"}


@pytest.mark.parametrize("written,shown", [("007", "007"), ("0.10", "0.10"), ("yes", "yes")])
def test_antigravity_description_keeps_its_literal_text(tmp_path: Path, written: str, shown: str) -> None:
    skill = make_skill(tmp_path, "lit-desc", fm("lit-desc", written))
    message = [f.message for f in validate(skill, ["antigravity"]) if f.rule_id == "antigravity-description-scalar-coercion"]
    assert message and f"'{shown}'" in message[0], message


def test_cursor_tolerates_the_tabs_js_yaml_tolerates(tmp_path: Path) -> None:
    after_colon = make_skill(tmp_path, "tab-colon", "", raw=f"---\nname: tab-colon\ndescription:\t{DESC}\n---\n{BODY}")
    findings = validate(after_colon, ["cursor"])
    assert severities(findings, "cursor-frontmatter-tabs") == {"warning"}
    assert "cursor-frontmatter-yaml-exception-effect" not in rule_ids(findings)
    item = make_skill(tmp_path, "tab-item", fm("tab-item", extra="paths:\n\t- a\n\t- b"))
    assert severities(validate(item, ["cursor"]), "cursor-frontmatter-yaml-exception-effect") == {"error"}


def test_cursor_null_paths_keeps_the_globs_fallback(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, "paths-null", fm("paths-null", extra="paths:\nglobs: lib/**"))
    findings = validate(skill, ["cursor"])
    assert "cursor-key-paths-globs" not in {f.rule_id for f in findings if f.severity == "warning"}
    assert severities(findings, "cursor-key-paths-globs") == {"info"}
    assert rule_ids(validate(make_skill(tmp_path, "paths-bare", fm("paths-bare", extra="paths:")), ["cursor"])) == set()


def test_cursor_disabled_environments(tmp_path: Path) -> None:
    local = make_skill(tmp_path, "dis-local", fm("dis-local", extra="disabled-environments: local"))
    assert severities(validate(local, ["cursor"]), "cursor-key-environments") == {"warning"}
    both = make_skill(tmp_path, "dis-both", fm("dis-both", extra="disabled-environments: local, cloud"))
    assert severities(validate(both, ["cursor"]), "cursor-key-environments") == {"error"}
    cloud = make_skill(tmp_path, "dis-cloud", fm("dis-cloud", extra="disabled-environments: cloud"))
    assert "cursor-key-environments" not in rule_ids(validate(cloud, ["cursor"]))


def test_cursor_reports_one_finding_per_defect(tmp_path: Path) -> None:
    tagged = make_skill(tmp_path, "local-tag", fm("local-tag", f"!custom {DESC}"))
    findings = [f for f in validate(tagged, ["cursor"]) if f.rule_id == "cursor-frontmatter-yaml-exception-effect"]
    assert len(findings) == 1, [f.message for f in findings]
    early = make_skill(tmp_path, "early-close", "", raw=f"---\nname: early-close\n--- note\ndescription: {DESC}\n---\n{BODY}")
    assert rule_ids(validate(early, ["cursor"])) == {"cursor-frontmatter-first-closing-anywhere"}


def test_positional_parameters_inside_a_fence_are_not_substitutions(tmp_path: Path) -> None:
    fenced = make_skill(tmp_path, "fenced-arg", fm("fenced-arg"), body="# T\n\n```bash\nawk '{print $4}'\n```\n")
    assert rule_ids(validate(fenced)) == set()
    plain = make_skill(tmp_path, "plain-arg", fm("plain-arg"), body="# T\n\nPass $1 to the script.\n")
    assert "spec-sem-body-tokens-not-interpreted" in rule_ids(validate(plain))
    in_fence = make_skill(tmp_path, "fenced-args", fm("fenced-args"), body="# T\n\n```bash\necho $ARGUMENTS\n```\n")
    assert "spec-sem-body-tokens-not-interpreted" in rule_ids(validate(in_fence))


def test_metadata_mapping_of_strings_is_fine(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, "meta-skill", fm("meta-skill", extra='metadata:\n  author: example-org\n  version: "1.0"'))
    assert validate(skill) == []


def test_symlinked_skill_directory_is_one_skill(tmp_path: Path) -> None:
    """unpack.sh installs directory symlinks; validating the link alone must be clean."""

    real = make_skill(tmp_path / "catalog", "link-skill", fm("link-skill"))
    install = tmp_path / "install"
    install.mkdir()
    os.symlink(real, install / "link-skill")
    discovery = vs.discover_skills([install / "link-skill"])
    assert [e.name for e in discovery.entries] == ["link-skill"]
    assert validate(install / "link-skill") == []


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def test_discovery_stops_at_the_first_skill_md(tmp_path: Path) -> None:
    outer = make_skill(tmp_path / "domain", "outer-skill", fm("outer-skill"))
    make_skill(outer, "references", "name: inner\ndescription: nested")
    discovery = vs.discover_skills([tmp_path])
    assert [e.dir for e in discovery.entries] == [outer]
    assert discovery.orphans == [] and discovery.case_variants == []


def test_discovery_skips_hidden_directories_in_a_walk(tmp_path: Path) -> None:
    make_skill(tmp_path, ".hidden-skill", fm(".hidden-skill"))
    make_skill(tmp_path / ".hidden-domain", "inner-skill", fm("inner-skill"))
    visible = make_skill(tmp_path / "domain", "visible-skill", fm("visible-skill"))
    assert [e.dir for e in vs.discover_skills([tmp_path]).entries] == [visible]
    # an explicit hidden path is still validated
    assert [e.name for e in vs.discover_skills([tmp_path / ".hidden-skill"]).entries] == [".hidden-skill"]


def test_discovery_accepts_a_path_to_the_skill_md_file(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, "file-skill", fm("file-skill"))
    discovery = vs.discover_skills([skill / "SKILL.md"])
    assert [(e.dir, e.filename) for e in discovery.entries] == [(skill, "SKILL.md")]


def test_discovery_accepts_lowercase_skill_md(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, "lower-skill", fm("lower-skill"), filename="skill.md")
    discovery = vs.discover_skills([tmp_path])
    assert [(e.dir, e.filename, e.skill_md.name) for e in discovery.entries] == [(skill, "skill.md", "skill.md")]


def test_discovery_reports_case_variants_and_orphans(tmp_path: Path) -> None:
    make_skill(tmp_path, "variant-skill", fm("variant-skill"), filename="Skill.md")
    readme = tmp_path / "README.md"
    readme.write_text("x", encoding="utf-8")
    discovery = vs.discover_skills([tmp_path, readme])
    assert discovery.entries == []
    assert [(d.name, v) for d, v in discovery.case_variants] == [("variant-skill", ["Skill.md"])]
    assert [p for p, _ in discovery.orphans] == [readme]


def test_discovery_walks_in_sorted_order_and_dedupes(tmp_path: Path) -> None:
    b = make_skill(tmp_path / "beta", "b-skill", fm("b-skill"))
    a = make_skill(tmp_path / "alpha", "a-skill", fm("a-skill"))
    discovery = vs.discover_skills([tmp_path, a, tmp_path / "alpha"])
    assert [e.dir for e in discovery.entries] == [a, b]


def test_discovery_raises_for_a_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        vs.discover_skills([tmp_path / "nope"])


def test_load_skill_records_file_facts(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, "facts-skill", fm("facts-skill", extra="license: MIT"), body="# H\n\ntext\n")
    ctx = vs.load_skill(vs.discover_skills([skill]).entries[0])
    assert ctx.name == "facts-skill"
    assert ctx.fm_present and ctx.parse_error is None
    assert ctx.data["name"] == "facts-skill" and ctx.data["license"] == "MIT"
    assert ctx.key_lines == {"name": 2, "description": 3, "license": 4}
    assert ctx.body_start_line == 6 and ctx.headings == [(6, "H")]
    assert not (ctx.has_bom or ctx.has_crlf or ctx.has_cr_only or ctx.has_nul or ctx.is_symlink)


# --------------------------------------------------------------------------- #
# Catalog-level checks
# --------------------------------------------------------------------------- #


def test_duplicate_directory_names_across_domains(tmp_path: Path) -> None:
    a = make_skill(tmp_path / "domain-a", "dup-skill", fm("dup-skill"))
    b = make_skill(tmp_path / "domain-b", "dup-skill", fm("dup-skill"))
    findings = validate(tmp_path)
    collisions = [f for f in findings if f.rule_id == "spec-discovery-name-collisions"]
    assert len(collisions) == 2 and {f.severity for f in collisions} == {"error"}
    assert {f.file for f in collisions} == {vs._display_path(a / "SKILL.md"), vs._display_path(b / "SKILL.md")}
    assert severities(findings, "antigravity-name-dedup") == {"error"}
    assert severities(findings, "codex-disc-dedupe-by-path-only") == {"warning"}


def test_nested_skill_md_is_reported_once_per_profile_that_cares(tmp_path: Path) -> None:
    findings = validate(nested_skill_md(tmp_path))
    assert rule_ids(findings) == {"spec-discovery-skill-md-required", "codex-disc-every-skill-md-is-a-skill", "cursor-discovery-walk"}
    assert {f.severity for f in findings} == {"warning"}
    assert all(f.file.endswith("references/SKILL.md") for f in findings)


def test_frontmatter_name_shadowing_another_directory(tmp_path: Path) -> None:
    make_skill(tmp_path, "alpha", fm("alpha"))
    make_skill(tmp_path, "beta", fm("alpha"))
    findings = validate(tmp_path, ["claude-code"])
    errors = [f for f in findings if f.rule_id == "claude-code-name-from-directory" and f.severity == "error"]
    assert [f.skill for f in errors] == ["beta"]


def test_changed_since_selection_skips_catalog_budget(tmp_path: Path) -> None:
    findings = validate(budget_catalog(tmp_path), full_catalog=False)
    assert "claude-code-description-listing-budget" not in rule_ids(findings)
    assert "codex-desc-catalog-budget" not in rule_ids(findings)


def test_emitter_rejects_unregistered_or_wrong_profile_rules(tmp_path: Path) -> None:
    ctx = vs.load_skill(vs.discover_skills([make_skill(tmp_path, "e-skill", fm("e-skill"))]).entries[0])
    emitter = vs.Emitter(ctx, "spec")
    with pytest.raises(KeyError):
        emitter.add("spec-not-a-rule", "error", "x", "y")
    with pytest.raises(ValueError):
        emitter.add("codex-desc-required-nonempty", "error", "x", "y")
    with pytest.raises(ValueError):
        emitter.add("spec-desc-max-1024", "info", "x", "y")


# --------------------------------------------------------------------------- #
# CLI: --list, --changed-since, formats, --summary, --strict, exit codes
# --------------------------------------------------------------------------- #


def test_cli_list_prints_name_tab_path(tmp_path: Path) -> None:
    b = make_skill(tmp_path / "domain-b", "b-skill", fm("b-skill"))
    a = make_skill(tmp_path / "domain-a", "a-skill", fm("a-skill"))
    proc = run_cli(str(tmp_path), "--list")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == f"a-skill\t{a.resolve()}\nb-skill\t{b.resolve()}\n"


def test_cli_changed_since(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    skills = repo / "skills"
    for name in ("a-skill", "b-skill", "c-skill", "d-skill"):
        make_skill(skills / "domain", name, fm(name))
    git("init", "-q", cwd=repo)
    git("add", ".", cwd=repo)
    git("commit", "-q", "-m", "base", cwd=repo)
    base = git("rev-parse", "HEAD", cwd=repo).strip()

    (skills / "domain" / "b-skill" / "SKILL.md").write_text(f"---\n{fm('b-skill')}\n---\n# changed\n", encoding="utf-8")
    make_skill(skills / "domain", "e-skill", fm("e-skill"))
    shutil.rmtree(skills / "domain" / "d-skill")
    git("add", "-A", cwd=repo)
    git("commit", "-q", "-m", "change", cwd=repo)

    proc = run_cli("skills", "--list", "--changed-since", base, cwd=repo)
    assert proc.returncode == 0, proc.stderr
    assert [line.split("\t")[0] for line in proc.stdout.splitlines()] == ["b-skill", "e-skill"]

    # untracked skills count as changed too
    make_skill(skills / "domain", "f-skill", fm("f-skill"))
    proc = run_cli("skills", "--list", "--changed-since", base, cwd=repo)
    assert [line.split("\t")[0] for line in proc.stdout.splitlines()] == ["b-skill", "e-skill", "f-skill"]

    proc = run_cli("skills", "--changed-since", base, "--format", "json", cwd=repo)
    doc = json.loads(proc.stdout)
    assert [s["name"] for s in doc["skills"]] == ["b-skill", "e-skill", "f-skill"]
    assert doc["summary"]["note"].startswith("validating 3 changed skill(s): b-skill, e-skill, f-skill")

    proc = run_cli("skills", "--changed-since", "HEAD", "--format", "text", cwd=repo)
    assert proc.returncode == 0
    assert proc.stdout.startswith("validating 1 changed skill(s): f-skill")

    git("add", "-A", cwd=repo)
    git("commit", "-q", "-m", "all", cwd=repo)
    proc = run_cli("skills", "--changed-since", "HEAD", cwd=repo)
    assert proc.returncode == 0 and proc.stdout == "no skills changed\n"


def test_cli_changed_since_outside_git_is_a_usage_error(tmp_path: Path) -> None:
    make_skill(tmp_path, "g-skill", fm("g-skill"))
    proc = run_cli(str(tmp_path), "--changed-since", "HEAD", cwd=tmp_path)
    assert proc.returncode == 2
    assert "not inside a git repository" in proc.stderr


def test_cli_text_format(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, "colon-skill", fm("colon-skill", "Use when: the user wants summaries"))
    proc = run_cli(str(skill), "--profile", "spec", "--no-color")
    assert proc.returncode == 1
    lines = proc.stdout.splitlines()
    assert lines[0].startswith("colon-skill  ") and lines[0].endswith("SKILL.md")
    assert lines[1] == "  [spec]"
    assert re.match(r"^    ERROR    spec-desc-unquoted-colon-space  \S+SKILL\.md:3  .*\[fix: .*\]$", lines[2]), lines[2]
    assert re.match(r"^1 skills checked: \d+ errors, \d+ warnings, \d+ infos \(spec: \d+E/\d+W/\d+I\)$", lines[-1]), lines[-1]


def test_cli_json_format(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, "colon-skill", fm("colon-skill", "Use when: the user wants summaries"))
    proc = run_cli(str(skill), "--profile", "spec", "--profile", "codex", "--format", "json")
    assert proc.returncode == 1
    doc = json.loads(proc.stdout)
    assert list(doc) == ["skills", "findings", "summary"]
    assert [s["name"] for s in doc["skills"]] == ["colon-skill"]
    assert list(doc["skills"][0]) == ["name", "path", "skill_md", "profiles"]
    assert list(doc["skills"][0]["profiles"]) == ["spec", "codex"]
    assert list(doc["findings"][0]) == ["profile", "rule_id", "severity", "skill", "file", "line", "message", "hint"]
    assert {f["rule_id"] for f in doc["findings"]} >= {"spec-desc-unquoted-colon-space", "codex-fm-colon-repair"}
    summary = doc["summary"]
    assert summary["skills"] == 1 and summary["profiles"] == ["spec", "codex"]
    assert summary["failing_level"] == "error" and summary["exit_code"] == 1
    assert summary["errors"] >= 1 and set(summary["by_profile"]) == {"spec", "codex"}


def test_cli_github_format(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, "colon-skill", fm("colon-skill", "Use when: the user wants summaries"))
    proc = run_cli(str(skill), "--profile", "spec", "--format", "github")
    assert proc.returncode == 1
    commands = [line for line in proc.stdout.splitlines() if line.startswith("::")]
    assert commands, proc.stdout
    pattern = r"^::error file=[^,]*SKILL\.md,line=3,title=spec-desc-unquoted-colon-space::spec-desc-unquoted-colon-space: .*\[fix: .*\]$"
    assert any(re.match(pattern, c) for c in commands), commands
    assert all(re.match(r"^::(error|warning|notice) file=", c) for c in commands)
    assert "1 skills checked:" in proc.stdout  # the text report precedes the commands


def test_cli_summary_file(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, "colon-skill", fm("colon-skill", "Use when: the user wants summaries"))
    summary = tmp_path / "summary.md"
    summary.write_text("existing\n", encoding="utf-8")
    proc = run_cli(str(skill), "--profile", "spec", "--profile", "cursor", "--summary", str(summary))
    assert proc.returncode == 1
    text = summary.read_text(encoding="utf-8")
    assert text.startswith("existing\n## Skill compatibility (spec, cursor; 1 skills)\n")
    assert re.search(r"^### spec - \d+ errors, \d+ warnings, \d+ infos$", text, re.MULTILINE)
    assert re.search(r"^### cursor - \d+ errors", text, re.MULTILINE)
    assert "| skill | errors | warnings | infos |" in text
    assert re.search(r"^\| `\S*colon-skill` \| \d+ \| \d+ \| \d+ \|$", text, re.MULTILINE)


def test_cli_strict_turns_warnings_into_failures(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, "comma-tools", fm("comma-tools", extra="allowed-tools: Bash, Read"))
    assert run_cli(str(skill), "--profile", "spec").returncode == 0
    proc = run_cli(str(skill), "--profile", "spec", "--strict", "--format", "json")
    assert proc.returncode == 1
    doc = json.loads(proc.stdout)
    assert doc["summary"]["failing_level"] == "warning" and doc["summary"]["errors"] == 0


def test_cli_exit_codes(tmp_path: Path) -> None:
    clean = make_skill(tmp_path, "clean-skill", fm("clean-skill"))
    assert run_cli(str(clean)).returncode == 0
    broken = make_skill(tmp_path, "no-desc", "name: no-desc")
    assert run_cli(str(broken)).returncode == 1
    assert run_cli(str(tmp_path / "missing")).returncode == 2
    assert run_cli(str(clean), "--profile", "bogus").returncode == 2


def test_cli_profile_all_and_repeatable(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, "p-skill", fm("p-skill"))
    doc = json.loads(run_cli(str(skill), "--profile", "all", "--format", "json").stdout)
    assert doc["summary"]["profiles"] == list(ALL)
    doc = json.loads(run_cli(str(skill), "--profile", "cursor", "--profile", "spec", "--format", "json").stdout)
    assert doc["summary"]["profiles"] == ["spec", "cursor"]  # canonical order


# --------------------------------------------------------------------------- #
# Shipped catalog and documentation
# --------------------------------------------------------------------------- #


def test_shipped_catalog_discovers_all_skills() -> None:
    discovery = vs.discover_skills([REPO_ROOT / "skills"])
    assert sorted(entry.name for entry in discovery.entries) == ["autogen-skill", "parallel-attention-skill", "primus-skill", "vllm-skill"]
    assert all(entry.filename == "SKILL.md" for entry in discovery.entries)
    assert discovery.orphans == [] and discovery.case_variants == []


def test_shipped_skills_have_no_spec_errors() -> None:
    for name in ("autogen-skill", "parallel-attention-skill", "primus-skill", "vllm-skill"):
        findings = validate(REPO_ROOT / "skills" / name, ["spec"])
        assert [f.message for f in findings if f.severity == "error"] == []


def test_cli_default_paths_cover_the_catalog() -> None:
    proc = run_cli("--profile", "spec", "--list", cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stderr
    names = [line.split("\t")[0] for line in proc.stdout.splitlines()]
    assert sorted(names) == ["autogen-skill", "parallel-attention-skill", "primus-skill", "vllm-skill"]


def test_rule_ids_are_namespaced_by_profile() -> None:
    for rid, info in vs.RULES.items():
        assert rid.startswith(info.profile + "-"), rid
