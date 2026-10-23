#!/usr/bin/env python3
"""Validate repo-policy files for license and attribution consistency."""

from __future__ import annotations

import argparse
import difflib
from dataclasses import dataclass
from pathlib import Path

try:  # pragma: no cover - exercised via the script entry point
    from tools.repo_inventory import build_readme_contributor_names, build_readme_tree_lines
except ModuleNotFoundError:  # pragma: no cover - script execution path
    from repo_inventory import build_readme_contributor_names, build_readme_tree_lines


ROOT = Path(__file__).resolve().parents[1]

# These checks mirror the repo-policy checks of genesis-skills
# (https://gitlab.osti.gov/genesis/genesis-skills), which follow the BaseTemplate
# "Required Elements for ModCon Base Public Repositories" section. We verify the
# pieces this repo can enforce directly: README acknowledgment, CONTRIBUTING
# guidance, Code of Conduct, getting-started documentation, and the attribution
# and licence files of the skills.
REQUIRED_FILES = (
    "LICENSE",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "README.md",
    "NOTICE",
    "docs/getting_started.md",
    "skills/ATTRIBUTION.md",
    "skills/autogen-skill/LICENSE",
    "skills/primus-skill/LICENSE",
    "skills/vllm-skill/LICENSE",
)

CONTENT_CHECKS = (
    ("README.md", "](LICENSE)"),
    ("README.md", "](NOTICE)"),
    ("README.md", "## Acknowledgment"),
    (
        "README.md",
        "This work was supported by the U.S. Department of Energy (DOE), Office of Science, "
        "Office of Advanced Scientific Computing Research in alignment with DOE's Genesis Mission.",
    ),
    (
        "README.md",
        "The repository-level summary of individual skill and third-party licensing details is in [NOTICE](NOTICE);",
    ),
    (
        "CONTRIBUTING.md",
        "[Guidelines for AI/LLM-Assisted Contributions](#guidelines-for-ai-llm-assisted-contributions)",
    ),
    ("CONTRIBUTING.md", "### Guidelines for AI/LLM-Assisted Contributions"),
    ("NOTICE", "skills/ATTRIBUTION.md"),
    ("NOTICE", "skills/autogen-skill/LICENSE"),
    ("NOTICE", "skills/primus-skill/LICENSE"),
    ("NOTICE", "skills/vllm-skill/LICENSE"),
    ("skills/ATTRIBUTION.md", "Apache-2.0"),
)


@dataclass(frozen=True)
class Finding:
    path: str
    message: str


def _add_missing_file(findings: list[Finding], relpath: str) -> None:
    findings.append(Finding(relpath, "missing required file"))


def _normalized_contains(text: str, needle: str) -> bool:
    normalized_text = " ".join(text.split())
    normalized_needle = " ".join(needle.split())
    return normalized_needle in normalized_text


def _extract_code_block(text: str, heading: str) -> list[str] | None:
    lines = text.splitlines()
    try:
        heading_index = next(i for i, line in enumerate(lines) if line.strip() == heading)
    except StopIteration:
        return None

    for start in range(heading_index + 1, len(lines)):
        if lines[start].strip().startswith("```"):
            break
    else:
        return None

    block: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip().startswith("```"):
            return block
        block.append(line.rstrip())
    return None


def _extract_bullet_lines(text: str, heading: str) -> list[str] | None:
    lines = text.splitlines()
    try:
        heading_index = next(i for i, line in enumerate(lines) if line.strip() == heading)
    except StopIteration:
        return None

    bullets: list[str] = []
    for line in lines[heading_index + 1 :]:
        stripped = line.strip()
        if stripped.startswith("## "):
            break
        if stripped.startswith("- "):
            bullets.append(stripped)
    return bullets


def _normalize_tree_lines(lines: list[str]) -> list[str]:
    return [line.rstrip() for line in lines if line.strip()]


def _normalize_bullets(lines: list[str]) -> list[str]:
    return [line.strip() for line in lines if line.strip()]


def _compare_sections(
    findings: list[Finding],
    path: str,
    actual: list[str] | None,
    expected: list[str],
    section_name: str,
) -> None:
    if actual is None:
        findings.append(Finding(path, f"missing {section_name} section"))
        return

    actual_norm = _normalize_tree_lines(actual) if section_name == "repository structure" else _normalize_bullets(actual)
    expected_norm = _normalize_tree_lines(expected) if section_name == "repository structure" else _normalize_bullets(expected)
    if actual_norm == expected_norm:
        return

    diff = "\n".join(
        difflib.unified_diff(
            expected_norm,
            actual_norm,
            fromfile=f"expected {section_name}",
            tofile=f"actual {section_name}",
            lineterm="",
        )
    )
    findings.append(Finding(path, f"{section_name} section does not match current repository inventory:\n{diff}"))


def validate(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    cache: dict[str, str | None] = {}
    tracked_paths = set(REQUIRED_FILES) | {relpath for relpath, _ in CONTENT_CHECKS}

    for relpath in tracked_paths:
        path = root / relpath
        if path.is_file():
            cache[relpath] = path.read_text(encoding="utf-8")
        else:
            cache[relpath] = None
            if relpath in REQUIRED_FILES:
                _add_missing_file(findings, relpath)

    for relpath, needle in CONTENT_CHECKS:
        text = cache[relpath]
        if text is not None and not _normalized_contains(text, needle):
            findings.append(Finding(relpath, f"missing reference to {needle}"))

    readme = cache.get("README.md")
    if readme is not None:
        _compare_sections(
            findings,
            "README.md",
            _extract_code_block(readme, "## Repository Structure"),
            build_readme_tree_lines(root),
            "repository structure",
        )
        _compare_sections(
            findings,
            "README.md",
            _extract_bullet_lines(readme, "## Contributors"),
            [f"- {name}" for name in build_readme_contributor_names(root)],
            "contributors",
        )

    return findings


def _format_finding(finding: Finding, output_format: str) -> str:
    if output_format == "github":
        return f"::error file={finding.path},title=repo policy::{finding.message}"
    return f"{finding.path}: {finding.message}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate repo-policy files.")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root to validate (defaults to the current repo).",
    )
    parser.add_argument(
        "--format",
        choices=("text", "github"),
        default="text",
        help="Output format for findings.",
    )
    args = parser.parse_args()

    findings = validate(args.root)
    if not findings:
        print("repo policy: OK")
        return 0

    print(f"repo policy: {len(findings)} finding(s)")
    for finding in findings:
        print(_format_finding(finding, args.format))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
