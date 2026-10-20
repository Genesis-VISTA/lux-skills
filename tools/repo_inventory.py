#!/usr/bin/env python3
"""Helpers for validating the repository structure and provenance docs."""

from __future__ import annotations

import re
from pathlib import Path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _skill_names(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    names = {path.parent.name for path in root.rglob("SKILL.md") if path.is_file()}
    return sorted(names, key=str.casefold)


def _top_level_skill_dirs(skills_root: Path) -> list[Path]:
    if not skills_root.is_dir():
        return []

    candidates = [
        path
        for path in skills_root.iterdir()
        if path.is_dir() and not path.name.startswith(".") and any(child.is_file() for child in path.rglob("SKILL.md"))
    ]
    return sorted(candidates, key=lambda path: path.name.casefold())


def _skill_tree_line(skill_root: Path) -> str:
    direct_skill = skill_root / "SKILL.md"
    skill_files = [path for path in skill_root.rglob("SKILL.md") if path.is_file()]
    if len(skill_files) == 1 and skill_files[0] == direct_skill:
        return f"{skill_root.name}/SKILL.md"

    names = _skill_names(skill_root)
    return f"{skill_root.name}/  # {', '.join(names)} ({len(names)})"


def _people_sort_key(name: str) -> tuple[str, str]:
    parts = name.split()
    last = parts[-1] if parts else name
    return (last.casefold(), name.casefold())


def _extract_name(fragment: str) -> str:
    fragment = fragment.strip()
    fragment = re.sub(r"\s*\[.*?\]$", "", fragment)
    fragment = re.split(r"\s*[\(<]", fragment, maxsplit=1)[0].strip()
    return fragment


def _notice_contact(root: Path) -> str | None:
    notice = root / "NOTICE"
    if not notice.is_file():
        return None
    match = re.search(r"^Contact:\s*(.+?)\s*<", _read(notice), re.MULTILINE)
    if not match:
        return None
    return _extract_name(match.group(1))


def _attribution_authors(path: Path) -> list[str]:
    """Names from an ``**Authors:**`` bullet list (``- Name, Institution``)."""

    if not path.is_file():
        return []
    names: list[str] = []
    in_authors = False
    for line in _read(path).splitlines():
        if line.startswith("**Authors:**"):
            in_authors = True
            continue
        if in_authors:
            if line.startswith("- "):
                names.append(_extract_name(line[2:].split(",", 1)[0]))
            elif line.strip():
                break
    return names


def build_readme_tree_lines(root: Path) -> list[str]:
    """Return the canonical README repository-structure tree as lines."""

    skills = root / "skills"
    lines = [
        "lux-skills/",
        "├── unpack.sh                  # flatten skills into an agent's skills dir",
        "├── skills/",
    ]

    skill_roots = _top_level_skill_dirs(skills)
    for index, skill_root in enumerate(skill_roots):
        branch = "└──" if index == len(skill_roots) - 1 else "├──"
        lines.append(f"│   {branch} {_skill_tree_line(skill_root)}")

    lines.extend(
        [
            "├── CONTRIBUTING.md",
            "├── LICENSE",
            "├── NOTICE                     # third-party licensing and attribution inventory",
            "├── README.md",
            "└── skill_spec.md              # Agent Skills format specification",
        ]
    )
    return lines


def build_readme_contributor_names(root: Path) -> list[str]:
    """Return the canonical contributor list for the README.

    The NOTICE contact plus the ``**Authors:**`` list of
    ``skills/ATTRIBUTION.md``.
    """

    individuals: set[str] = set()

    contact = _notice_contact(root)
    if contact:
        individuals.add(contact)

    individuals.update(_attribution_authors(root / "skills" / "ATTRIBUTION.md"))

    return sorted(individuals, key=_people_sort_key)
