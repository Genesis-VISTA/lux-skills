"""Shared helpers for building temporary skill fixtures in tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_skill(path: Path, *, name: str | None = None, description: str, body: str = "Body\n") -> None:
    """Create a minimal SKILL.md for a test fixture."""

    skill_name = name or path.name
    if not body.endswith("\n"):
        body += "\n"
    write_text(
        path / "SKILL.md",
        "\n".join(
            [
                "---",
                f"name: {skill_name}",
                f"description: {description}",
                "---",
                body.rstrip("\n"),
                "",
            ]
        ),
    )


def run_validate_skills(*paths: str | Path) -> subprocess.CompletedProcess[str]:
    """Run the catalog skill validator on one or more explicit paths."""

    command = [sys.executable, str(ROOT / "tools" / "validate_skills.py"), *(str(path) for path in paths)]
    return subprocess.run(command, check=False, capture_output=True, text=True)
