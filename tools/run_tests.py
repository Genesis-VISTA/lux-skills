#!/usr/bin/env python3
"""Run the repository's Python test suites.

The helper keeps the local test loop dependency-free:

- `python3 tools/run_tests.py` runs the standard `unittest` suites.
- `python3 tools/run_tests.py --coverage` runs the same suites under
  `trace` and prints a coverage summary without leaving report files behind.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUITES = ("tests",)


def _prepare_sys_path() -> None:
    """Ensure the repo root is importable."""

    os.chdir(ROOT)


def _cleanup_trace_artifacts() -> None:
    """Remove trace-generated ``.cover`` files from the repository tree."""

    for path in ROOT.rglob("*.cover"):
        path.unlink()


def _run_suite(start_dir: str, coverage: bool) -> None:
    """Run one test discovery root."""

    command = [sys.executable, "-m", "unittest", "discover", "-s", start_dir, "-q"]

    if coverage:
        command = [
            sys.executable,
            "-m",
            "trace",
            "--count",
            "--summary",
            "--ignore-dir",
            sys.base_prefix,
            "--module",
            "unittest",
            "discover",
            "-s",
            start_dir,
            "-q",
        ]

    completed = subprocess.run(command, check=False, cwd=ROOT)
    if coverage:
        _cleanup_trace_artifacts()
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the repo's Python test suites.")
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run the suites under trace and print a coverage summary.",
    )
    args = parser.parse_args()

    _prepare_sys_path()
    for suite in SUITES:
        _run_suite(suite, args.coverage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
