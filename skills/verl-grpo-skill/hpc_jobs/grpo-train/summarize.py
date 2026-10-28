#!/usr/bin/env python3
"""Summarize the tutorial's verl log without copying generated questions into VISTA outputs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
STEP_RE = re.compile(r"(?:^|\s)step:(\d+)\b")
METRICS = {
    "reward": "critic/score/mean",
    "response_tokens": "response_length/mean",
    "step_seconds": "timing_s/step",
}
METRIC_RE = {name: re.compile(rf"(?<!\S){re.escape(key)}:\s*({NUMBER})") for name, key in METRICS.items()}
VAL_RE = re.compile(rf"(?<!\S)(val-core/[^:\s]+/acc/mean@1):\s*({NUMBER})")
DRIVER_RE = re.compile(r"driver exit (\d+)")


def summarize(log: Path, mode: str, target: int, run: str, rc: int, workdir: Path) -> bool:
    steps: dict[int, dict[str, str]] = {}
    validation: dict[str, str] = {}
    rails = "not reported"
    gpu_check = "not reported"
    driver_rc = "not reported"
    with log.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if "nodes RAILS OK" in line:
                rails = line.strip()
            if "GPU_CHECK ALL OK" in line:
                gpu_check = "passed"
            if match := DRIVER_RE.search(line):
                driver_rc = match.group(1)
            if match := STEP_RE.search(line):
                step = int(match.group(1))
                steps[step] = {name: m.group(1) for name, regex in METRIC_RE.items() if (m := regex.search(line))}
            for key, value in VAL_RE.findall(line):
                validation[key] = value

    last = max(steps, default=0)
    print(f"mode={mode} run={run} target_step={target} last_step={last} train_exit={rc} driver_exit={driver_rc}")
    print(f"rails: {rails}; GPU check: {gpu_check}")
    print("step reward response_tokens step_seconds")
    for step, metrics in sorted(steps.items()):
        print(f"{step} {metrics.get('reward', '-')} {metrics.get('response_tokens', '-')} {metrics.get('step_seconds', '-')}")
    if validation:
        print("latest validation accuracy:")
        for key, value in sorted(validation.items()):
            print(f"  {key}: {value}")
    checkpoint = workdir / "work" / "ckpt" / run / f"global_step_{target}"
    if checkpoint.is_dir():
        print(f"checkpoint: {checkpoint}")
    elif mode != "smoke":
        print(f"checkpoint missing: {checkpoint}")
    ok = rc == 0 and last >= target and (mode == "smoke" or checkpoint.is_dir())
    print("VERDICT: OK" if ok else "VERDICT: FAILED")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--mode", choices=("smoke", "control", "pubonly"), required=True)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    return 0 if summarize(args.log, args.mode, args.target_step, args.run_name, args.exit_code, args.workdir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
