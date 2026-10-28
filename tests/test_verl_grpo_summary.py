"""Check that VISTA gets a trustworthy outcome from the verl GRPO log."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills" / "verl-grpo-skill" / "hpc_jobs" / "grpo-train" / "summarize.py"


class GrpoSummaryTests(unittest.TestCase):
    def run_summary(self, root: Path, mode: str, target: int, log_text: str) -> subprocess.CompletedProcess[str]:
        log = root / "train.log"
        log.write_text(log_text, encoding="utf-8")
        return subprocess.run(
            [
                sys.executable, str(SCRIPT), str(log), "--mode", mode,
                "--target-step", str(target), "--run-name", "rl_control",
                "--exit-code", "0", "--workdir", str(root),
            ],
            capture_output=True, text=True, check=False,
        )

    def test_smoke_reports_measured_steps_and_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_summary(root, "smoke", 3, "\n".join([
                "8 nodes RAILS OK",
                "lux001 GPU_CHECK ALL OK",
                "step:1 - critic/score/mean:0.563 response_length/mean:1640 timing_s/step:265",
                "step:2 - critic/score/mean:0.559 response_length/mean:1700 timing_s/step:180",
                "step:3 - critic/score/mean:0.557 response_length/mean:1710 timing_s/step:181",
                "=== driver exit 0",
            ]))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("last_step=3", result.stdout)
            self.assertIn("3 0.557 1710 181", result.stdout)
            self.assertIn("VERDICT: OK", result.stdout)

    def test_completed_full_segment_requires_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = "8 nodes RAILS OK\nstep:25 - critic/score/mean:0.60\n=== driver exit 0\n"
            missing = self.run_summary(root, "control", 25, log)
            self.assertEqual(missing.returncode, 1)
            self.assertIn("checkpoint missing", missing.stdout)
            checkpoint = root / "work" / "ckpt" / "rl_control" / "global_step_25"
            checkpoint.mkdir(parents=True)
            saved = self.run_summary(root, "control", 25, log)
            self.assertEqual(saved.returncode, 0, saved.stderr)
            self.assertIn(str(checkpoint), saved.stdout)
            self.assertIn("VERDICT: OK", saved.stdout)


if __name__ == "__main__":
    unittest.main()
