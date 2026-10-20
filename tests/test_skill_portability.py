"""Regression tests for Claude-specific portability warnings."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.skill_fixtures import run_validate_skills, write_skill


class SkillPortabilityTests(unittest.TestCase):
    def test_validator_flags_claude_specific_paths_and_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            skill_dir = root / "skills" / "portable-skill"
            write_skill(
                skill_dir,
                description="Exercise portability checks across supported clients",
                body="\n".join(
                    [
                        "Read files from .claude/skills/portable-skill/references/ when you need the bundled notes.",
                        "Run python ${CLAUDE_SKILL_DIR}/scripts/helper.py with $ARGUMENTS and $1.",
                    ]
                ),
            )

            completed = run_validate_skills(skill_dir)

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            output = completed.stdout + completed.stderr
            self.assertIn("spec-sem-body-tokens-not-interpreted", output)
            self.assertIn("codex-disc-no-claude-skills-dir", output)
            self.assertIn("codex-body-explicit-injection-no-cap", output)
            self.assertIn("antigravity-workspace-root", output)
            self.assertIn("antigravity-body-verbatim", output)
            self.assertIn("cursor-semantics-third-party-dirs-default-on", output)
            self.assertIn("cursor-body-model-visible-form", output)


if __name__ == "__main__":
    unittest.main()
