"""Smoke tests for the repository policy validator."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.skill_fixtures import write_skill, write_text
from tools.repo_inventory import build_readme_contributor_names, build_readme_tree_lines


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "validate_repo_policy.py"


def _write_valid_fixture(root: Path) -> None:
    write_text(root / "LICENSE", "Root license placeholder\n")
    write_text(
        root / "CONTRIBUTING.md",
        "\n".join(
            [
                "AI/LLM-assisted contributions are permitted when the author reviews and validates the result.",
                "For AI/LLM-assisted contributions, see [Guidelines for AI/LLM-Assisted Contributions](#guidelines-for-ai-llm-assisted-contributions) below.",
                "",
                "### Guidelines for AI/LLM-Assisted Contributions",
                "",
                "Placeholder policy text.",
                "",
            ]
        ),
    )
    write_text(root / "CODE_OF_CONDUCT.md", "Code of conduct placeholder\n")
    write_text(
        root / "NOTICE",
        "\n".join(
            [
                "Lux Skills",
                "",
                "Contact: Junqi Yin <yinj@ornl.gov>",
                "",
                "Known subtrees with separate licensing:",
                "",
                "  skills/primus-skill/LICENSE",
                "  skills/vllm-skill/LICENSE",
                "  skills/ATTRIBUTION.md",
                "",
            ]
        ),
    )
    write_text(root / "docs" / "getting_started.md", "# Getting Started\n")
    write_text(root / "unpack.sh", "#!/usr/bin/env bash\n")
    write_text(
        root / "skills" / "ATTRIBUTION.md",
        "\n".join(
            [
                "# Attribution",
                "",
                "Licensed Apache-2.0.",
                "",
                "**Authors:**",
                "- Junqi Yin, Oak Ridge National Laboratory (ORNL)",
                "- Ada Example, Example Laboratory",
                "",
            ]
        ),
    )
    write_skill(root / "skills" / "primus-skill", description="Primus training on Lux")
    write_skill(root / "skills" / "vllm-skill", description="vLLM serving on Lux")
    for name in ("primus-skill", "vllm-skill"):
        write_text(root / "skills" / name / "LICENSE", "Apache-2.0 license placeholder\n")

    tree = "\n".join(build_readme_tree_lines(root))
    contributors = "\n".join(f"- {name}" for name in build_readme_contributor_names(root))
    write_text(
        root / "README.md",
        "\n".join(
            [
                "# Lux Skills",
                "",
                "## Repository Structure",
                "",
                "```",
                tree,
                "```",
                "",
                "## Contributors",
                "",
                "Individual skills and their directories contain specific attribution details. The catalog-level and skill contributors include the following:",
                "",
                contributors,
                "",
                "## License",
                "This product is licensed under the root [LICENSE](LICENSE) found at the root of this repository.",
                "Some skills under `skills/` are sourced from third parties. Where a `LICENSE` (or `LICENSE.txt`) file is present in a subdirectory, that license governs the contents of that subdirectory and supersedes the root license for that subtree.",
                "The repository-level summary of individual skill and third-party licensing details is in [NOTICE](NOTICE); see the attribution file in each identified subtree for the licensing information supplied with that content.",
                "",
                "## Acknowledgment",
                "This work was supported by the U.S. Department of Energy (DOE), Office of Science, Office of Advanced Scientific Computing Research in alignment with DOE's Genesis Mission.",
                "",
            ]
        ),
    )


class RepoPolicyValidatorTests(unittest.TestCase):
    def test_tree_inventory_discovers_new_skill_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_valid_fixture(root)
            write_skill(root / "skills" / "zeta-skills" / "alpha", description="Alpha skill")
            write_skill(root / "skills" / "zeta-skills" / "beta", description="Beta skill")

            tree_lines = build_readme_tree_lines(root)

            self.assertIn("│   └── zeta-skills/  # alpha, beta (2)", tree_lines)
            self.assertLess(tree_lines.index("│   └── zeta-skills/  # alpha, beta (2)"), tree_lines.index("├── CONTRIBUTING.md"))

    def test_validator_accepts_a_minimal_consistent_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_valid_fixture(root)

            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--root", str(root)],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn("repo policy: OK", completed.stdout)

    def test_validator_reports_missing_readme_notice_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_valid_fixture(root)
            write_text(
                root / "README.md",
                "\n".join(
                    [
                        "# Lux Skills",
                        "",
                        "See [LICENSE](LICENSE) only.",
                        "",
                    ]
                ),
            )

            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--root", str(root)],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("README.md", completed.stdout)
            self.assertIn("NOTICE", completed.stdout)

    def test_validator_reports_missing_acknowledgment_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_valid_fixture(root)
            write_text(
                root / "README.md",
                "\n".join(
                    [
                        "# Lux Skills",
                        "",
                        "See [LICENSE](LICENSE) and [NOTICE](NOTICE).",
                        "",
                    ]
                ),
            )

            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--root", str(root)],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("README.md", completed.stdout)
            self.assertIn("Acknowledgment", completed.stdout)

    def test_validator_reports_missing_ai_assisted_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_valid_fixture(root)
            write_text(
                root / "CONTRIBUTING.md",
                "\n".join(
                    [
                        "AI/LLM-assisted contributions are permitted when the author reviews and validates the result.",
                        "",
                        "### Guidelines for AI/LLM-Assisted Contributions",
                        "",
                        "Placeholder policy text.",
                        "",
                    ]
                ),
            )

            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--root", str(root)],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("CONTRIBUTING.md", completed.stdout)
            self.assertIn("Guidelines for AI/LLM-Assisted Contributions", completed.stdout)

    def test_validator_reports_stale_repository_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_valid_fixture(root)
            write_text(
                root / "README.md",
                (root / "README.md").read_text(encoding="utf-8").replace(
                    "└── skill_spec.md              # Agent Skills format specification\n",
                    "",
                ),
            )

            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--root", str(root)],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("repository structure", completed.stdout)

    def test_validator_reports_stale_contributor_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_valid_fixture(root)
            write_text(
                root / "README.md",
                (root / "README.md").read_text(encoding="utf-8").replace("- Ada Example\n", ""),
            )

            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--root", str(root)],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("contributors", completed.stdout)


if __name__ == "__main__":
    unittest.main()
