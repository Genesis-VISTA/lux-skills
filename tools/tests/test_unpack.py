"""Black-box unpack checks; run with stdlib unittest or pytest.

All destinations are temporary. Tests never install into a real user home.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "unpack.sh"


def manifest(root: Path) -> dict[str, tuple]:
    """Independently compare names, bytes, link text, and executable bits."""
    result = {}
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        if path.is_symlink():
            result[rel] = ("link", os.readlink(path))
        elif path.is_file():
            result[rel] = ("file", hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mode & 0o111)
        elif path.is_dir():
            result[rel] = ("dir",)
    return result


def leaves(root: Path) -> list[Path]:
    """Independent discovery; stop at entry points, skip hidden directories."""
    result = []
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = sorted(d for d in subdirs if not d.startswith("."))
        if "SKILL.md" in files:
            result.append(Path(directory))
            subdirs[:] = []
    return sorted(result)


class UnpackTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lux unpack ")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.repo = self.base / "clone with spaces"
        self.repo.mkdir()
        self.script = self.repo / "unpack.sh"
        shutil.copy2(SCRIPT, self.script)
        self.skills = self.repo / "skills"
        self.skills.mkdir()
        self.dest = self.base / "installed skills"
        self.flat = self.skill("solo")
        self.alpha = self.skill("domain-a/alpha")
        self.beta = self.skill("domain-a/deep/beta")
        self.skill(".hidden-domain/hidden-one")
        self.skill("domain-a/.hidden-skill")
        # A SKILL.md inside an existing skill is a resource, not another leaf.
        self.skill("domain-a/alpha/references/nested")
        (self.alpha / "scripts").mkdir()
        script = self.alpha / "scripts" / "run.sh"
        script.write_text("#!/bin/sh\nprintf 'alpha resource\\n'\n")
        script.chmod(0o755)
        (self.alpha / ".hidden-resource").write_bytes(b"hidden\x00asset")
        (self.beta / "alpha-resource").symlink_to("../alpha/.hidden-resource")

    def skill(self, relative):
        path = self.skills / relative
        path.mkdir(parents=True)
        (path / "SKILL.md").write_text(f"---\nname: {path.name}\ndescription: Test {path.name}.\n---\nUse this skill.\n")
        return path

    def run_unpack(self, *args, script=None, cwd=None):
        return subprocess.run(
            [str(script or self.script), *map(str, args)],
            cwd=cwd or self.base, capture_output=True, text=True, timeout=30,
            env={**os.environ, "LC_ALL": "C"},
        )

    def success(self, *args, **kwargs):
        proc = self.run_unpack(*args, **kwargs)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        return proc

    def failure(self, *args, **kwargs):
        proc = self.run_unpack(*args, **kwargs)
        self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("unpack.sh:", proc.stderr)
        return proc

    def test_copy_preserves_all_leaf_resources_and_modes(self):
        proc = self.success("--mode", "copy", "--target", self.dest)
        self.assertIn("3 skill(s) (copy)", proc.stdout)
        self.assertEqual({p.name for p in self.dest.iterdir()}, {"solo", "alpha", "beta"})
        for source in (self.flat, self.alpha, self.beta):
            self.assertEqual(manifest(source), manifest(self.dest / source.name))
            self.assertFalse((self.dest / source.name).is_symlink())
        self.assertEqual(subprocess.check_output([self.dest / "alpha/scripts/run.sh"], text=True), "alpha resource\n")
        # Copy is independent: deleting the source does not affect installed bytes.
        shutil.rmtree(self.repo)
        self.assertTrue((self.dest / "solo/SKILL.md").is_file())
        self.assertEqual((self.dest / "beta/alpha-resource").read_bytes(), b"hidden\x00asset")

    def test_symlink_live_updates(self):
        self.success("--target", self.dest)
        for source in (self.flat, self.alpha, self.beta):
            output = self.dest / source.name
            self.assertTrue(output.is_symlink())
            self.assertEqual(output.resolve(), source.resolve())
        (self.flat / "SKILL.md").write_text("updated")
        self.assertEqual((self.dest / "solo/SKILL.md").read_text(), "updated")

    def test_list_has_no_destination_side_effect(self):
        proc = self.success("--target", self.dest, "--list")
        self.assertIn("3 skill(s).", proc.stdout)
        self.assertIn("[domain-a]", proc.stdout)
        self.assertIn("(flat)", proc.stdout)
        self.assertNotIn("nested", proc.stdout)
        self.assertNotIn("hidden", proc.stdout)
        self.assertFalse(self.dest.exists())

    def test_domain_and_flat_filter(self):
        self.success("domain-a", "--mode=copy", f"--target={self.dest}")
        self.assertEqual({p.name for p in self.dest.iterdir()}, {"alpha", "beta"})
        proc = self.success("solo", "domain-a", "solo", "--list")
        self.assertIn("3 skill(s).", proc.stdout)

    def test_root_harnesses_and_target_precedence(self):
        root = self.base / "project"
        for harness, expected in (("claude", [".claude"]), ("agents", [".agents"]), ("all", [".claude", ".agents"])):
            with self.subTest(harness=harness):
                scoped = root / harness
                self.success("solo", "--root", scoped, "--harness", harness)
                self.assertEqual({p.name for p in scoped.iterdir()}, set(expected))
                for directory in expected:
                    self.assertTrue((scoped / directory / "skills/solo/SKILL.md").is_file())
        unused = self.base / "unused"
        self.success("solo", "--root", unused, "--harness=all", "--target", self.dest)
        self.assertFalse(unused.exists())
        self.assertTrue((self.dest / "solo/SKILL.md").is_file())

    def test_default_is_working_directory_claude(self):
        self.success("solo")
        self.assertTrue((self.base / ".claude/skills/solo/SKILL.md").is_file())

    def test_enclosing_harness_controls_default_root(self):
        for harness in (".claude", ".agents"):
            with self.subTest(harness=harness):
                root = self.base / ("workspace" + harness)
                nested = root / harness / "catalog"
                shutil.copytree(self.repo, nested, symlinks=True)
                self.success("solo", script=nested / "unpack.sh")
                self.assertTrue((root / ".claude/skills/solo/SKILL.md").is_file())

    def test_repeat_replaces_stale_entry_and_keeps_unrelated(self):
        self.success("--target", self.dest, "--mode=copy")
        (self.dest / "alpha/stale").write_text("remove")
        (self.dest / "unrelated").write_text("preserve")
        for mode in ("copy", "symlink", "copy", "copy"):
            self.success("--target", self.dest, "--mode", mode)
            self.assertFalse((self.dest / "alpha/stale").exists())
            self.assertEqual((self.dest / "unrelated").read_text(), "preserve")
            self.assertEqual((self.dest / "alpha").is_symlink(), mode == "symlink")

    def test_existing_external_and_broken_symlinks_are_replaced_safely(self):
        self.dest.mkdir()
        external = self.base / "external"
        external.mkdir()
        (external / "keep").write_text("safe")
        (self.dest / "alpha").symlink_to(external)
        (self.dest / "beta").symlink_to(self.base / "missing")
        (self.dest / "solo").write_text("old file")
        self.success("--target", self.dest, "--mode=copy")
        self.assertEqual((external / "keep").read_text(), "safe")
        self.assertTrue((self.dest / "beta/SKILL.md").is_file())

    def test_invalid_arguments_fail_before_install(self):
        cases = [("--unknown",), ("-z",), ("--mode", "bad"), ("--harness", "grok"), ("typo",), ("domain-a", "typo"), ("../skills",)]
        for option in ("--root", "--target", "--mode", "--harness"):
            cases.extend([(option,), (option + "=",), (option, ""), (option, "--list")])
        for args in cases:
            with self.subTest(args=args):
                self.failure(*args)
        self.assertFalse((self.base / ".claude").exists())

    def test_duplicate_name_fails_without_replacing_existing_destination(self):
        self.skill("domain-b/alpha")
        self.dest.mkdir()
        (self.dest / "sentinel").write_text("preserve")
        proc = self.failure("--target", self.dest)
        self.assertIn("duplicate skill name", proc.stderr)
        self.assertEqual(list(self.dest.iterdir()), [self.dest / "sentinel"])

    def test_empty_or_missing_catalog_fails(self):
        shutil.rmtree(self.skills)
        self.failure("--target", self.dest)
        self.skills.mkdir()
        self.failure("--target", self.dest)
        self.assertFalse(self.dest.exists())

    def test_catalog_symlink_cycle_is_rejected(self):
        (self.skills / "cycle").symlink_to(self.skills)
        self.assertIn("symlinked catalog directory", self.failure("--list").stderr)

    def test_target_cannot_destroy_source(self):
        original = manifest(self.skills)
        alias = self.base / "source-alias"
        alias.symlink_to(self.skills)
        for target in (self.skills, self.skills / "domain-a", alias / "domain-a", self.skills / "new/../domain-a"):
            with self.subTest(target=target):
                self.assertIn("overlaps source", self.failure("--target", target).stderr)
                self.assertEqual(manifest(self.skills), original)

    def test_output_ancestor_cannot_destroy_clone(self):
        # Name a skill after the clone; writing it next to the clone would rm -rf it.
        self.skill(self.repo.name)
        before = manifest(self.skills)
        self.assertIn("replace the source", self.failure("--target", self.base).stderr)
        self.assertEqual(manifest(self.skills), before)
        self.assertTrue(self.script.exists())

    def test_target_through_existing_symlink_and_relative_parent(self):
        physical = self.base / "physical"
        physical.mkdir()
        (self.base / "alias").symlink_to(physical)
        self.success("solo", "--target", "alias/missing/../new", cwd=self.base)
        self.assertTrue((physical / "new/solo/SKILL.md").is_file())

    def test_non_directory_target_fails_before_any_install(self):
        file = self.base / "file"
        file.write_text("safe")
        self.failure("--target", file)
        self.failure("--target", file / "child")
        self.assertEqual(file.read_text(), "safe")

    def test_all_preflights_both_destinations(self):
        root = self.base / "project"
        (root / ".agents").mkdir(parents=True)
        (root / ".agents/skills").symlink_to(self.skills)
        self.failure("--root", root, "--harness=all")
        self.assertFalse((root / ".claude").exists())

    def test_help_explains_options(self):
        for switch in ("-h", "--help"):
            proc = self.success(switch)
            for option in ("--target", "--root", "--mode", "--harness", "--list"):
                self.assertIn(option, proc.stdout)


class ActualCatalogTest(unittest.TestCase):
    def test_entire_catalog_survives_both_modes(self):
        sources = leaves(REPO / "skills")
        self.assertEqual(len(sources), 4)
        self.assertEqual(len({source.name for source in sources}), len(sources))
        with tempfile.TemporaryDirectory(prefix="lux full catalog ") as tmp:
            for mode in ("copy", "symlink"):
                with self.subTest(mode=mode):
                    destination = Path(tmp) / mode
                    proc = subprocess.run([str(SCRIPT), "--mode", mode, "--target", str(destination)], capture_output=True, text=True, timeout=60)
                    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                    self.assertEqual({path.name for path in destination.iterdir()}, {source.name for source in sources})
                    for source in sources:
                        self.assertEqual(manifest(source), manifest(destination / source.name), source.name)
                        for link in (destination / source.name).rglob("*"):
                            if link.is_symlink():
                                self.assertTrue(link.exists(), f"Broken installed support link: {link}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
