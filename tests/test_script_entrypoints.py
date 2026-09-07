from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.task_semantic_studio import build_audit_command
from tools.trajectory_language_studio import build_enrichment_command

REPO_ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    "pipeline/data_quality_audit.py",
    "pipeline/dataset_merge.py",
    "pipeline/dataset_clean_build.py",
    "pipeline/feedback_dataset_build.py",
    "pipeline/trajectory_language_enrichment.py",
    "pipeline/policy_iteration.py",
    "pipeline/hf_push.py",
    "tools/dataset_completeness_check.py",
    "tools/depth_visualize.py",
    "tools/threshold_calibration.py",
]

MODULES = ["pipeline.dataset_merge", "pipeline.data_quality_audit", "tools.threshold_calibration"]


def run_help(argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)


class ScriptEntrypointTests(unittest.TestCase):
    def test_every_script_runs_by_path_from_an_unrelated_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for script in SCRIPTS:
                with self.subTest(script=script):
                    completed = run_help([sys.executable, str(REPO_ROOT / script), "--help"], Path(tmp))
                    self.assertEqual(completed.returncode, 0, completed.stderr[-800:])
                    self.assertIn("usage", completed.stdout.lower())

    def test_modules_run_from_the_repo_root(self) -> None:
        for module in MODULES:
            with self.subTest(module=module):
                completed = run_help([sys.executable, "-m", module, "--help"], REPO_ROOT)
                self.assertEqual(completed.returncode, 0, completed.stderr[-800:])

    def test_studio_commands_point_at_existing_scripts(self) -> None:
        audit = build_audit_command(Path("dataset"), None, "backend:create_evaluator")
        self.assertTrue(Path(audit[1]).is_file(), audit[1])
        self.assertEqual(Path(audit[1]).parent.name, "pipeline")
        enrich = build_enrichment_command(Path("dataset"), "backend:create_annotator")
        self.assertTrue(Path(enrich[1]).is_file(), enrich[1])


if __name__ == "__main__":
    unittest.main()
