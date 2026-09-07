from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.threshold_calibration import calibrate, read_labels, read_quality_report

REPO_ROOT = Path(__file__).resolve().parents[1]

# episode -> (semantic_score, review_reasons, rejected by a human?)
FIXTURE = {
    0: (0.10, "semantic_low_score_review", True),
    1: (0.20, "semantic_low_score_review", True),
    2: (0.30, "semantic_low_score_review", True),
    3: (0.40, "semantic_low_score_review;video_file_black_high_review:observation.images.front", True),
    4: (0.50, "semantic_low_score_review", False),
    5: (0.55, "semantic_uncertain_score_review", True),
    6: (0.70, "semantic_uncertain_score_review", False),
    7: (0.85, "semantic_uncertain_score_review", False),
    8: (0.92, "video_file_black_high_review:observation.images.front", False),
    9: (0.99, "", False),
}


def fixture_rows() -> dict[int, dict[str, str]]:
    rows = {}
    for episode, (score, reasons, _) in FIXTURE.items():
        rows[episode] = {
            "episode_index": str(episode),
            "status": "review" if reasons else "keep",
            "semantic_score": f"{score}",
            "review_reasons": reasons,
            "drop_reasons": "",
        }
    return rows


def fixture_labels() -> dict[int, bool]:
    return {episode: rejected for episode, (_, _, rejected) in FIXTURE.items()}


class ThresholdCalibrationTests(unittest.TestCase):
    def test_reason_precision_and_recall(self) -> None:
        report = calibrate(fixture_rows(), fixture_labels())
        by_reason = {item["reason"]: item for item in report["reasons"]}
        low = by_reason["semantic_low_score_review"]
        self.assertEqual(low["flagged"], 5)
        self.assertAlmostEqual(low["precision"], 4 / 5)
        self.assertAlmostEqual(low["recall"], 4 / 5)
        black = by_reason["video_file_black_high_review"]
        self.assertEqual(black["flagged"], 2)
        self.assertAlmostEqual(black["precision"], 0.5)
        self.assertEqual(report["status_confusion"]["keep"], {"accepted": 1, "rejected": 0})

    def test_threshold_recommendation(self) -> None:
        report = calibrate(fixture_rows(), fixture_labels(), precision_target=0.9, pass_false_rate=0.05)
        recommendation = report["recommendation"]
        self.assertAlmostEqual(recommendation["semantic_failure_threshold"], 0.5)
        self.assertAlmostEqual(recommendation["semantic_pass_threshold"], 0.7)
        self.assertTrue(recommendation["consistent"])

    def test_inconsistent_targets_are_reported_instead_of_forced(self) -> None:
        report = calibrate(fixture_rows(), fixture_labels(), precision_target=1.0, pass_false_rate=0.0)
        self.assertAlmostEqual(report["recommendation"]["semantic_failure_threshold"], 0.5)
        self.assertAlmostEqual(report["recommendation"]["semantic_pass_threshold"], 0.7)
        report = calibrate(fixture_rows(), {0: False, 1: False}, precision_target=0.9)
        self.assertIsNone(report["recommendation"]["semantic_failure_threshold"])
        self.assertFalse(report["recommendation"]["consistent"])

    def test_no_overlap_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "no episode appears in both"):
            calibrate(fixture_rows(), {99: True})

    def test_cli_writes_calibration_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "quality_audit_reports" / "dataset"
            report_dir.mkdir(parents=True)
            fields = ["episode_index", "status", "semantic_score", "review_reasons", "drop_reasons"]
            with (report_dir / "quality_report.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in fixture_rows().values():
                    writer.writerow(row)
            labels = root / "labels.csv"
            with labels.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["episode_index", "label"])
                writer.writeheader()
                for episode, rejected in fixture_labels().items():
                    writer.writerow({"episode_index": episode, "label": "reject" if rejected else "accept"})
            rows = read_quality_report(report_dir)
            self.assertEqual(len(rows), 10)
            self.assertEqual(read_labels(labels, "label", {"reject"}), fixture_labels())

            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "threshold_calibration.py"),
                    "--quality-report",
                    str(report_dir),
                    "--labels",
                    str(labels),
                ],
                capture_output=True,
                text=True,
                cwd=tmp,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn("Recommended semantic thresholds", completed.stdout)
            payload = json.loads((report_dir / "calibration_report.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["episodes_matched"], 10)


if __name__ == "__main__":
    unittest.main()
