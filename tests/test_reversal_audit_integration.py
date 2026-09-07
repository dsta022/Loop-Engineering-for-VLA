"""End-to-end audit test for the temporal-reversal review.

Runs the full audit over a multi-episode dataset with a semantic evaluator that
emits a per-episode ``time_correlation`` (as a temporal-direction evaluator does).
A forward majority sets the direction consensus; the reversed minority must be
flagged in the dataset pass, surface in the report rows, and be counted in the
summary — while forward episodes stay clean.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.test_semantic_consistency import run_motion_audit, write_motion_dataset


class DirectionTaggingEvaluator:
    """Emits a fixed, rising progress curve plus a scripted time_correlation.

    The curve is deliberately flat-of-regressions and only 3 points so it trips
    no progress-shape check; the reversal signal lives entirely in
    details["time_correlation"], isolating what this test exercises.
    """

    name = "direction_tagging_test_evaluator"

    def __init__(self, correlations: list[float]) -> None:
        self.correlations = correlations

    def evaluate(self, episode):
        corr = self.correlations[episode.episode_index]
        return {
            "progress_curve": [0.3, 0.5, 0.7],
            "success_probability": 0.95,  # >= pass threshold, so semantic status = pass
            "details": {"method": "temporal_direction", "time_correlation": corr},
        }


class ReversalAuditIntegrationTests(unittest.TestCase):
    def test_reversed_minority_is_flagged_end_to_end(self) -> None:
        # 4 forward (positive consensus) + 2 reversed (negative, opposite sign).
        correlations = [0.42, 0.50, 0.45, 0.40, -0.50, -0.55]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_motion_dataset(
                root, "reversal", "clean the table", gripper_cycles=[1] * 6
            )
            summary, rows = run_motion_audit(
                dataset,
                root / "report",
                semantic_evaluator=DirectionTaggingEvaluator(correlations),
                motion_check="none",  # isolate the reversal signal from motion checks
            )

            # summary counts exactly the two reversed episodes
            self.assertEqual(summary["semantic_consistency"]["reversed_episode_review"], 2)

            # the reversed minority (4, 5) is flagged and routed to review
            for ep in (4, 5):
                self.assertIn(
                    "semantic_reversed_episode_review", rows[ep]["review_reasons"]
                )
                self.assertEqual(rows[ep]["status"], "review")
                self.assertEqual(rows[ep]["structural_status"], "keep")

            # the forward majority is clean
            for ep in (0, 1, 2, 3):
                self.assertNotIn(
                    "semantic_reversed_episode_review", rows[ep]["review_reasons"]
                )
                self.assertEqual(rows[ep]["status"], "keep")
                self.assertEqual(rows[ep]["semantic_status"], "pass")

            # the per-episode correlation was captured into the report
            self.assertAlmostEqual(float(rows[4]["semantic_time_correlation"]), -0.50)
            self.assertAlmostEqual(float(rows[0]["semantic_time_correlation"]), 0.42)

    def test_all_forward_population_flags_nothing(self) -> None:
        correlations = [0.4, 0.45, 0.5, 0.42, 0.48, 0.41]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_motion_dataset(
                root, "all_forward", "clean the table", gripper_cycles=[1] * 6
            )
            summary, rows = run_motion_audit(
                dataset,
                root / "report",
                semantic_evaluator=DirectionTaggingEvaluator(correlations),
                motion_check="none",
            )
            self.assertEqual(summary["semantic_consistency"]["reversed_episode_review"], 0)
            for ep in range(6):
                self.assertNotIn(
                    "semantic_reversed_episode_review", rows[ep]["review_reasons"]
                )


if __name__ == "__main__":
    unittest.main()
