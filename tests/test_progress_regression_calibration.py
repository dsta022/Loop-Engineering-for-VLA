"""Dataset-relative calibration of the progress-regression review.

Verifies the fix for the calibration finding: a fixed 0.15 max-drop threshold
over-fires on clean data, so the regression review is decided against the
population via robust_upper_threshold, floored at the old constant.
"""

from __future__ import annotations

import unittest

from pipeline.data_quality_audit import (
    apply_progress_regression_reviews,
    apply_reversal_reviews,
)
from utils.quality_utils import EpisodeQuality
from utils.semantic_consistency import PROGRESS_REGRESSION_DROP


def episode(index: int, max_drop: float | None) -> EpisodeQuality:
    audit = EpisodeQuality(index)
    audit.semantic_progress_max_drop = max_drop
    return audit


def flagged(audits) -> set[int]:
    return {
        a.episode_index
        for a in audits
        if "semantic_progress_regression_review" in a.review_reasons
    }


class RegressionCalibrationTest(unittest.TestCase):
    def test_clean_population_above_fixed_threshold_is_not_flagged(self):
        # 20 clean episodes whose drops cluster near 0.12-0.21 — several exceed the
        # old fixed 0.15, but none is an outlier relative to the population.
        drops = [0.10, 0.11, 0.12, 0.12, 0.13, 0.13, 0.14, 0.15, 0.16, 0.17,
                 0.18, 0.19, 0.20, 0.21, 0.12, 0.13, 0.14, 0.11, 0.15, 0.16]
        audits = [episode(i, d) for i, d in enumerate(drops)]
        threshold = apply_progress_regression_reviews(audits)
        self.assertIsNotNone(threshold)
        self.assertGreater(threshold, PROGRESS_REGRESSION_DROP)  # relaxed above 0.15
        self.assertEqual(flagged(audits), set())  # zero false positives

    def test_genuine_outlier_is_flagged(self):
        drops = [0.10, 0.11, 0.12, 0.12, 0.13, 0.13, 0.14, 0.11, 0.12, 0.13]
        audits = [episode(i, d) for i, d in enumerate(drops)]
        audits.append(episode(99, 0.85))  # a real collapse
        apply_progress_regression_reviews(audits)
        self.assertEqual(flagged(audits), {99})

    def test_small_population_falls_back_to_fixed_threshold(self):
        # < 5 episodes: no robust estimate, so the old fixed 0.15 behavior applies.
        audits = [episode(0, 0.70), episode(1, 0.05)]
        threshold = apply_progress_regression_reviews(audits)
        self.assertEqual(threshold, PROGRESS_REGRESSION_DROP)
        self.assertEqual(flagged(audits), {0})

    def test_episodes_without_a_curve_are_ignored(self):
        audits = [episode(0, None), episode(1, None)]
        self.assertIsNone(apply_progress_regression_reviews(audits))
        self.assertEqual(flagged(audits), set())

    def test_threshold_never_below_the_floor(self):
        # An ultra-tight distribution must not drop the bar under the 0.15 floor.
        audits = [episode(i, 0.01) for i in range(10)]
        threshold = apply_progress_regression_reviews(audits)
        self.assertGreaterEqual(threshold, PROGRESS_REGRESSION_DROP)
        self.assertEqual(flagged(audits), set())


class ReversalReviewTest(unittest.TestCase):
    @staticmethod
    def _ep(index, corr):
        a = EpisodeQuality(index)
        a.semantic_time_correlation = corr
        return a

    def test_reversed_minority_flagged(self):
        audits = [self._ep(i, -0.3) for i in range(20)]
        audits += [self._ep(20, 0.6), self._ep(21, 0.55)]  # reversed (opposite sign)
        count = apply_reversal_reviews(audits)
        self.assertEqual(count, 2)
        flagged = {a.episode_index for a in audits if "semantic_reversed_episode_review" in a.review_reasons}
        self.assertEqual(flagged, {20, 21})

    def test_no_evaluator_correlations_is_noop(self):
        audits = [self._ep(i, None) for i in range(6)]
        self.assertEqual(apply_reversal_reviews(audits), 0)
        self.assertTrue(all(not a.review_reasons for a in audits))

    def test_consistent_population_not_flagged(self):
        audits = [self._ep(i, -0.3) for i in range(8)]
        self.assertEqual(apply_reversal_reviews(audits), 0)


if __name__ == "__main__":
    unittest.main()
