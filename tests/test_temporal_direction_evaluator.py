"""Tests for the temporal-direction (reversal) evaluator.

A fake backend places each frame on a start->goal axis by its embedded code, so a
forward frame order projects as rising and a reversed order as falling. No torch,
no GPU, no video.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from semantic.temporal_direction_evaluator import (
    TemporalDirectionEvaluator,
    directional_consensus_outliers,
    spearman_with_time,
)
from utils.semantic_evaluator import (
    SemanticEpisode,
    VideoSegment,
    normalize_result,
)


class AxisBackend:
    """start=e0, goal=e1; a frame coded c in [0,1] embeds to (1-c)*e0 + c*e1."""

    def __init__(self):
        self.name = "axis_fake_backend"
        self.embedding_dim = 2

    def embed_texts(self, texts):
        # texts[0] -> start basis, texts[1] -> goal basis
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)[: len(texts)]

    def embed_images(self, frames):
        out = []
        for f in frames:
            c = float(np.asarray(f).reshape(-1)[0]) / 255.0
            out.append([1.0 - c, c])
        return np.asarray(out, dtype=np.float64)


def coded_frames(codes):
    return np.stack(
        [np.full((2, 2, 3), int(c), dtype=np.uint8) for c in codes], axis=0
    )


PROFILE = {
    "stages": [
        {"name": "s", "observable_condition": "a cluttered messy table", "weight": 1.0},
        {"name": "g", "observable_condition": "a tidy clean table", "weight": 1.0},
    ]
}


class SpearmanTest(unittest.TestCase):
    def test_rising_is_positive(self):
        self.assertAlmostEqual(spearman_with_time(np.array([0.1, 0.4, 0.6, 0.9])), 1.0)

    def test_falling_is_negative(self):
        self.assertAlmostEqual(spearman_with_time(np.array([0.9, 0.6, 0.4, 0.1])), -1.0)

    def test_constant_is_zero(self):
        self.assertEqual(spearman_with_time(np.array([0.5, 0.5, 0.5, 0.5])), 0.0)


class DirectionTest(unittest.TestCase):
    def setUp(self):
        self.ev = TemporalDirectionEvaluator(AxisBackend(), num_frames=6)

    def _score(self, codes):
        return normalize_result(
            self.ev.score_frames(coded_frames(codes), ("Table Clean",), PROFILE)
        )

    def test_forward_scores_high(self):
        r = self._score([10, 60, 110, 160, 210, 250])
        self.assertGreater(r.success_probability, 0.9)
        self.assertGreater(r.details["time_correlation"], 0.9)

    def test_reversed_scores_low(self):
        r = self._score([250, 210, 160, 110, 60, 10])
        self.assertLess(r.success_probability, 0.1)
        self.assertLess(r.details["time_correlation"], -0.9)

    def test_forward_and_reverse_are_separated(self):
        fwd = self._score([10, 60, 110, 160, 210, 250]).success_probability
        rev = self._score([250, 210, 160, 110, 60, 10]).success_probability
        self.assertGreater(fwd - rev, 0.8)

    def test_curve_declines_when_reversed(self):
        r = self._score([250, 210, 160, 110, 60, 10])
        self.assertGreater(r.progress_curve[0], r.progress_curve[-1])


class AnchorTest(unittest.TestCase):
    def test_config_text_overrides_profile(self):
        ev = TemporalDirectionEvaluator(
            AxisBackend(), num_frames=4, start_text="messy", goal_text="clean"
        )
        r = ev.score_frames(coded_frames([10, 90, 170, 250]), ("t",), None)
        self.assertGreater(r.success_probability, 0.9)

    def test_missing_anchor_raises(self):
        ev = TemporalDirectionEvaluator(AxisBackend(), num_frames=4)
        with self.assertRaises(RuntimeError):
            ev.score_frames(coded_frames([10, 90, 170, 250]), ("t",), None)

    def test_too_few_frames_raises(self):
        ev = TemporalDirectionEvaluator(AxisBackend(), num_frames=3)
        with self.assertRaises(RuntimeError):
            ev.score_frames(coded_frames([10, 250]), ("t",), PROFILE)


class ConsensusOutlierTest(unittest.TestCase):
    def test_minority_opposite_sign_is_flagged(self):
        # 20 forward (negative here) + 3 reversed (positive) -> the 3 are flagged.
        corrs = [-0.3] * 20 + [0.6, 0.5, 0.55]
        flagged = directional_consensus_outliers(corrs)
        self.assertEqual(flagged, {20, 21, 22})

    def test_no_false_positive_on_consistent_population(self):
        self.assertEqual(directional_consensus_outliers([-0.3, -0.4, -0.2, -0.35, -0.3]), set())

    def test_near_zero_not_flagged(self):
        # opposite sign but below the magnitude floor -> directionless, not flagged
        corrs = [-0.3] * 6 + [0.05]
        self.assertEqual(directional_consensus_outliers(corrs, min_magnitude=0.15), set())

    def test_small_population_returns_empty(self):
        self.assertEqual(directional_consensus_outliers([-0.3, 0.6]), set())

    def test_ignores_none_entries(self):
        corrs = [-0.3, -0.4, None, -0.3, -0.35, 0.6]
        self.assertEqual(directional_consensus_outliers(corrs), {5})

    def test_works_when_consensus_is_positive(self):
        corrs = [0.4] * 10 + [-0.5]
        self.assertEqual(directional_consensus_outliers(corrs), {10})


class EvaluateContractTest(unittest.TestCase):
    def test_missing_video_raises(self):
        ev = TemporalDirectionEvaluator(AxisBackend())
        ep = SemanticEpisode(
            dataset_root=Path("/d"), episode_index=0, length=6,
            tasks=("t",), videos=(), task_profile=PROFILE,
        )
        with self.assertRaises(ValueError):
            ev.evaluate(ep)


if __name__ == "__main__":
    unittest.main()
