"""Tests for the proprioceptive motion evaluator.

Pure numpy — no model, no GPU, no video. Trajectories are scripted so each
failure mode (freeze, temporal scramble, no-motion) has a predictable effect on
the score, and the SemanticResult contract is verified against normalize_result.
"""

from __future__ import annotations

import unittest

import numpy as np

from semantic.motion_evaluator import (
    MotionEvaluator,
    create_evaluator,
    cumulative_progress_curve,
    plausibility_metrics,
    step_magnitudes,
)
from utils.semantic_evaluator import EpisodeMotion, SemanticEpisode, normalize_result


def smooth_reach(T=40, D=6):
    """A smooth, non-collinear, roughly constant-speed trajectory.

    First two dims trace a constant-speed arc (uniform step size, low jerk); a
    third drifts linearly; the rest are flat. Shuffling it produces genuine
    non-physical jerk, as a real robot trajectory would.
    """
    theta = np.linspace(0.0, 1.5 * np.pi, T)
    traj = np.zeros((T, D), dtype=np.float64)
    traj[:, 0] = np.cos(theta)
    traj[:, 1] = np.sin(theta)
    if D > 2:
        traj[:, 2] = np.linspace(0.0, 1.0, T)
    return traj


class CurveTest(unittest.TestCase):
    def test_cumulative_curve_monotone_and_unit(self):
        curve = cumulative_progress_curve(smooth_reach(), 32)
        self.assertEqual(len(curve), 32)
        self.assertAlmostEqual(curve[0], 0.0, places=6)
        self.assertAlmostEqual(curve[-1], 1.0, places=6)
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(curve, curve[1:])))

    def test_frozen_curve_is_zero(self):
        frozen = np.repeat(smooth_reach()[:1], 40, axis=0)
        self.assertEqual(cumulative_progress_curve(frozen, 16), [0.0] * 16)


class MetricsTest(unittest.TestCase):
    def test_freeze_has_no_motion(self):
        frozen = np.repeat(smooth_reach()[:1], 40, axis=0)
        m = plausibility_metrics(frozen, 0.10, 8.0)
        self.assertEqual(m["total_path"], 0.0)
        self.assertEqual(m["motion_fraction"], 0.0)

    def test_smooth_reach_is_directionally_consistent(self):
        m = plausibility_metrics(smooth_reach(), 0.10, 8.0)
        self.assertGreater(m["direction_consistency"], 0.9)
        self.assertGreater(m["motion_fraction"], 0.9)

    def test_shuffle_breaks_direction_consistency(self):
        traj = smooth_reach()
        rng = np.random.default_rng(0)
        good = plausibility_metrics(traj, 0.10, 8.0)["direction_consistency"]
        bad = plausibility_metrics(traj[rng.permutation(traj.shape[0])], 0.10, 8.0)["direction_consistency"]
        self.assertLess(bad, 0.5)
        self.assertLess(bad, good)


class ScoreTest(unittest.TestCase):
    def setUp(self):
        self.ev = MotionEvaluator()

    def test_good_reach_scores_high(self):
        r = normalize_result(self.ev.score_trajectory(smooth_reach()))
        self.assertGreater(r.success_probability, 0.9)
        self.assertAlmostEqual(r.final_progress, 1.0, places=6)

    def test_freeze_scores_zero(self):
        frozen = np.repeat(smooth_reach()[:1], 40, axis=0)
        r = self.ev.score_trajectory(frozen)
        self.assertEqual(r.success_probability, 0.0)

    def test_shuffle_scores_below_good(self):
        traj = smooth_reach()
        rng = np.random.default_rng(1)
        good = self.ev.score_trajectory(traj).success_probability
        bad = self.ev.score_trajectory(traj[rng.permutation(traj.shape[0])]).success_probability
        self.assertLess(bad, good)

    def test_too_short_raises(self):
        with self.assertRaises(RuntimeError):
            self.ev.score_trajectory(np.zeros((1, 6)))

    def test_non_finite_raises(self):
        traj = smooth_reach()
        traj[3, 2] = np.nan
        with self.assertRaises(RuntimeError):
            self.ev.score_trajectory(traj)


class EvaluateContractTest(unittest.TestCase):
    def _episode(self, motion):
        return SemanticEpisode(
            dataset_root=__import__("pathlib").Path("/d"),
            episode_index=0, length=40, tasks=("Table Clean",), videos=(), motion=motion,
        )

    def test_evaluate_prefers_state_then_action(self):
        ev = MotionEvaluator()
        m = EpisodeMotion(action=smooth_reach(), state=None)
        r = normalize_result(ev.evaluate(self._episode(m)))
        self.assertEqual(r.details["method"], "motion_cumulative_path")

    def test_missing_motion_raises(self):
        with self.assertRaises(RuntimeError):
            MotionEvaluator().evaluate(self._episode(None))

    def test_factory_and_json_details(self):
        ev = create_evaluator({"jerk_multiple": 6.0})
        self.assertEqual(ev.jerk_multiple, 6.0)
        r = normalize_result(ev.score_trajectory(smooth_reach()))
        import json
        json.dumps(r.details)


if __name__ == "__main__":
    unittest.main()
