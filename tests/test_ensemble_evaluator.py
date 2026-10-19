"""Tests for the ensemble semantic evaluator.

Fake child evaluators make the combination logic, graceful degradation, and curve
selection testable with no model, GPU, or video.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from semantic.ensemble_evaluator import EnsembleChild, EnsembleEvaluator, create_evaluator
from utils.semantic_evaluator import SemanticEpisode, SemanticResult, normalize_result


class FakeEvaluator:
    def __init__(self, name, score=None, curve=(), fail=False):
        self.name = name
        self._score = score
        self._curve = tuple(curve)
        self._fail = fail

    def evaluate(self, episode):
        if self._fail:
            raise RuntimeError(f"{self.name} cannot run here")
        return SemanticResult(
            progress_curve=self._curve,
            success_probability=self._score,
            details={"method": self.name},
        )


def episode():
    return SemanticEpisode(
        dataset_root=Path("/d"), episode_index=0, length=10,
        tasks=("Table Clean",), videos=(),
    )


class CombineTest(unittest.TestCase):
    def test_min_is_conservative(self):
        ev = EnsembleEvaluator(
            [EnsembleChild("a", FakeEvaluator("a", score=0.9)),
             EnsembleChild("b", FakeEvaluator("b", score=0.2))],
            combine="min",
        )
        r = normalize_result(ev.evaluate(episode()))
        self.assertAlmostEqual(r.success_probability, 0.2)

    def test_mean(self):
        ev = EnsembleEvaluator(
            [EnsembleChild("a", FakeEvaluator("a", score=0.8)),
             EnsembleChild("b", FakeEvaluator("b", score=0.4))],
            combine="mean",
        )
        self.assertAlmostEqual(ev.evaluate(episode()).success_probability, 0.6)

    def test_weighted(self):
        ev = EnsembleEvaluator(
            [EnsembleChild("a", FakeEvaluator("a", score=1.0), weight=3.0),
             EnsembleChild("b", FakeEvaluator("b", score=0.0), weight=1.0)],
            combine="weighted",
        )
        self.assertAlmostEqual(ev.evaluate(episode()).success_probability, 0.75)


class DegradationTest(unittest.TestCase):
    def test_failed_child_is_skipped(self):
        ev = EnsembleEvaluator(
            [EnsembleChild("dead", FakeEvaluator("dead", fail=True)),
             EnsembleChild("ok", FakeEvaluator("ok", score=0.7))],
            combine="min",
        )
        r = ev.evaluate(episode())
        self.assertEqual(r.details["n_children_ok"], 1)
        self.assertAlmostEqual(r.success_probability, 0.7)
        reports = {c["name"]: c for c in r.details["children"]}
        self.assertFalse(reports["dead"]["ok"])
        self.assertIn("error", reports["dead"])

    def test_require_not_met_raises(self):
        ev = EnsembleEvaluator(
            [EnsembleChild("dead", FakeEvaluator("dead", fail=True)),
             EnsembleChild("ok", FakeEvaluator("ok", score=0.7))],
            combine="min", require=2,
        )
        with self.assertRaises(RuntimeError):
            ev.evaluate(episode())

    def test_all_children_fail_raises(self):
        ev = EnsembleEvaluator(
            [EnsembleChild("a", FakeEvaluator("a", fail=True))], combine="min",
        )
        with self.assertRaises(RuntimeError):
            ev.evaluate(episode())


class CurveSelectionTest(unittest.TestCase):
    def test_curve_from_named_child(self):
        ev = EnsembleEvaluator(
            [EnsembleChild("motion", FakeEvaluator("motion", score=0.6, curve=(0.1, 0.5, 0.9))),
             EnsembleChild("clip", FakeEvaluator("clip", score=0.6, curve=(0.2, 0.2)))],
            combine="min", curve_from="clip",
        )
        self.assertEqual(ev.evaluate(episode()).progress_curve, (0.2, 0.2))

    def test_falls_back_to_first_curve(self):
        ev = EnsembleEvaluator(
            [EnsembleChild("noc", FakeEvaluator("noc", score=0.6, curve=())),
             EnsembleChild("withc", FakeEvaluator("withc", score=0.6, curve=(0.3, 0.7)))],
            combine="min",
        )
        self.assertEqual(ev.evaluate(episode()).progress_curve, (0.3, 0.7))


class FactoryTest(unittest.TestCase):
    def test_builds_children_from_specs(self):
        cfg = {
            "combine": "min",
            "curve_from": "motion",
            "children": [
                {"name": "motion", "spec": "semantic.motion_evaluator:create_evaluator"},
            ],
        }
        ev = create_evaluator(cfg)
        self.assertEqual(len(ev.children), 1)
        self.assertEqual(ev.children[0].name, "motion")

    def test_empty_children_raises(self):
        with self.assertRaises(ValueError):
            create_evaluator({"children": []})

    def test_bad_combine_raises(self):
        with self.assertRaises(ValueError):
            EnsembleEvaluator([EnsembleChild("a", FakeEvaluator("a", score=0.5))], combine="bogus")


if __name__ == "__main__":
    unittest.main()
