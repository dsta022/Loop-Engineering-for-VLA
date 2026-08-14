from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipeline.data_quality_audit import audit_dataset
from semantic_backends.vla_checkpoint_evaluator import (
    VLACheckpointEvaluator,
    chunk_skill,
    corrupt_instructions,
    create_evaluator,
    informative_anchors,
    interleaved_means,
    rename_camera_features,
)
from tests.test_dataset_etl_pipeline import RGB_KEY, write_rgb_dataset
from utils.semantic_evaluator import (
    EpisodeMotion,
    SemanticEpisode,
    VideoSegment,
    normalize_result,
)


class FakeFrameReader:
    """Returns black frames; the synthetic mp4 placeholders are never decoded."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, float]] = []

    def read_frame(self, path: Path, seconds: float, height: int, width: int) -> np.ndarray:
        self.requests.append((str(path), seconds))
        return np.zeros((height, width, 3), dtype=np.uint8)


class OraclePredictor:
    """Reconstructs the synthetic executed actions from the state observation.

    The synthetic datasets define state[t] = [base+t+1.0, base+t+1.5] and
    action[t] = [base+t, base+t+0.5], so the anchor is recoverable from
    ``state_0`` alone. Under the true instruction the oracle reproduces the
    executed chunk (skill 1.0); under any other instruction it degrades to the
    hold baseline (skill 0.0), giving a large counterfactual margin.
    """

    def __init__(self, true_task: str) -> None:
        self.true_task = true_task
        self.tasks_seen: list[str] = []

    def predict_chunk(
        self,
        observation: Mapping[str, Any],
        task: str,
        horizon: int,
        *,
        features: Mapping[str, Any],
        action_names: Sequence[str],
    ) -> np.ndarray:
        self.tasks_seen.append(task)
        start = float(observation["state_0"]) - 1.0
        if task == self.true_task:
            return np.asarray(
                [[start + step, start + 0.5 + step] for step in range(horizon)], dtype=np.float64
            )
        return np.asarray([[start, start + 0.5]] * horizon, dtype=np.float64)


class HoldPredictor:
    """Always predicts the hold baseline, i.e. a policy with zero skill."""

    def predict_chunk(
        self,
        observation: Mapping[str, Any],
        task: str,
        horizon: int,
        *,
        features: Mapping[str, Any],
        action_names: Sequence[str],
    ) -> np.ndarray:
        start = float(observation["state_0"]) - 1.0
        return np.asarray([[start, start + 0.5]] * horizon, dtype=np.float64)


def build_evaluator(predictor: Any) -> VLACheckpointEvaluator:
    return VLACheckpointEvaluator(predictor, FakeFrameReader())


def read_report(out_dir: Path) -> list[dict[str, str]]:
    with (out_dir / "quality_report.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class CorruptInstructionTests(unittest.TestCase):
    def test_english_verb_direction_color_and_noun_swaps(self) -> None:
        variants = corrupt_instructions("pick up the red cup", limit=4)
        self.assertIn("put down the red cup", variants)
        self.assertIn("pick up the blue cup", variants)
        self.assertTrue(any("bottle" in item for item in variants))
        self.assertNotIn("pick up the red cup", variants)

    def test_chinese_swaps_prefer_same_script_nouns(self) -> None:
        variants = corrupt_instructions("拿起红色杯子", limit=4)
        self.assertIn("放下红色杯子", variants)
        self.assertIn("拿起蓝色杯子", variants)
        noun_variants = [item for item in variants if "杯子" not in item]
        self.assertTrue(noun_variants)
        self.assertTrue(all(not item.isascii() for item in noun_variants))

    def test_limit_and_no_match(self) -> None:
        self.assertEqual(len(corrupt_instructions("pick up the red cup", limit=2)), 2)
        self.assertEqual(corrupt_instructions("pick up the red cup", limit=0), ())
        self.assertEqual(corrupt_instructions("do the thing"), ())

    def test_profile_objects_join_the_noun_pool(self) -> None:
        variants = corrupt_instructions("move the widget", object_names=("widget",), limit=4)
        self.assertTrue(any("widget" not in item for item in variants))


class ScoringMathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executed = np.asarray(
            [[step, step + 0.5] for step in range(12)], dtype=np.float64
        )

    def test_perfect_prediction_scores_one(self) -> None:
        predicted = self.executed[2:7]
        self.assertEqual(chunk_skill(predicted, self.executed, 2, 5), 1.0)

    def test_hold_prediction_scores_zero(self) -> None:
        predicted = np.repeat(self.executed[2:3], 5, axis=0)
        self.assertEqual(chunk_skill(predicted, self.executed, 2, 5), 0.0)

    def test_static_demonstration_is_uninformative(self) -> None:
        static = np.ones((10, 2), dtype=np.float64)
        self.assertIsNone(chunk_skill(static[:5], static, 0, 5))
        self.assertEqual(informative_anchors(static, [0, 2, 4], 5), ())

    def test_informative_anchors_keep_moving_regions(self) -> None:
        self.assertEqual(informative_anchors(self.executed, [0, 3, 6], 5), (0, 3, 6))

    def test_interleaved_means(self) -> None:
        self.assertEqual(interleaved_means([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], 3), (0.5, 0.5, 0.5))
        self.assertEqual(interleaved_means([1.0], 3), ())
        self.assertEqual(interleaved_means([1.0, 0.0], 0), ())

    def test_rename_camera_features(self) -> None:
        features = {
            "observation.images.front": {"dtype": "video"},
            "action": {"dtype": "float32"},
        }
        renamed = rename_camera_features(features, {"front": "head"})
        self.assertIn("observation.images.head", renamed)
        self.assertNotIn("observation.images.front", renamed)
        self.assertIn("action", renamed)


class EvaluatorUnitTests(unittest.TestCase):
    def test_result_passes_contract_and_separates_instructions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = write_rgb_dataset(Path(tmp), "unit_set", [40], base_value=0)
            evaluator = build_evaluator(OraclePredictor("pick up cups"))
            episode = SemanticEpisode(
                dataset_root=dataset,
                episode_index=0,
                length=40,
                tasks=("pick up cups",),
                videos=(
                    VideoSegment(
                        key=RGB_KEY,
                        path=dataset / "videos" / RGB_KEY / "chunk-000" / "file-000.mp4",
                        from_timestamp=0.0,
                        to_timestamp=40 / 30.0,
                    ),
                ),
                motion=EpisodeMotion(
                    action=np.asarray([[step, step + 0.5] for step in range(40)]),
                    state=np.asarray([[step + 1.0, step + 1.5] for step in range(40)]),
                    fps=30.0,
                ),
            )
            result = normalize_result(evaluator.evaluate(episode))
            self.assertEqual(result.score, 1.0)
            self.assertEqual(result.uncertainty, 0.0)
            self.assertEqual(result.counterfactual_margin, 1.0)
            self.assertEqual(result.progress_curve, ())
            self.assertIn("put down cups", result.details["counterfactual_scores"])

    def test_static_episode_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = write_rgb_dataset(Path(tmp), "static_set", [40], base_value=0)
            evaluator = build_evaluator(OraclePredictor("pick up cups"))
            episode = SemanticEpisode(
                dataset_root=dataset,
                episode_index=0,
                length=8,
                tasks=("pick up cups",),
                videos=(
                    VideoSegment(
                        key=RGB_KEY,
                        path=dataset / "videos" / RGB_KEY / "chunk-000" / "file-000.mp4",
                        from_timestamp=0.0,
                        to_timestamp=8 / 30.0,
                    ),
                ),
                motion=EpisodeMotion(action=np.ones((8, 2)), state=None, fps=30.0),
            )
            with self.assertRaisesRegex(ValueError, "static"):
                evaluator.evaluate(episode)

    def test_missing_motion_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = write_rgb_dataset(Path(tmp), "no_motion_set", [40], base_value=0)
            evaluator = build_evaluator(OraclePredictor("pick up cups"))
            episode = SemanticEpisode(
                dataset_root=dataset,
                episode_index=0,
                length=40,
                tasks=("pick up cups",),
                videos=(),
                motion=None,
            )
            with self.assertRaisesRegex(ValueError, "action stream"):
                evaluator.evaluate(episode)


class AuditIntegrationTests(unittest.TestCase):
    def run_audit(self, dataset: Path, out_dir: Path, predictor: Any) -> int:
        return audit_dataset(
            root=dataset,
            out_dir=out_dir,
            depth_check="none",
            video_check="metadata",
            video_content_check="none",
            sample_frames=0,
            depth_sample_frames=0,
            duplicate_check="none",
            video_sample_stride=1,
            semantic_evaluator=build_evaluator(predictor),
            motion_check="none",
        )

    def test_consistent_episodes_pass_semantic_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_rgb_dataset(root, "good_set", [40, 40], base_value=0)
            out_dir = root / "report"
            exit_code = self.run_audit(dataset, out_dir, OraclePredictor("pick up cups"))
            self.assertEqual(exit_code, 0)
            rows = read_report(out_dir)
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row["status"], "keep")
                self.assertEqual(row["semantic_status"], "pass")
                self.assertEqual(float(row["semantic_score"]), 1.0)
                self.assertEqual(float(row["semantic_counterfactual_margin"]), 1.0)
                self.assertEqual(row["semantic_progress_curve_json"], "[]")

    def test_zero_skill_policy_sends_episodes_to_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_rgb_dataset(root, "bad_set", [40, 40], base_value=0)
            out_dir = root / "report"
            self.run_audit(dataset, out_dir, HoldPredictor())
            for row in read_report(out_dir):
                self.assertEqual(row["status"], "review")
                self.assertEqual(row["semantic_status"], "review")
                self.assertIn("semantic_low_score_review", row["review_reasons"])


class FactoryTests(unittest.TestCase):
    def test_policy_path_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "policy_path"):
            create_evaluator({})

    def test_missing_checkpoint_is_rejected(self) -> None:
        with self.assertRaises(FileNotFoundError):
            create_evaluator({"policy_path": "does/not/exist"})

    def test_unknown_config_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "unknown"):
                create_evaluator({"policy_path": tmp, "frame_rate": 30})


if __name__ == "__main__":
    unittest.main()
