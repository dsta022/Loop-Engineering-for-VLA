from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from language_backends.kinematic_annotator import create_annotator, merge_segments, select_boundaries
from pipeline.policy_iteration import IterationConfig, run_policy_iteration
from pipeline.trajectory_language_enrichment import LanguageEnrichmentConfig, run_language_enrichment
from policy_backends.command_trainer import CommandTrainer
from policy_backends.filesystem_deployer import FilesystemDeployer
from policy_backends.offline_evaluator import OfflineConsistencyEvaluator
from policy_improvement.plugins import (
    DeploymentRequest,
    EvaluationRequest,
    PromotionPolicy,
    TrainingRequest,
)
from semantic_backends.vla_checkpoint_evaluator import VLACheckpointEvaluator
from tests.test_dataset_etl_pipeline import write_rgb_dataset
from tests.test_semantic_consistency import (
    three_stage_profile,
    write_minimal_quality_report,
    write_motion_dataset,
)
from tests.test_vla_checkpoint_evaluator import FakeFrameReader, OraclePredictor
from utils.language_annotation_profile import LanguageAnnotationEpisode, normalize_annotation_result
from utils.semantic_evaluator import EpisodeMotion

MAKE_CHECKPOINT = (
    "import json, pathlib, sys; "
    "p = pathlib.Path(sys.argv[1]); p.mkdir(parents=True, exist_ok=True); "
    "(p / 'ok').write_text('1'); "
    "pathlib.Path(sys.argv[2]).write_text(json.dumps(dict(loss=0.5)))"
)


def oracle_factory(config: Mapping[str, Any]) -> VLACheckpointEvaluator:
    assert Path(config["policy_path"]).name == "checkpoint-under-test", config
    return VLACheckpointEvaluator(OraclePredictor("pick up cups"), FakeFrameReader())


def create_test_offline_evaluator(config: Mapping[str, Any]) -> OfflineConsistencyEvaluator:
    return OfflineConsistencyEvaluator(evaluator_factory=oracle_factory, **dict(config))


class KinematicAnnotatorTests(unittest.TestCase):
    @staticmethod
    def one_cycle_motion(length: int = 60) -> EpisodeMotion:
        arm = np.linspace(0.0, 1.0, length)
        gripper = np.zeros(length)
        gripper[20:40] = 1.0
        return EpisodeMotion(action=np.stack([arm, gripper], axis=1), fps=30.0)

    def episode(self, task_profile: dict | None) -> LanguageAnnotationEpisode:
        return LanguageAnnotationEpisode(
            dataset_root=Path("dataset"),
            episode_index=0,
            length=60,
            tasks=("pick up the cup",),
            videos=(),
            task_profile=task_profile,
            motion=self.one_cycle_motion(),
        )

    def test_segments_follow_gripper_events_and_profile_stages(self) -> None:
        episode = self.episode(three_stage_profile().to_dict())
        result = create_annotator({}).annotate(episode)
        annotation = normalize_annotation_result(result, episode, "test", "pick_up_cups")
        self.assertEqual([item.stage for item in annotation.subtasks], ["approach", "grasp", "lift"])
        self.assertEqual(
            [(item.start_frame, item.end_frame) for item in annotation.subtasks],
            [(0, 19), (20, 39), (40, 59)],
        )
        self.assertEqual([item.frame_index for item in annotation.events], [20, 40])
        self.assertEqual(result["details"]["gripper_index"], 1)
        self.assertEqual(result["details"]["stage_source"], "task_profile")
        self.assertEqual(len(annotation.state_descriptions), 3)

    def test_without_profile_segments_get_generic_text(self) -> None:
        episode = self.episode(None)
        annotation = normalize_annotation_result(create_annotator({}).annotate(episode), episode, "test")
        self.assertEqual([item.stage for item in annotation.subtasks], ["", "", ""])
        self.assertIn("segment 1 of 3", annotation.subtasks[0].text)

    def test_extra_segments_are_merged_down_to_the_stage_count(self) -> None:
        profile = three_stage_profile().to_dict()
        profile["stages"] = profile["stages"][:2]
        episode = self.episode(profile)
        result = create_annotator({}).annotate(episode)
        self.assertEqual(len(result["subtasks"]), 2)
        self.assertEqual(result["subtasks"][-1]["end_frame"], 59)
        self.assertEqual(result["subtasks"][0]["start_frame"], 0)

    def test_helpers(self) -> None:
        self.assertEqual(select_boundaries([2, 20, 21, 40, 59], 60, 5), [20, 40])
        self.assertEqual(merge_segments([(0, 9), (10, 12), (13, 29)], 2), [(0, 12), (13, 29)])
        with self.assertRaisesRegex(ValueError, "unknown"):
            create_annotator({"nope": 1})

    def test_missing_motion_is_an_error(self) -> None:
        episode = LanguageAnnotationEpisode(
            dataset_root=Path("dataset"), episode_index=0, length=10, tasks=("t",), videos=()
        )
        with self.assertRaisesRegex(ValueError, "action stream"):
            create_annotator({}).annotate(episode)

    def test_enrichment_pipeline_marks_kinematic_boundaries_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_motion_dataset(root, "kinematic", "pick up the cup", gripper_cycles=[1])
            report_dir = root / "quality"
            write_minimal_quality_report(report_dir, dataset, [60])
            out_dir = root / "language"
            exit_code = run_language_enrichment(
                LanguageEnrichmentConfig(
                    dataset_root=dataset,
                    quality_report=report_dir / "quality_report.csv",
                    out_dir=out_dir,
                    annotator=create_annotator({}),
                )
            )
            self.assertEqual(exit_code, 0)
            rows = [
                json.loads(line)
                for line in (out_dir / "subtask_boundary_alignment.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(rows[0]["status"], "aligned")
            manifest = json.loads((out_dir / "enrichment_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["annotator"], "kinematic_segment_annotator")
            self.assertEqual(manifest["annotated_episodes"], [0])


class CommandTrainerTests(unittest.TestCase):
    def test_runs_command_and_returns_checkpoint_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "training"
            trainer = CommandTrainer(
                command=[sys.executable, "-c", MAKE_CHECKPOINT, "{output_dir}/ckpt", "{output_dir}/metrics.json"],
                checkpoint="{output_dir}/ckpt",
                metrics_file="{output_dir}/metrics.json",
            )
            result = trainer.train(
                TrainingRequest(
                    iteration_id="it-1",
                    base_policy="base",
                    training_dataset=Path(tmp) / "dataset",
                    output_dir=output,
                    config={"steps": 5},
                )
            )
            self.assertTrue(Path(result["checkpoint"]).is_dir())
            self.assertEqual(result["metrics"], {"loss": 0.5})
            self.assertEqual(result["details"]["returncode"], 0)

    def test_failing_command_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = CommandTrainer(command=[sys.executable, "-c", "raise SystemExit(3)"], checkpoint="{output_dir}/x")
            with self.assertRaisesRegex(RuntimeError, "exit code 3"):
                trainer.train(
                    TrainingRequest("it", "base", Path(tmp), Path(tmp) / "out")
                )

    def test_missing_checkpoint_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = CommandTrainer(command=[sys.executable, "-c", "pass"], checkpoint="{output_dir}/never")
            with self.assertRaisesRegex(RuntimeError, "no checkpoint"):
                trainer.train(TrainingRequest("it", "base", Path(tmp), Path(tmp) / "out"))


class FilesystemDeployerTests(unittest.TestCase):
    def test_copies_checkpoint_and_updates_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "ckpt"
            checkpoint.mkdir()
            (checkpoint / "model.bin").write_bytes(b"weights")
            deployer = FilesystemDeployer(target_root=str(root / "deployments"))
            request = DeploymentRequest(iteration_id="it-1", checkpoint=str(checkpoint), target="lab-robot")
            result = deployer.deploy(request)
            self.assertEqual(result["deployment_id"], "lab-robot/it-1")
            deployed = Path(result["details"]["checkpoint"])
            self.assertEqual((deployed / "model.bin").read_bytes(), b"weights")
            pointer = json.loads((root / "deployments" / "lab-robot" / "current.json").read_text(encoding="utf-8"))
            self.assertEqual(pointer["iteration_id"], "it-1")
            with self.assertRaises(FileExistsError):
                deployer.deploy(request)
            FilesystemDeployer(target_root=str(root / "deployments"), overwrite=True).deploy(request)

    def test_rejects_unsafe_target_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "ckpt"
            checkpoint.write_text("x")
            deployer = FilesystemDeployer(target_root=tmp)
            with self.assertRaisesRegex(ValueError, "simple directory name"):
                deployer.deploy(DeploymentRequest("it", str(checkpoint), "../escape"))


class OfflineEvaluatorTests(unittest.TestCase):
    def test_scores_every_episode_with_the_injected_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = write_rgb_dataset(Path(tmp), "eval_set", [40, 40], base_value=0)
            evaluator = OfflineConsistencyEvaluator(
                evaluator_factory=oracle_factory,
                extra_metrics={"intervention_rate": 0.0, "critical_failures": 0},
            )
            result = evaluator.evaluate(
                EvaluationRequest(iteration_id="it", checkpoint="checkpoint-under-test", evaluation_dataset=dataset)
            )
            metrics = result["metrics"]
            self.assertEqual(metrics["success_rate"], 1.0)
            self.assertEqual(metrics["episodes_scored"], 2.0)
            self.assertEqual(metrics["episodes_failed"], 0.0)
            self.assertEqual(metrics["mean_counterfactual_margin"], 1.0)
            self.assertEqual(metrics["intervention_rate"], 0.0)
            self.assertTrue(result["details"]["offline_proxy"])

    def test_requires_an_evaluation_dataset(self) -> None:
        evaluator = OfflineConsistencyEvaluator(evaluator_factory=oracle_factory)
        with self.assertRaisesRegex(ValueError, "evaluation_dataset"):
            evaluator.evaluate(EvaluationRequest("it", "checkpoint-under-test", None))


class ReferenceIterationTests(unittest.TestCase):
    def test_reference_backends_complete_a_full_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training = write_rgb_dataset(root, "train_set", [40], base_value=0)
            evaluation = write_rgb_dataset(root, "eval_set", [40, 40], base_value=0)
            approval = root / "approval.json"
            approval.write_text(json.dumps({"iteration_id": "it-1", "approved": True}), encoding="utf-8")
            config = IterationConfig(
                iteration_id="it-1",
                base_policy="policy-v0",
                training_dataset=training,
                output_dir=root / "iteration",
                trainer_spec="policy_backends.command_trainer:create_trainer",
                evaluator_spec=f"{__name__}:create_test_offline_evaluator",
                deployer_spec="policy_backends.filesystem_deployer:create_deployer",
                deployment_target="lab-robot",
                evaluation_dataset=evaluation,
                approval_file=approval,
                trainer_init={
                    "command": [sys.executable, "-c", MAKE_CHECKPOINT, "{output_dir}/checkpoint-under-test", "{output_dir}/metrics.json"],
                    "checkpoint": "{output_dir}/checkpoint-under-test",
                    "metrics_file": "{output_dir}/metrics.json",
                },
                trainer_request={"steps": 10},
                evaluator_init={"extra_metrics": {"intervention_rate": 0.0, "critical_failures": 0}},
                evaluator_request={},
                deployer_init={"target_root": str(root / "deployments")},
                deployer_request={},
                promotion=PromotionPolicy(min_success_rate=0.8, max_intervention_rate=0.2, max_critical_failures=0),
            )
            exit_code = run_policy_iteration(config)
            manifest = json.loads((root / "iteration" / "iteration_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0, manifest)
            self.assertEqual(manifest["status"], "deployed")
            self.assertEqual(manifest["evaluation"]["metrics"]["success_rate"], 1.0)
            self.assertTrue((root / "deployments" / "lab-robot" / "current.json").exists())


if __name__ == "__main__":
    unittest.main()
