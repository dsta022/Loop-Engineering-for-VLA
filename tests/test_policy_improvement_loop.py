from __future__ import annotations

import csv
import json
import tempfile
import time
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from pipeline.data_quality_audit import audit_dataset
from pipeline.feedback_dataset_build import FeedbackBuildConfig, run_feedback_build
from policy_improvement.capture import (
    AsyncTriggerMonitor,
    BoundedCaptureBuffer,
    CapturedPacket,
    estimate_payload_bytes,
)
from policy_improvement.contracts import (
    CaptureMode,
    ControlPhase,
    EpisodeFeedback,
    FeedbackFrame,
    FeedbackSidecar,
    FeedbackSidecarWriter,
    InterventionSegment,
    ReviewStatus,
    TriggerSource,
)
from policy_improvement.intervention import InterventionController
from policy_improvement.plugins import PromotionPolicy
from pipeline.policy_iteration import IterationConfig, run_policy_iteration
from tests.test_dataset_etl_pipeline import write_rgb_dataset


def write_feedback_sidecar(
    root: Path,
    dataset: Path,
    length: int,
    review_status: ReviewStatus = ReviewStatus.APPROVED,
) -> Path:
    frames = [
        FeedbackFrame(
            episode_index=0,
            frame_index=frame_index,
            timestamp=frame_index / 30.0,
            phase=ControlPhase.RECOVERY if frame_index == 0 else ControlPhase.CORRECTION,
            intervention_id=0,
            trigger_source=TriggerSource.MANUAL,
            policy_action={"joint": float(frame_index)},
            executed_action={"joint": float(frame_index + 1)},
            training_eligible=True,
        )
        for frame_index in range(length)
    ]
    with FeedbackSidecarWriter(
        root,
        session_id="test-session",
        source_dataset=dataset,
        source_policy="policy-v1",
        capture_mode=CaptureMode.CORRECTIONS_ONLY,
        fps=30,
    ) as writer:
        writer.append_episode(
            frames,
            EpisodeFeedback(
                episode_index=0,
                length=length,
                source_policy="policy-v1",
                capture_mode=CaptureMode.CORRECTIONS_ONLY,
                outcome="success_after_intervention",
                intervention_count=1,
                review_status=review_status,
            ),
            [
                InterventionSegment(
                    episode_index=0,
                    intervention_id=0,
                    recovery_start_frame=0,
                    recovery_end_frame=0,
                    correction_start_frame=1,
                    correction_end_frame=length - 1,
                )
            ],
        )
    return root


def write_keep_report(path: Path, dataset: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode_index", "status"])
        writer.writeheader()
        writer.writerow({"episode_index": 0, "status": "keep"})
    (path.parent / "quality_summary.json").write_text(
        json.dumps({"dataset": str(dataset.resolve())}), encoding="utf-8"
    )


class DeterministicTrigger:
    name = "deterministic_trigger"

    def evaluate_window(self, packets):
        return {
            "triggered": len(packets) >= 2,
            "source": "semantic",
            "score": 0.2,
            "reason": "low_progress",
        }


class TestTrainer:
    name = "test_trainer"

    def train(self, request):
        request.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = request.output_dir / "checkpoint"
        checkpoint.mkdir()
        return {"checkpoint": str(checkpoint), "metrics": {"loss": 0.1}}


class TestEvaluator:
    name = "test_evaluator"

    def evaluate(self, request):
        return {
            "metrics": {
                "success_rate": 0.9,
                "intervention_rate": 0.1,
                "critical_failures": 0,
            }
        }


class TestDeployer:
    name = "test_deployer"

    def deploy(self, request):
        return {"deployment_id": "deployment-1", "target": request.target}


class PolicyImprovementTests(unittest.TestCase):
    def test_intervention_state_machine_requires_recovery_and_correction(self) -> None:
        controller = InterventionController()
        intervention_id = controller.request_takeover()
        self.assertEqual(intervention_id, 0)
        self.assertEqual(controller.phase, ControlPhase.SYNCING)
        controller.confirm_synchronized(0)
        decision = controller.select_action({"joint": 0.0}, {"joint": 1.0})
        self.assertEqual(decision.phase, ControlPhase.RECOVERY)
        self.assertTrue(decision.training_eligible)
        controller.begin_correction(2)
        window = controller.complete(5)
        self.assertEqual(window.recovery_end_frame, 1)
        self.assertEqual(window.correction_start_frame, 2)
        self.assertEqual(window.correction_end_frame, 4)
        self.assertEqual(controller.phase, ControlPhase.AUTONOMOUS)

    def test_capture_buffer_is_bounded_and_trigger_runs_asynchronously(self) -> None:
        buffer = BoundedCaptureBuffer(max_seconds=0.2, fps=10, max_memory_mb=1)
        for index in range(4):
            buffer.append(CapturedPacket(float(index), {"frame": bytes(10)}))
        self.assertEqual(len(buffer), 2)
        self.assertEqual([item.timestamp for item in buffer.snapshot()], [2.0, 3.0])
        with AsyncTriggerMonitor(DeterministicTrigger()) as monitor:
            self.assertTrue(monitor.submit(buffer.snapshot()))
            signal = None
            for _ in range(100):
                signal = monitor.poll()
                if signal is not None:
                    break
                time.sleep(0.001)
        self.assertIsNotNone(signal)
        self.assertTrue(signal.triggered)
        self.assertEqual(signal.source, TriggerSource.SEMANTIC)

    def test_capture_buffer_accounts_dataclass_packet_payloads(self) -> None:
        from dataclasses import dataclass, field

        @dataclass(frozen=True)
        class FramePacket:
            observation: dict
            executed_action: tuple
            details: dict = field(default_factory=dict)

        payload = FramePacket(observation={"camera": bytes(256 * 1024)}, executed_action=(1.0, 2.0))
        self.assertGreaterEqual(estimate_payload_bytes(payload), 256 * 1024)

        buffer = BoundedCaptureBuffer(max_seconds=10, fps=10, max_memory_mb=1)
        for index in range(8):
            buffer.append(CapturedPacket(float(index), payload))
        self.assertLessEqual(buffer.memory_bytes, buffer.max_bytes)
        self.assertLess(len(buffer), 8)

    def test_feedback_sidecar_is_audited_and_builds_weighted_training_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = write_rgb_dataset(root, "base_dataset", [2], 10)
            feedback = write_rgb_dataset(root, "feedback_dataset", [4], 100)
            sidecar_path = write_feedback_sidecar(root / "feedback_sidecar", feedback, 4)
            sidecar = FeedbackSidecar.load(sidecar_path)
            self.assertEqual(sidecar.training_eligible_episodes(), [0])

            quality_dir = root / "quality"
            result = audit_dataset(
                root=feedback,
                out_dir=quality_dir,
                depth_check="header",
                video_check="metadata",
                video_content_check="none",
                sample_frames=0,
                depth_sample_frames=0,
                duplicate_check="none",
                video_sample_stride=30,
                feedback_sidecar=sidecar,
            )
            self.assertEqual(result, 0)
            summary = json.loads((quality_dir / "quality_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["feedback_validation"]["enabled"])
            self.assertEqual(summary["feedback_validation"]["training_eligible_frames"], 4)

            report = root / "feedback_quality" / "quality_report.csv"
            write_keep_report(report, feedback)
            output = root / "iteration_dataset"
            run_feedback_build(
                FeedbackBuildConfig(
                    base_datasets=(base,),
                    feedback_dataset=feedback,
                    feedback_sidecar=sidecar_path,
                    quality_report=report,
                    out_dir=output,
                )
            )
            info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["total_episodes"], 2)
            weights = pq.read_table(output / "training_sidecars" / "training_weights.parquet")
            rows = weights.to_pylist()
            self.assertEqual(len(rows), 6)
            feedback_rows = [row for row in rows if row["source"] == "human_feedback"]
            self.assertEqual(feedback_rows[0]["phase"], "recovery")
            self.assertEqual(feedback_rows[0]["weight"], 1.0)
            self.assertEqual(feedback_rows[-1]["phase"], "correction")
            self.assertEqual(feedback_rows[-1]["weight"], 2.0)
            manifest = json.loads(
                (output / "feedback_build_manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["autonomous_failure_actions_used_as_expert_targets"])

    def test_pending_feedback_requires_human_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feedback = write_rgb_dataset(root, "feedback_dataset", [4], 100)
            sidecar_path = write_feedback_sidecar(
                root / "feedback_sidecar",
                feedback,
                4,
                review_status=ReviewStatus.PENDING,
            )
            quality_dir = root / "quality"
            result = audit_dataset(
                root=feedback,
                out_dir=quality_dir,
                depth_check="header",
                video_check="metadata",
                video_content_check="none",
                sample_frames=0,
                depth_sample_frames=0,
                duplicate_check="none",
                video_sample_stride=30,
                feedback_sidecar=FeedbackSidecar.load(sidecar_path),
            )
            self.assertEqual(result, 0)
            summary = json.loads((quality_dir / "quality_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["review"], 1)
            with (quality_dir / "quality_report.csv").open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["status"], "review")
            self.assertIn("feedback_human_approval_required", row["review_reasons"])

    def test_policy_iteration_requires_metrics_and_human_approval_before_deploy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            approval = root / "approval.json"
            approval.write_text(
                json.dumps({"iteration_id": "iteration-1", "approved": True}),
                encoding="utf-8",
            )
            config = IterationConfig(
                iteration_id="iteration-1",
                base_policy="policy-v1",
                training_dataset=dataset,
                output_dir=root / "iteration-output",
                trainer_spec=f"{__name__}:TestTrainer",
                evaluator_spec=f"{__name__}:TestEvaluator",
                deployer_spec=f"{__name__}:TestDeployer",
                deployment_target="robot-lab",
                evaluation_dataset=None,
                approval_file=approval,
                trainer_init={},
                trainer_request={},
                evaluator_init={},
                evaluator_request={},
                deployer_init={},
                deployer_request={},
                promotion=PromotionPolicy(
                    min_success_rate=0.8,
                    max_intervention_rate=0.2,
                    max_critical_failures=0,
                    require_human_approval=True,
                ),
            )
            self.assertEqual(run_policy_iteration(config), 0)
            manifest = json.loads(
                (config.output_dir / "iteration_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "deployed")
            self.assertTrue(manifest["promotion"]["approved"])


if __name__ == "__main__":
    unittest.main()
