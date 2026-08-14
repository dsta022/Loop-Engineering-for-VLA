from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tests.test_dataset_etl_pipeline import write_rgb_dataset
from pipeline.trajectory_language_enrichment import (
    LanguageEnrichmentConfig,
    run_language_enrichment,
    select_episode_indices,
)
from tools.trajectory_language_studio import build_enrichment_command
from utils.language_annotation_profile import (
    LanguageAnnotationEpisode,
    normalize_annotation_result,
)
from utils.task_semantic_profile import TaskProfileRegistry, TaskSemanticProfile


class DeterministicLanguageAnnotator:
    name = "deterministic_language_annotator"

    def __init__(self) -> None:
        self.seen: list[int] = []

    def annotate(self, episode):
        self.seen.append(episode.episode_index)
        if episode.task_profile is None:
            raise AssertionError("Task Profile was not passed to the annotator")
        split = max(1, episode.length // 2)
        return {
            "subtasks": [
                {
                    "start_frame": 0,
                    "end_frame": split - 1,
                    "stage": "approach",
                    "text": "approach the cups",
                    "confidence": 0.95,
                },
                {
                    "start_frame": split,
                    "end_frame": episode.length - 1,
                    "stage": "grasp",
                    "text": "grasp and lift the cups",
                    "confidence": 0.91,
                },
            ],
            "events": [
                {
                    "frame_index": split,
                    "event": "grasp_started",
                    "description": "The gripper starts closing around the cups.",
                    "confidence": 0.90,
                }
            ],
            "state_descriptions": [
                {
                    "start_frame": 0,
                    "end_frame": episode.length - 1,
                    "level": "state",
                    "text": "The robot moves from approach to grasp.",
                }
            ],
            "summary": "The robot approaches, grasps, and lifts the cups.",
            "details": {"semantic_score": episode.semantic_context.get("semantic_score")},
        }


def write_quality_report(path: Path) -> None:
    rows = [
        (0, "keep", "pass", "0.96"),
        (1, "review", "review", "0.75"),
        (2, "drop", "skipped_structural_drop", ""),
    ]
    fields = [
        "episode_index",
        "status",
        "structural_status",
        "semantic_status",
        "semantic_evaluator",
        "semantic_progress_curve_json",
        "semantic_success_probability",
        "semantic_score",
        "semantic_failure_stage",
        "review_reasons",
        "metrics_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for episode_index, status, semantic_status, score in rows:
            writer.writerow(
                {
                    "episode_index": episode_index,
                    "status": status,
                    "structural_status": "drop" if status == "drop" else "keep",
                    "semantic_status": semantic_status,
                    "semantic_evaluator": "test_evaluator",
                    "semantic_progress_curve_json": "[0.1, 0.9]",
                    "semantic_success_probability": score,
                    "semantic_score": score,
                    "semantic_failure_stage": "",
                    "review_reasons": "semantic_uncertain_score_review" if status == "review" else "",
                    "metrics_json": "{}",
                }
            )


def test_profile_registry() -> TaskProfileRegistry:
    profile = TaskSemanticProfile.from_dict(
        {
            "task_key": "pick_up_cups",
            "task_name": "pick up cups",
            "instruction": "pick up cups",
            "stages": [
                {
                    "name": "approach",
                    "description": "Approach the cups",
                    "observable_condition": "The gripper is near the cups",
                },
                {
                    "name": "grasp",
                    "description": "Grasp and lift the cups",
                    "observable_condition": "The cups are held above the table",
                },
            ],
            "success_criteria": ["The cups are lifted"],
            "failure_criteria": ["The grasp is unstable"],
        }
    )
    return TaskProfileRegistry((profile,))


class TrajectoryLanguageEnrichmentTests(unittest.TestCase):
    def test_default_selection_annotates_only_keep_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_rgb_dataset(root, "language_dataset", [6, 6, 6], 10)
            report = root / "quality" / "quality_report.csv"
            write_quality_report(report)
            (report.parent / "quality_summary.json").write_text(
                json.dumps({"dataset": str(dataset.resolve())}), encoding="utf-8"
            )
            out_dir = root / "language_output"
            annotator = DeterministicLanguageAnnotator()

            exit_code = run_language_enrichment(
                LanguageEnrichmentConfig(
                    dataset_root=dataset,
                    quality_report=report,
                    out_dir=out_dir,
                    annotator=annotator,
                    task_profiles=test_profile_registry(),
                )
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(annotator.seen, [0])
            annotation = json.loads(
                (out_dir / "annotations" / "episode-000000.json").read_text(encoding="utf-8")
            )
            self.assertEqual(annotation["task_profile"], "pick_up_cups")
            self.assertEqual(annotation["subtasks"][1]["stage"], "grasp")
            self.assertEqual(annotation["events"][0]["event"], "grasp_started")
            self.assertEqual(annotation["quality_context"]["semantic_score"], 0.96)
            manifest = json.loads(
                (out_dir / "enrichment_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["selection_mode"], "quality_report_keep")
            self.assertEqual(manifest["annotated_episodes"], [0])
            self.assertTrue(manifest["quality_report_dataset_verified"])
            self.assertFalse(manifest["source_dataset_modified"])
            self.assertFalse((dataset / "language_annotations.jsonl").exists())
            self.assertEqual(
                len((out_dir / "subtasks.jsonl").read_text(encoding="utf-8").splitlines()), 2
            )

    def test_explicit_approval_can_include_review_but_rejects_drop(self) -> None:
        rows = {
            0: {"status": "keep"},
            1: {"status": "review"},
            2: {"status": "drop"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            approved = root / "approved.txt"
            approved.write_text("0\n1\n", encoding="utf-8")
            selected, mode = select_episode_indices(rows, approved)
            self.assertEqual(selected, [0, 1])
            self.assertEqual(mode, "explicit_approved_episode_list")

            approved.write_text("2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "drop episodes"):
                select_episode_indices(rows, approved)

    def test_annotation_validation_rejects_overlapping_subtasks(self) -> None:
        episode = LanguageAnnotationEpisode(
            dataset_root=Path("dataset"),
            episode_index=0,
            length=10,
            tasks=("pick up cups",),
            videos=(),
        )
        result = {
            "subtasks": [
                {"start_frame": 0, "end_frame": 5, "text": "approach"},
                {"start_frame": 5, "end_frame": 9, "text": "grasp"},
            ],
            "events": [],
            "state_descriptions": [],
            "summary": "Attempted pickup.",
        }
        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            normalize_annotation_result(result, episode, "test_annotator")

    def test_valid_annotation_does_not_require_task_profile(self) -> None:
        episode = LanguageAnnotationEpisode(
            dataset_root=Path("dataset"),
            episode_index=4,
            length=5,
            tasks=("move object",),
            videos=(),
            semantic_context={"semantic_score": 0.94},
        )
        annotation = normalize_annotation_result(
            {
                "subtasks": [
                    {"start_frame": 0, "end_frame": 4, "text": "move the object"}
                ],
                "events": [],
                "state_descriptions": [],
                "summary": "The robot moves the object.",
            },
            episode,
            "test_annotator",
        )
        self.assertEqual(annotation.task_profile, "")
        self.assertEqual(annotation.quality_context["semantic_score"], 0.94)

    def test_studio_builds_optional_sidecar_command(self) -> None:
        command = build_enrichment_command(
            dataset=Path("dataset"),
            annotator_spec="backend:create_annotator",
            quality_report=Path("quality_report.csv"),
            approved_episodes=Path("approved.txt"),
            profile_dir=Path("task_profiles"),
            out_dir=Path("language_output"),
            overwrite=True,
            python_executable="python",
        )
        self.assertIn("trajectory_language_enrichment.py", command[1])
        self.assertIn("--approved-episodes", command)
        self.assertIn("--task-profile-dir", command)
        self.assertIn("--overwrite", command)


if __name__ == "__main__":
    unittest.main()
