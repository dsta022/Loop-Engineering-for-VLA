from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.data_quality_audit import audit_dataset, audit_dataset_semantic_reuse
from utils.semantic_consistency import (
    analyze_progress_curve,
    boundary_alignment_findings,
    build_stage_timeline,
    channel_traces,
    dwell_outliers,
    gripper_pattern_findings,
    gripper_trace,
    kinematic_change_points,
    progress_curve_findings,
    resolve_actuation,
    resolve_motion_signature,
    select_gripper_index,
    stage_timeline_findings,
    subtask_boundary_alignment,
    vertical_direction_findings,
    vertical_excursions,
)
from utils.semantic_evaluator import EpisodeMotion, SemanticResult, normalize_result
from utils.task_semantic_profile import TaskSemanticProfile


RGB_KEY = "observation.images.front"


def three_stage_profile(**overrides) -> TaskSemanticProfile:
    payload = {
        "task_key": "pick_up_cups",
        "task_name": "pick up cups",
        "instruction": "Pick up the cups",
        "aliases": ["pick up cups"],
        "stages": [
            {
                "name": "approach",
                "description": "Approach the cups",
                "observable_condition": "The gripper moves near the cups",
            },
            {
                "name": "grasp",
                "description": "Grasp a cup",
                "observable_condition": "A cup is held by the gripper",
            },
            {
                "name": "lift",
                "description": "Lift the cup",
                "observable_condition": "The cup leaves the table",
            },
        ],
        "success_criteria": ["The cup is lifted"],
        "failure_criteria": ["The cup remains on the table"],
    }
    payload.update(overrides)
    return TaskSemanticProfile.from_dict(payload)


class ProgressCurveShapeTests(unittest.TestCase):
    def test_empty_curve_has_no_metrics(self) -> None:
        self.assertIsNone(analyze_progress_curve([]))

    def test_monotonic_curve_produces_no_findings(self) -> None:
        metrics = analyze_progress_curve([0.1, 0.3, 0.6, 0.85, 0.99])
        assert metrics is not None
        self.assertTrue(metrics.monotonic)
        self.assertEqual(metrics.regression_count, 0)
        self.assertEqual(metrics.max_drop, 0.0)
        self.assertEqual(progress_curve_findings(metrics), ())

    def test_mid_episode_drop_is_reviewed_even_when_the_final_value_recovers(self) -> None:
        metrics = analyze_progress_curve([0.1, 0.7, 0.2, 0.6, 0.98])
        assert metrics is not None
        self.assertEqual(metrics.regression_count, 1)
        self.assertAlmostEqual(metrics.max_drop, 0.5)
        self.assertIn("semantic_progress_regression_review", progress_curve_findings(metrics))

    def test_early_saturation_flags_an_idle_tail(self) -> None:
        metrics = analyze_progress_curve([0.2, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        assert metrics is not None
        self.assertAlmostEqual(metrics.saturation_ratio, 1 / 9)
        self.assertGreaterEqual(metrics.idle_tail_ratio, 0.30)
        self.assertIn("semantic_progress_idle_tail_review", progress_curve_findings(metrics))

    def test_late_onset_flags_an_idle_head(self) -> None:
        metrics = analyze_progress_curve([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.6, 0.9, 1.0])
        assert metrics is not None
        self.assertAlmostEqual(metrics.onset_ratio, 6 / 9)
        self.assertIn("semantic_progress_idle_head_review", progress_curve_findings(metrics))

    def test_short_curves_skip_the_idle_checks(self) -> None:
        metrics = analyze_progress_curve([0.99, 1.0, 1.0])
        assert metrics is not None
        self.assertEqual(progress_curve_findings(metrics), ())


class StageTimelineTests(unittest.TestCase):
    def test_derived_timeline_orders_stages_by_cumulative_weight(self) -> None:
        profile = three_stage_profile()
        timeline = build_stage_timeline(profile, [0.1, 0.5, 0.9])
        assert timeline is not None
        self.assertEqual(timeline.source, "derived_from_progress")
        self.assertEqual(timeline.sequence, ("approach", "grasp", "lift"))
        self.assertEqual(stage_timeline_findings(timeline), ())
        self.assertEqual(set(timeline.dwell_ratios), {"approach", "grasp", "lift"})

    def test_derived_timeline_ignores_a_boundary_crossing_inside_curve_noise(self) -> None:
        profile = three_stage_profile()
        timeline = build_stage_timeline(profile, [0.1, 0.35, 0.32, 0.9])
        assert timeline is not None
        self.assertEqual(timeline.order_violations, ())
        self.assertEqual(stage_timeline_findings(timeline), ())

    def test_derived_timeline_reports_a_large_backwards_step(self) -> None:
        profile = three_stage_profile()
        timeline = build_stage_timeline(profile, [0.1, 0.9, 0.2, 0.95])
        assert timeline is not None
        self.assertEqual(len(timeline.order_violations), 1)
        self.assertIn("semantic_stage_order_violation_review", stage_timeline_findings(timeline))

    def test_evaluator_timeline_wins_over_the_derived_one(self) -> None:
        profile = three_stage_profile()
        timeline = build_stage_timeline(
            profile,
            [0.1, 0.5, 0.9],
            {"stage_timeline": ["approach", "lift", "grasp"]},
        )
        assert timeline is not None
        self.assertEqual(timeline.source, "evaluator")
        self.assertEqual(timeline.sequence, ("approach", "lift", "grasp"))
        self.assertIn("semantic_stage_order_violation_review", stage_timeline_findings(timeline))

    def test_evaluator_timeline_accepts_index_spans(self) -> None:
        profile = three_stage_profile()
        timeline = build_stage_timeline(
            profile,
            [0.1, 0.2, 0.5, 0.9],
            {
                "stage_timeline": [
                    {"stage": "approach", "start_index": 0, "end_index": 1},
                    {"stage": "grasp", "start_index": 2, "end_index": 2},
                    {"stage": "lift", "start_index": 3, "end_index": 3},
                ]
            },
        )
        assert timeline is not None
        self.assertEqual(timeline.source, "evaluator")
        self.assertEqual(timeline.sequence, ("approach", "approach", "grasp", "lift"))
        self.assertAlmostEqual(timeline.dwell_ratios["approach"], 0.5)

    def test_unknown_stage_name_from_the_evaluator_is_reviewed(self) -> None:
        profile = three_stage_profile()
        timeline = build_stage_timeline(
            profile, [0.1, 0.5, 0.9], {"stage_timeline": ["approach", "hover", "lift"]}
        )
        assert timeline is not None
        self.assertEqual(timeline.unknown_stages, ("hover",))
        self.assertIn("semantic_stage_unknown_review", stage_timeline_findings(timeline))

    def test_mismatched_explicit_timeline_length_falls_back_to_derivation(self) -> None:
        profile = three_stage_profile()
        timeline = build_stage_timeline(
            profile, [0.1, 0.5, 0.9], {"stage_timeline": ["approach"]}
        )
        assert timeline is not None
        self.assertEqual(timeline.source, "derived_from_progress")

    def test_skipped_middle_stage_is_reported_on_a_long_curve(self) -> None:
        profile = three_stage_profile()
        curve = [0.05] * 4 + [0.95] * 4
        timeline = build_stage_timeline(profile, curve)
        assert timeline is not None
        self.assertEqual(timeline.skipped_stages, ("grasp",))
        self.assertIn("semantic_stage_skipped_review", stage_timeline_findings(timeline))

    def test_profile_without_progress_has_no_timeline(self) -> None:
        self.assertIsNone(build_stage_timeline(three_stage_profile(), []))


class StageDwellOutlierTests(unittest.TestCase):
    def test_tight_distribution_produces_no_outliers(self) -> None:
        samples = {("pick_up_cups", "grasp"): [(i, 0.30 + 0.01 * i) for i in range(8)]}
        self.assertEqual(dwell_outliers(samples), {})

    def test_a_long_dwell_is_flagged_against_its_task_stage_median(self) -> None:
        entries = [(i, 0.30) for i in range(8)] + [(99, 0.85)]
        flagged = dwell_outliers({("pick_up_cups", "grasp"): entries})
        self.assertEqual(list(flagged), [99])
        self.assertIn("long", flagged[99][0])

    def test_too_few_samples_are_left_alone(self) -> None:
        entries = [(0, 0.30), (1, 0.31), (2, 0.90)]
        self.assertEqual(dwell_outliers({("pick_up_cups", "grasp"): entries}), {})


class MotionSignatureTests(unittest.TestCase):
    def test_grasp_verb_requires_one_gripper_transition(self) -> None:
        signature = resolve_motion_signature(["pick up the red block"])
        assert signature is not None
        self.assertIn("grasp", signature.names)
        self.assertEqual(signature.min_transitions, 1)
        self.assertFalse(signature.conflicted)

    def test_inflected_verb_still_matches(self) -> None:
        signature = resolve_motion_signature(["picks up the red block"])
        assert signature is not None
        self.assertIn("grasp", signature.names)

    def test_grasp_and_release_together_require_a_round_trip(self) -> None:
        signature = resolve_motion_signature(["pick up the block and place it in the box"])
        assert signature is not None
        self.assertGreaterEqual(signature.min_transitions, 2)
        self.assertEqual(signature.min_round_trips, 1)

    def test_chinese_instruction_matches_by_substring(self) -> None:
        signature = resolve_motion_signature(["把红色积木抓起来放进盒子"])
        assert signature is not None
        self.assertEqual(signature.min_round_trips, 1)

    def test_static_verb_forbids_gripper_actuation(self) -> None:
        signature = resolve_motion_signature(["push the drawer closed"])
        assert signature is not None
        self.assertEqual(signature.max_transitions, 0)

    def test_conflicting_verbs_are_reported_instead_of_guessed(self) -> None:
        signature = resolve_motion_signature(["pick up the sponge and wipe the table"])
        assert signature is not None
        self.assertTrue(signature.conflicted)

    def test_unknown_verb_yields_no_signature(self) -> None:
        self.assertIsNone(resolve_motion_signature(["observe the scene"]))

    def test_empty_task_text_yields_no_signature(self) -> None:
        self.assertIsNone(resolve_motion_signature([]))


class GripperTraceTests(unittest.TestCase):
    @staticmethod
    def square_wave(cycles: int, hold: int = 10) -> np.ndarray:
        values: list[float] = [0.0] * hold
        for _ in range(cycles):
            values.extend([1.0] * hold)
            values.extend([0.0] * hold)
        return np.asarray(values, dtype=np.float64)

    def test_transitions_and_round_trips_are_counted(self) -> None:
        trace = gripper_trace(self.square_wave(2), index=6)
        assert trace is not None
        self.assertEqual(trace.transitions, 4)
        self.assertEqual(trace.round_trips, 2)
        self.assertGreaterEqual(trace.extreme_ratio, 0.99)

    def test_constant_channel_reports_no_actuation(self) -> None:
        trace = gripper_trace(np.zeros(20), index=0)
        assert trace is not None
        self.assertEqual(trace.span, 0.0)
        self.assertEqual(trace.transitions, 0)

    def test_bimodal_channel_is_preferred_over_a_ramp(self) -> None:
        length = 60
        ramp = np.linspace(0.0, 1.0, length)
        gripper = self.square_wave(1, hold=20)
        action = np.stack([ramp, gripper], axis=1)
        traces = {0: channel_traces(action)}
        self.assertEqual(select_gripper_index(traces), 1)

    def test_no_channel_is_claimed_when_nothing_is_bimodal(self) -> None:
        action = np.stack([np.linspace(0.0, 1.0, 60), np.linspace(1.0, 0.0, 60)], axis=1)
        self.assertIsNone(select_gripper_index({0: channel_traces(action)}))

    def test_a_barely_moving_gripper_is_treated_as_inactive(self) -> None:
        noisy = np.asarray([0.0, 0.001, 0.0, 0.001] * 10, dtype=np.float64)
        traces = channel_traces(noisy.reshape(-1, 1))
        self.assertGreater(traces[0].transitions, 0)
        resolved = resolve_actuation(traces, index=0, median_span=1.0)
        assert resolved is not None
        self.assertEqual(resolved.transitions, 0)

    def test_an_active_gripper_survives_the_actuation_guard(self) -> None:
        traces = channel_traces(self.square_wave(1).reshape(-1, 1))
        resolved = resolve_actuation(traces, index=0, median_span=1.0)
        assert resolved is not None
        self.assertEqual(resolved.transitions, 2)


class MotionFindingTests(unittest.TestCase):
    def test_pick_task_without_gripper_actuation_is_reviewed(self) -> None:
        signature = resolve_motion_signature(["pick up the red block"])
        assert signature is not None
        trace = gripper_trace(np.zeros(30), index=0)
        assert trace is not None
        reasons, metrics = gripper_pattern_findings(signature, trace)
        self.assertEqual(reasons, ("semantic_motion_gripper_pattern_review",))
        self.assertIn("transitions=0<min=1", metrics["mismatches"])

    def test_push_task_with_a_full_grasp_cycle_is_reviewed(self) -> None:
        signature = resolve_motion_signature(["push the block to the target"])
        assert signature is not None
        trace = GripperTraceTests.square_wave(1)
        resolved = gripper_trace(trace, index=0)
        assert resolved is not None
        reasons, _ = gripper_pattern_findings(signature, resolved)
        self.assertEqual(reasons, ("semantic_motion_gripper_pattern_review",))

    def test_matching_gripper_pattern_produces_no_finding(self) -> None:
        signature = resolve_motion_signature(["pick up the block and place it in the box"])
        assert signature is not None
        resolved = gripper_trace(GripperTraceTests.square_wave(1), index=0)
        assert resolved is not None
        self.assertEqual(gripper_pattern_findings(signature, resolved)[0], ())

    def test_lift_moving_the_vertical_axis_downwards_is_reviewed(self) -> None:
        signature = resolve_motion_signature(["lift the cup off the table"])
        assert signature is not None
        excursions = vertical_excursions(np.asarray([0.5, 0.4, 0.1, 0.2]))
        reasons, metrics = vertical_direction_findings(signature, excursions, vertical_index=2)
        self.assertEqual(reasons, ("semantic_motion_vertical_direction_review",))
        self.assertGreater(metrics["fall"], metrics["rise"])

    def test_lift_moving_upwards_produces_no_finding(self) -> None:
        signature = resolve_motion_signature(["lift the cup off the table"])
        assert signature is not None
        excursions = vertical_excursions(np.asarray([0.1, 0.4, 0.9, 0.8]))
        reasons, _ = vertical_direction_findings(signature, excursions, vertical_index=2)
        self.assertEqual(reasons, ())

    def test_vertical_check_is_skipped_without_a_configured_axis(self) -> None:
        signature = resolve_motion_signature(["lift the cup off the table"])
        assert signature is not None
        reasons, metrics = vertical_direction_findings(signature, None, vertical_index=-1)
        self.assertEqual(reasons, ())
        self.assertEqual(metrics, {})


class BoundaryAlignmentTests(unittest.TestCase):
    @staticmethod
    def motion_with_grasp_at(frame: int, length: int = 40) -> EpisodeMotion:
        gripper = np.zeros(length)
        gripper[frame:] = 1.0
        arm = np.linspace(0.0, 1.0, length)
        return EpisodeMotion(action=np.stack([arm, gripper], axis=1), gripper_index=1)

    def test_gripper_transition_is_a_change_point(self) -> None:
        points = kinematic_change_points(self.motion_with_grasp_at(20))
        self.assertIn(20, points)

    def test_constant_speed_arm_contributes_no_change_points(self) -> None:
        arm = np.linspace(0.0, 1.0, 40).reshape(-1, 1)
        self.assertEqual(kinematic_change_points(EpisodeMotion(action=arm)), ())

    def test_an_arm_pause_marks_its_edges(self) -> None:
        arm = np.concatenate(
            [np.linspace(0.0, 1.0, 15), np.full(10, 1.0), np.linspace(1.0, 2.0, 15)]
        ).reshape(-1, 1)
        points = kinematic_change_points(EpisodeMotion(action=arm))
        self.assertTrue(points)
        self.assertTrue(all(14 <= frame <= 26 for frame in points), points)

    def test_boundaries_on_change_points_are_aligned(self) -> None:
        alignment = subtask_boundary_alignment([20], [20], tolerance_frames=5)
        assert alignment is not None
        self.assertEqual(alignment.aligned, 1)
        self.assertEqual(alignment.max_distance, 0)
        self.assertEqual(boundary_alignment_findings(alignment), ())

    def test_boundaries_far_from_any_change_point_are_reviewed(self) -> None:
        alignment = subtask_boundary_alignment([5, 33], [20], tolerance_frames=5)
        assert alignment is not None
        self.assertEqual(alignment.aligned, 0)
        self.assertEqual(
            boundary_alignment_findings(alignment),
            ("language_subtask_boundary_unaligned_review",),
        )

    def test_alignment_needs_both_boundaries_and_change_points(self) -> None:
        self.assertIsNone(subtask_boundary_alignment([], [20]))
        self.assertIsNone(subtask_boundary_alignment([20], []))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


WRIST_KEY = "observation.images.wrist"


def motion_dataset_info(
    total_episodes: int, total_frames: int, video_keys: tuple[str, ...] = (RGB_KEY,)
) -> dict:
    features = {
        "action": {"dtype": "float32", "shape": [2]},
        "observation.state": {"dtype": "float32", "shape": [2]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    for video_key in video_keys:
        features[video_key] = {
            "dtype": "video",
            "shape": [2, 2, 3],
            "info": {"video.fps": 30, "video.codec": "h264"},
        }
    return {
        "codebase_version": "v3.0",
        "fps": 30,
        "chunks_size": 1000,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "splits": {"train": f"0:{total_episodes}"},
        "features": features,
    }


def write_motion_dataset(
    root: Path,
    name: str,
    task: str,
    gripper_cycles: list[int],
    length: int = 60,
    task_column: str = "task",
    video_keys: tuple[str, ...] = (RGB_KEY,),
    pack_files: bool = False,
) -> Path:
    """Write a dataset whose second action dimension behaves like a gripper.

    `pack_files=True` reproduces the aggregated LeRobot v3 layout where every
    episode shares one data parquet file instead of owning its own.
    """
    dataset_root = root / name
    write_json(
        dataset_root / "meta" / "info.json",
        motion_dataset_info(len(gripper_cycles), len(gripper_cycles) * length, video_keys),
    )
    write_json(dataset_root / "meta" / "stats.json", {})
    pq.write_table(
        pa.table({"task_index": [0], task_column: [task]}),
        dataset_root / "meta" / "tasks.parquet",
    )

    rows = []
    episode_tables = []
    global_from = 0
    for episode_index, cycles in enumerate(gripper_cycles):
        arm = np.linspace(0.0, 1.0, length)
        gripper = np.zeros(length)
        if cycles:
            segment = length // (2 * cycles + 1)
            for cycle in range(cycles):
                start = (2 * cycle + 1) * segment
                gripper[start : start + segment] = 1.0
        table = pa.table(
            {
                "action": [[float(arm[i]), float(gripper[i])] for i in range(length)],
                "observation.state": [[float(arm[i]), float(gripper[i])] for i in range(length)],
                "timestamp": [i / 30.0 for i in range(length)],
                "frame_index": list(range(length)),
                "episode_index": [episode_index] * length,
                "index": list(range(global_from, global_from + length)),
                "task_index": [0] * length,
            }
        )
        if pack_files:
            episode_tables.append(table)
        else:
            data_path = dataset_root / "data" / "chunk-000" / f"file-{episode_index:03d}.parquet"
            data_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, data_path)

        row = {
            "episode_index": episode_index,
            "length": length,
            "data/chunk_index": 0,
            "data/file_index": 0 if pack_files else episode_index,
            "dataset_from_index": global_from,
            "dataset_to_index": global_from + length,
        }
        for video_key in video_keys:
            video_path = (
                dataset_root / "videos" / video_key / "chunk-000" / f"file-{episode_index:03d}.mp4"
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(b"synthetic video placeholder")
            row.update(
                {
                    f"videos/{video_key}/chunk_index": 0,
                    f"videos/{video_key}/file_index": episode_index,
                    f"videos/{video_key}/from_timestamp": 0.0,
                    f"videos/{video_key}/to_timestamp": length / 30.0,
                }
            )
        rows.append(row)
        global_from += length

    if pack_files:
        data_path = dataset_root / "data" / "chunk-000" / "file-000.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.concat_tables(episode_tables), data_path)

    episodes = pa.table({key: [row[key] for row in rows] for key in rows[0]})
    episode_path = dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(episodes, episode_path)
    return dataset_root


def run_motion_audit(dataset: Path, report_dir: Path, **kwargs) -> tuple[dict, dict[int, dict[str, str]]]:
    audit_dataset(
        root=dataset,
        out_dir=report_dir,
        depth_check="none",
        video_check="metadata",
        video_content_check="none",
        sample_frames=0,
        depth_sample_frames=0,
        duplicate_check="none",
        video_sample_stride=1,
        **kwargs,
    )
    summary = json.loads((report_dir / "quality_summary.json").read_text(encoding="utf-8"))
    with (report_dir / "quality_report.csv").open(newline="", encoding="utf-8") as handle:
        rows = {int(row["episode_index"]): row for row in csv.DictReader(handle)}
    return summary, rows


class MotionAuditIntegrationTests(unittest.TestCase):
    def test_pick_task_without_gripper_actuation_is_reviewed_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_motion_dataset(
                root, "no_grasp", "pick up the cup", gripper_cycles=[1, 1, 0]
            )
            summary, rows = run_motion_audit(dataset, root / "report")

            consistency = summary["semantic_consistency"]
            self.assertEqual(consistency["motion"]["gripper_index"], 1)
            self.assertEqual(consistency["motion"]["gripper_index_source"], "detected")
            self.assertEqual(consistency["motion"]["episodes_checked"], 3)
            self.assertEqual(consistency["motion_gripper_pattern_review"], 1)

            self.assertNotIn("semantic_motion_gripper_pattern_review", rows[0]["review_reasons"])
            self.assertIn("semantic_motion_gripper_pattern_review", rows[2]["review_reasons"])
            self.assertEqual(rows[2]["status"], "review")
            self.assertEqual(rows[2]["structural_status"], "keep")
            self.assertEqual(rows[2]["semantic_motion_verbs"], "grasp")
            self.assertEqual(rows[2]["semantic_motion_gripper_transitions"], "0")

    def test_configured_gripper_index_overrides_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_motion_dataset(
                root, "configured", "pick up the cup", gripper_cycles=[1, 1]
            )
            detected_summary, detected_rows = run_motion_audit(dataset, root / "detected")
            self.assertEqual(
                detected_summary["semantic_consistency"]["motion"]["gripper_index"], 1
            )
            self.assertEqual(detected_rows[0]["semantic_motion_gripper_transitions"], "2")

            summary, rows = run_motion_audit(
                dataset, root / "report", motion_gripper_index=0
            )
            motion = summary["semantic_consistency"]["motion"]
            self.assertEqual(motion["gripper_index"], 0)
            self.assertEqual(motion["gripper_index_source"], "configured")
            self.assertEqual(rows[0]["semantic_motion_gripper_transitions"], "1")

    def test_unknown_verb_is_recorded_as_not_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_motion_dataset(
                root, "unknown_verb", "observe the scene", gripper_cycles=[1, 1]
            )
            summary, rows = run_motion_audit(dataset, root / "report")
            self.assertEqual(summary["semantic_consistency"]["motion"]["verb_unmatched"], 2)
            self.assertEqual(summary["semantic_consistency"]["motion_gripper_pattern_review"], 0)
            self.assertIn("semantic_motion_verb_unmatched", rows[0]["not_checked"])

    def test_motion_check_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_motion_dataset(
                root, "disabled", "pick up the cup", gripper_cycles=[0, 0]
            )
            summary, rows = run_motion_audit(dataset, root / "report", motion_check="none")
            self.assertFalse(summary["semantic_consistency"]["motion"]["enabled"])
            self.assertEqual(summary["semantic_consistency"]["motion_gripper_pattern_review"], 0)
            self.assertIn("semantic_motion_check_disabled", rows[0]["not_checked"])
            self.assertEqual(rows[0]["status"], "keep")


class RetryCurveEvaluator:
    """Returns a high-scoring episode whose progress collapses mid-episode."""

    name = "retry_curve_test_evaluator"

    def evaluate(self, episode):
        if episode.motion is None or episode.motion.action is None:
            raise AssertionError("semantic evaluator did not receive the motion streams")
        return {
            "progress_curve": [0.05, 0.80, 0.10, 0.70, 0.97, 0.99],
            "success_probability": 0.95,
        }


class ProgressShapeAuditIntegrationTests(unittest.TestCase):
    def test_a_retry_curve_is_reviewed_even_though_the_score_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_motion_dataset(
                root, "retry", "pick up the cup", gripper_cycles=[1]
            )
            profile = three_stage_profile(
                task_name="pick up the cup", aliases=["pick up the cup"]
            )
            profile_dir = root / "profiles"
            profile.save(profile_dir / "pick_up_cups.json")

            from utils.task_semantic_profile import TaskProfileRegistry

            summary, rows = run_motion_audit(
                dataset,
                root / "report",
                semantic_evaluator=RetryCurveEvaluator(),
                task_profiles=TaskProfileRegistry.from_directory(profile_dir),
            )

            self.assertEqual(rows[0]["semantic_status"], "pass")
            self.assertEqual(rows[0]["status"], "review")
            self.assertEqual(rows[0]["structural_status"], "keep")
            self.assertIn("semantic_progress_regression_review", rows[0]["review_reasons"])
            self.assertEqual(rows[0]["semantic_stage_timeline_source"], "derived_from_progress")
            self.assertAlmostEqual(float(rows[0]["semantic_progress_max_drop"]), 0.70)

            consistency = summary["semantic_consistency"]
            self.assertEqual(consistency["progress_regression_review"], 1)
            self.assertEqual(consistency["stage_order_violation_review"], 1)

            metrics = json.loads(rows[0]["metrics_json"])
            self.assertEqual(metrics["semantic_progress_metrics"]["regression_count"], 1)
            self.assertEqual(metrics["semantic_stage_timeline"]["source"], "derived_from_progress")


class SemanticReuseMotionTests(unittest.TestCase):
    def test_semantic_only_rerun_still_performs_the_motion_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_motion_dataset(
                root, "reuse", "pick up the cup", gripper_cycles=[1, 0]
            )
            report_dir = root / "report"
            run_motion_audit(dataset, report_dir, motion_check="none")

            audit_dataset_semantic_reuse(
                root=dataset,
                out_dir=report_dir,
                semantic_evaluator=RetryCurveEvaluator(),
            )
            summary = json.loads((report_dir / "quality_summary.json").read_text(encoding="utf-8"))
            with (report_dir / "quality_report.csv").open(newline="", encoding="utf-8") as handle:
                rows = {int(row["episode_index"]): row for row in csv.DictReader(handle)}

            self.assertEqual(summary["audit_mode"], "semantic_reuse")
            self.assertEqual(summary["semantic_consistency"]["motion"]["gripper_index"], 1)
            self.assertEqual(summary["semantic_consistency"]["motion_gripper_pattern_review"], 1)
            self.assertIn("semantic_motion_gripper_pattern_review", rows[1]["review_reasons"])
            self.assertNotIn("semantic_motion_gripper_pattern_review", rows[0]["review_reasons"])
            self.assertNotIn("semantic_motion_check_disabled", rows[0]["not_checked"])


class SplitAtAnnotator:
    """Emits two subtasks with the handover placed at a caller-chosen frame."""

    name = "split_at_test_annotator"

    def __init__(self, split: int) -> None:
        self.split = split

    def annotate(self, episode):
        return {
            "subtasks": [
                {
                    "start_frame": 0,
                    "end_frame": self.split - 1,
                    "text": "approach the cup",
                },
                {
                    "start_frame": self.split,
                    "end_frame": episode.length - 1,
                    "text": "grasp and lift the cup",
                },
            ],
            "summary": "The robot approaches and lifts the cup.",
        }


def write_minimal_quality_report(report_dir: Path, dataset: Path, lengths: list[int]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    fields = ["episode_index", "status", "semantic_status", "semantic_score", "length"]
    with (report_dir / "quality_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for episode_index, length in enumerate(lengths):
            writer.writerow(
                {
                    "episode_index": episode_index,
                    "status": "keep",
                    "semantic_status": "pass",
                    "semantic_score": "0.96",
                    "length": length,
                }
            )
    write_json(report_dir / "quality_summary.json", {"dataset": str(dataset.resolve())})


class BoundaryAlignmentEnrichmentTests(unittest.TestCase):
    """The gripper closes at frame 20 and opens at frame 40 in this fixture."""

    def run_enrichment(self, root: Path, split: int, name: str) -> dict:
        from pipeline.trajectory_language_enrichment import (
            LanguageEnrichmentConfig,
            run_language_enrichment,
        )

        dataset = write_motion_dataset(
            root, f"boundary_{name}", "pick up the cup", gripper_cycles=[1]
        )
        report_dir = root / f"quality_{name}"
        write_minimal_quality_report(report_dir, dataset, [60])
        out_dir = root / f"language_{name}"
        run_language_enrichment(
            LanguageEnrichmentConfig(
                dataset_root=dataset,
                quality_report=report_dir / "quality_report.csv",
                out_dir=out_dir,
                annotator=SplitAtAnnotator(split),
            )
        )
        rows = [
            json.loads(line)
            for line in (out_dir / "subtask_boundary_alignment.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        self.assertEqual(len(rows), 1)
        return rows[0]

    def test_boundary_on_the_grasp_frame_is_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            row = self.run_enrichment(Path(tmp), split=20, name="aligned")
            self.assertEqual(row["status"], "aligned")
            self.assertEqual(row["gripper_index"], 1)
            self.assertEqual(row["max_distance"], 0)
            self.assertEqual(row["reasons"], [])

    def test_boundary_far_from_any_gripper_event_is_reviewed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            row = self.run_enrichment(Path(tmp), split=8, name="unaligned")
            self.assertEqual(row["status"], "review")
            self.assertEqual(row["aligned"], 0)
            self.assertEqual(row["reasons"], ["language_subtask_boundary_unaligned_review"])


class NormalizeResultContractTests(unittest.TestCase):
    def test_samples_derive_the_missing_success_probability(self) -> None:
        result = normalize_result({"score_samples": [0.90, 0.92, 0.94]})
        self.assertAlmostEqual(result.success_probability, 0.92)
        self.assertAlmostEqual(result.score, 0.92)
        assert result.uncertainty is not None
        self.assertLess(result.uncertainty, 0.05)

    def test_explicit_success_probability_is_not_overwritten_by_samples(self) -> None:
        result = normalize_result(
            {"success_probability": 0.80, "score_samples": [0.10, 0.20]}
        )
        self.assertAlmostEqual(result.score, 0.80)

    def test_sample_out_of_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_result({"success_probability": 0.9, "score_samples": [0.5, 1.5]})

    def test_negative_score_std_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_result({"success_probability": 0.9, "score_std": -0.1})

    def test_margin_out_of_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_result(
                {"success_probability": 0.9, "details": {"counterfactual_margin": 1.5}}
            )

    def test_negative_margin_within_range_is_accepted(self) -> None:
        result = normalize_result(
            {"success_probability": 0.9, "details": {"counterfactual_margin": -0.2}}
        )
        self.assertAlmostEqual(result.counterfactual_margin, -0.2)

    def test_explicit_std_wins_over_samples(self) -> None:
        result = SemanticResult(
            success_probability=0.9, score_samples=(0.1, 0.9), score_std=0.01
        )
        self.assertAlmostEqual(result.uncertainty, 0.01)

    def test_single_sample_has_no_uncertainty(self) -> None:
        result = normalize_result({"score_samples": [0.9]})
        self.assertIsNone(result.uncertainty)


class TaskTextColumnTests(unittest.TestCase):
    """tasks.parquet written from a pandas frame stores text in __index_level_0__."""

    def test_pandas_index_column_feeds_the_motion_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_motion_dataset(
                root,
                "pandas_tasks",
                "Grab all the cups on the table then put them into the box",
                gripper_cycles=[1, 0],
                task_column="__index_level_0__",
            )
            summary, rows = run_motion_audit(dataset, root / "report")

            motion = summary["semantic_consistency"]["motion"]
            self.assertEqual(motion["verb_unmatched"], 0)
            self.assertEqual(motion["episodes_checked"], 2)
            self.assertEqual(rows[0]["semantic_motion_verbs"], "grasp;release")
            self.assertNotIn(
                "semantic_motion_gripper_pattern_review", rows[0]["review_reasons"]
            )
            self.assertIn(
                "semantic_motion_gripper_pattern_review", rows[1]["review_reasons"]
            )


class AggregatedLayoutTests(unittest.TestCase):
    """Recorded v3 datasets pack many episodes into one shared data file."""

    def test_shared_data_file_is_sliced_per_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_motion_dataset(
                root,
                "packed",
                "Grab all the cups on the table then put them into the box",
                gripper_cycles=[1, 0, 1],
                pack_files=True,
            )
            summary, rows = run_motion_audit(dataset, root / "report")

            self.assertEqual(summary["drop"], 0)
            for episode_index in (0, 1, 2):
                self.assertEqual(rows[episode_index]["drop_reasons"], "")
                self.assertEqual(int(rows[episode_index]["data_rows"]), 60)

            # Per-episode slicing: only the cycle-free middle episode is flagged.
            self.assertNotIn(
                "semantic_motion_gripper_pattern_review", rows[0]["review_reasons"]
            )
            self.assertIn(
                "semantic_motion_gripper_pattern_review", rows[1]["review_reasons"]
            )
            self.assertNotIn(
                "semantic_motion_gripper_pattern_review", rows[2]["review_reasons"]
            )
            self.assertEqual(rows[1]["semantic_motion_gripper_transitions"], "0")
            self.assertEqual(rows[0]["semantic_motion_gripper_transitions"], "2")

    def test_semantic_reuse_slices_the_shared_file_too(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_motion_dataset(
                root,
                "packed_reuse",
                "Grab all the cups on the table then put them into the box",
                gripper_cycles=[1, 0],
                pack_files=True,
            )
            report_dir = root / "report"
            run_motion_audit(dataset, report_dir, motion_check="none")
            audit_dataset_semantic_reuse(
                root=dataset,
                out_dir=report_dir,
                semantic_evaluator=RetryCurveEvaluator(),
            )
            with (report_dir / "quality_report.csv").open(newline="", encoding="utf-8") as handle:
                rows = {int(row["episode_index"]): row for row in csv.DictReader(handle)}
            self.assertIn(
                "semantic_motion_gripper_pattern_review", rows[1]["review_reasons"]
            )
            self.assertNotIn(
                "semantic_motion_gripper_pattern_review", rows[0]["review_reasons"]
            )


class ConfigurableEvaluator:
    """Returns a fixed verdict, optionally per-view, with contract extras."""

    name = "configurable_test_evaluator"

    def __init__(
        self,
        success: float = 0.95,
        samples: tuple[float, ...] = (),
        details: dict | None = None,
        view_scores: dict[str, float] | None = None,
    ) -> None:
        self.success = success
        self.samples = samples
        self.details = details
        self.view_scores = view_scores

    def evaluate(self, episode):
        if self.view_scores is not None and len(episode.videos) == 1:
            return {"success_probability": self.view_scores[episode.videos[0].key]}
        result: dict = {"success_probability": self.success}
        if self.samples:
            result["score_samples"] = list(self.samples)
        if self.details is not None:
            result["details"] = dict(self.details)
        return result


class ScoreReliabilityAuditTests(unittest.TestCase):
    def run_reliability_audit(self, root: Path, evaluator, **kwargs):
        dataset = write_motion_dataset(
            root, "reliability", "pick up the cup and place it in the box", gripper_cycles=[1]
        )
        return run_motion_audit(
            dataset, root / "report", semantic_evaluator=evaluator, **kwargs
        )

    def test_unstable_samples_demote_a_passing_score_to_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary, rows = self.run_reliability_audit(
                Path(tmp), ConfigurableEvaluator(samples=(0.95, 0.50, 0.99, 0.45))
            )
            self.assertEqual(rows[0]["semantic_status"], "review")
            self.assertIn("semantic_score_unstable_review", rows[0]["review_reasons"])
            self.assertGreaterEqual(float(rows[0]["semantic_score_uncertainty"]), 0.15)
            self.assertEqual(
                summary["semantic_consistency"]["score_unstable_review"], 1
            )

    def test_stable_samples_keep_the_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary, rows = self.run_reliability_audit(
                Path(tmp), ConfigurableEvaluator(samples=(0.94, 0.95, 0.96))
            )
            self.assertEqual(rows[0]["semantic_status"], "pass")
            self.assertNotIn("semantic_score_unstable_review", rows[0]["review_reasons"])

    def test_low_counterfactual_margin_is_reviewed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary, rows = self.run_reliability_audit(
                Path(tmp),
                ConfigurableEvaluator(details={"counterfactual_margin": 0.03}),
            )
            self.assertEqual(rows[0]["semantic_status"], "review")
            self.assertIn(
                "semantic_counterfactual_margin_low_review", rows[0]["review_reasons"]
            )
            self.assertAlmostEqual(
                float(rows[0]["semantic_counterfactual_margin"]), 0.03
            )
            self.assertEqual(
                summary["semantic_consistency"]["counterfactual_margin_low_review"], 1
            )

    def test_wide_margin_keeps_the_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, rows = self.run_reliability_audit(
                Path(tmp),
                ConfigurableEvaluator(details={"counterfactual_margin": 0.50}),
            )
            self.assertEqual(rows[0]["semantic_status"], "pass")
            self.assertNotIn(
                "semantic_counterfactual_margin_low_review", rows[0]["review_reasons"]
            )

    def test_backends_without_the_contracts_record_not_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, rows = self.run_reliability_audit(Path(tmp), ConfigurableEvaluator())
            self.assertIn("semantic_score_stability_not_checked", rows[0]["not_checked"])
            self.assertIn("semantic_counterfactual_not_checked", rows[0]["not_checked"])
            self.assertEqual(rows[0]["semantic_status"], "pass")


class PerViewAuditTests(unittest.TestCase):
    def two_view_dataset(self, root: Path) -> Path:
        return write_motion_dataset(
            root,
            "two_views",
            "pick up the cup and place it in the box",
            gripper_cycles=[1],
            video_keys=(RGB_KEY, WRIST_KEY),
        )

    def test_disagreeing_views_are_reviewed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluator = ConfigurableEvaluator(
                view_scores={RGB_KEY: 0.95, WRIST_KEY: 0.40}
            )
            summary, rows = run_motion_audit(
                self.two_view_dataset(root),
                root / "report",
                semantic_evaluator=evaluator,
                semantic_per_view=True,
            )
            self.assertEqual(rows[0]["semantic_status"], "review")
            self.assertIn("semantic_view_disagreement_review", rows[0]["review_reasons"])
            self.assertAlmostEqual(float(rows[0]["semantic_view_disagreement"]), 0.55)
            metrics = json.loads(rows[0]["metrics_json"])
            self.assertAlmostEqual(metrics["semantic_view_scores"][WRIST_KEY], 0.40)
            self.assertEqual(
                summary["semantic_consistency"]["view_disagreement_review"], 1
            )
            self.assertTrue(summary["semantic_evaluation"]["per_view"])

    def test_agreeing_views_keep_the_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluator = ConfigurableEvaluator(
                view_scores={RGB_KEY: 0.95, WRIST_KEY: 0.90}
            )
            _, rows = run_motion_audit(
                self.two_view_dataset(root),
                root / "report",
                semantic_evaluator=evaluator,
                semantic_per_view=True,
            )
            self.assertEqual(rows[0]["semantic_status"], "pass")
            self.assertNotIn(
                "semantic_view_disagreement_review", rows[0]["review_reasons"]
            )
            self.assertAlmostEqual(float(rows[0]["semantic_view_disagreement"]), 0.05)

    def test_per_view_stays_off_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluator = ConfigurableEvaluator(
                view_scores={RGB_KEY: 0.95, WRIST_KEY: 0.40}
            )
            _, rows = run_motion_audit(
                self.two_view_dataset(root),
                root / "report",
                semantic_evaluator=evaluator,
            )
            self.assertEqual(rows[0]["semantic_view_disagreement"], "")
            self.assertNotIn(
                "semantic_view_disagreement_review", rows[0]["review_reasons"]
            )

    def test_single_view_records_a_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_motion_dataset(
                root, "one_view", "pick up the cup and place it in the box", gripper_cycles=[1]
            )
            _, rows = run_motion_audit(
                dataset,
                root / "report",
                semantic_evaluator=ConfigurableEvaluator(),
                semantic_per_view=True,
            )
            self.assertIn(
                "semantic_view_disagreement_single_view", rows[0]["not_checked"]
            )


if __name__ == "__main__":
    unittest.main()
