from __future__ import annotations

import binascii
import csv
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.dataset_clean_build import build_clean_dataset
from pipeline.data_quality_audit import audit_dataset, audit_dataset_semantic_reuse
from pipeline.dataset_merge import merge_datasets
from utils.semantic_evaluator import load_evaluator
from utils.task_semantic_profile import TaskProfileRegistry, TaskSemanticProfile


DEPTH_KEY = "observation.images.front_depth"
RGB_KEY = "observation.images.front"


class DeterministicSemanticEvaluator:
    name = "deterministic_test_evaluator"

    def __init__(self) -> None:
        self.seen: list[int] = []

    def evaluate(self, episode):
        self.seen.append(episode.episode_index)
        if not episode.tasks or not episode.videos:
            raise AssertionError("semantic evaluator did not receive task and video context")
        if episode.episode_index == 0:
            return {
                "progress_curve": [0.05, 0.18, 0.27],
                "success_probability": 0.20,
                "details": {"source": "synthetic"},
            }
        if episode.episode_index == 1:
            return {
                "progress_curve": [0.10, 0.55, 0.92],
                "success_probability": 0.88,
            }
        return {
            "progress_curve": [0.15, 0.70, 0.97],
            "success_probability": 0.96,
        }


class ProfileAwareSemanticEvaluator:
    name = "profile_aware_test_evaluator"

    def evaluate(self, episode):
        if episode.task_profile is None:
            raise AssertionError("semantic evaluator did not receive the task profile")
        return {
            "progress_curve": [0.10, 0.45],
            "success_probability": 0.40,
        }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(kind)
    crc = binascii.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def write_depth_png(path: Path, values: np.ndarray) -> None:
    if values.dtype != np.uint16:
        raise ValueError("depth test PNG expects uint16 input")
    height, width = values.shape
    raw = b"".join(b"\x00" + values[row].astype(">u2").tobytes() for row in range(height))
    payload = [
        b"\x89PNG\r\n\x1a\n",
        png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)),
        png_chunk(b"IDAT", zlib.compress(raw)),
        png_chunk(b"IEND", b""),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(payload))


def dataset_info(total_episodes: int, total_frames: int) -> dict:
    return {
        "codebase_version": "v3.0",
        "fps": 30,
        "chunks_size": 1000,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "splits": {"train": f"0:{total_episodes}"},
        "features": {
            "action": {"dtype": "float32", "shape": [2]},
            "observation.state": {"dtype": "float32", "shape": [2]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            RGB_KEY: {
                "dtype": "video",
                "shape": [2, 2, 3],
                "info": {"video.fps": 30, "video.codec": "h264"},
            },
            DEPTH_KEY: {"dtype": "image", "shape": [2, 2, 1]},
        },
    }


def rgb_dataset_info(total_episodes: int, total_frames: int) -> dict:
    info = dataset_info(total_episodes, total_frames)
    info["features"].pop(DEPTH_KEY)
    info["features"][RGB_KEY] = {
        "dtype": "video",
        "shape": [2, 2, 3],
        "info": {"video.fps": 30, "video.codec": "h264"},
    }
    return info


def write_episode_data(dataset_root: Path, episode_index: int, global_from: int, length: int, base_value: int) -> None:
    paths = []
    for frame_index in range(length):
        rel = f"frame-{frame_index:06d}.png"
        depth = np.array(
            [
                [base_value + frame_index + 1, base_value + frame_index + 2],
                [base_value + frame_index + 3, base_value + frame_index + 4],
            ],
            dtype=np.uint16,
        )
        write_depth_png(dataset_root / "depth_sidecar" / DEPTH_KEY / f"episode-{episode_index:06d}" / rel, depth)
        paths.append({"path": rel, "bytes": None})

    table = pa.table(
        {
            "action": [[base_value + i, base_value + i + 0.5] for i in range(length)],
            "observation.state": [[base_value + i + 1.0, base_value + i + 1.5] for i in range(length)],
            "timestamp": [i / 30.0 for i in range(length)],
            "frame_index": list(range(length)),
            "episode_index": [episode_index] * length,
            "index": list(range(global_from, global_from + length)),
            "task_index": [0] * length,
            DEPTH_KEY: pa.array(paths, type=pa.struct([("path", pa.string()), ("bytes", pa.binary())])),
        }
    )
    out = dataset_root / "data" / "chunk-000" / f"file-{episode_index:03d}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    write_placeholder_video(dataset_root, RGB_KEY, episode_index)


def write_placeholder_video(dataset_root: Path, video_key: str, episode_index: int) -> None:
    video_path = dataset_root / "videos" / video_key / "chunk-000" / f"file-{episode_index:03d}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"synthetic video placeholder")


def write_dataset(root: Path, name: str, lengths: list[int], base_value: int) -> Path:
    dataset_root = root / name
    total_frames = sum(lengths)
    write_json(dataset_root / "meta" / "info.json", dataset_info(len(lengths), total_frames))
    write_json(dataset_root / "meta" / "stats.json", {})
    write_json(dataset_root / "meta" / "rgbd_vla_depth_recording.json", {"depth_streams": [DEPTH_KEY]})
    (dataset_root / "videos").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"task_index": [0], "task": ["pick up cups"]}), dataset_root / "meta" / "tasks.parquet")

    rows = []
    global_from = 0
    for episode_index, length in enumerate(lengths):
        write_episode_data(dataset_root, episode_index, global_from, length, base_value + episode_index * 100)
        rows.append(
            {
                "episode_index": episode_index,
                "length": length,
                "data/chunk_index": 0,
                "data/file_index": episode_index,
                "dataset_from_index": global_from,
                "dataset_to_index": global_from + length,
                f"videos/{RGB_KEY}/chunk_index": 0,
                f"videos/{RGB_KEY}/file_index": episode_index,
                f"videos/{RGB_KEY}/from_timestamp": 0.0,
                f"videos/{RGB_KEY}/to_timestamp": length / 30.0,
            }
        )
        global_from += length

    episodes = pa.table({key: [row[key] for row in rows] for key in rows[0]})
    episode_path = dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(episodes, episode_path)
    return dataset_root


def write_rgb_episode_data(dataset_root: Path, episode_index: int, global_from: int, length: int, base_value: int) -> None:
    table = pa.table(
        {
            "action": [[base_value + i, base_value + i + 0.5] for i in range(length)],
            "observation.state": [[base_value + i + 1.0, base_value + i + 1.5] for i in range(length)],
            "timestamp": [i / 30.0 for i in range(length)],
            "frame_index": list(range(length)),
            "episode_index": [episode_index] * length,
            "index": list(range(global_from, global_from + length)),
            "task_index": [0] * length,
        }
    )
    out = dataset_root / "data" / "chunk-000" / f"file-{episode_index:03d}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)


def write_rgb_dataset(root: Path, name: str, lengths: list[int], base_value: int) -> Path:
    dataset_root = root / name
    total_frames = sum(lengths)
    write_json(dataset_root / "meta" / "info.json", rgb_dataset_info(len(lengths), total_frames))
    write_json(dataset_root / "meta" / "stats.json", {})
    pq.write_table(pa.table({"task_index": [0], "task": ["pick up cups"]}), dataset_root / "meta" / "tasks.parquet")

    rows = []
    global_from = 0
    for episode_index, length in enumerate(lengths):
        write_rgb_episode_data(dataset_root, episode_index, global_from, length, base_value + episode_index * 100)
        rows.append(
            {
                "episode_index": episode_index,
                "length": length,
                "data/chunk_index": 0,
                "data/file_index": episode_index,
                "dataset_from_index": global_from,
                "dataset_to_index": global_from + length,
                f"videos/{RGB_KEY}/chunk_index": 0,
                f"videos/{RGB_KEY}/file_index": episode_index,
                f"videos/{RGB_KEY}/from_timestamp": 0.0,
                f"videos/{RGB_KEY}/to_timestamp": length / 30.0,
            }
        )
        write_placeholder_video(dataset_root, RGB_KEY, episode_index)
        global_from += length

    episodes = pa.table({key: [row[key] for row in rows] for key in rows[0]})
    episode_path = dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(episodes, episode_path)
    return dataset_root


class LeRobotPipelineTests(unittest.TestCase):
    def test_task_profile_controls_threshold_and_failure_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_rgb_dataset(root, "profile_dataset", [3], 10)
            report_dir = root / "profile_quality_report"
            profile = TaskSemanticProfile.from_dict(
                {
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
                    "review_threshold": 0.60,
                    "pass_threshold": 0.95,
                }
            )
            registry = TaskProfileRegistry((profile,))

            exit_code = audit_dataset(
                root=dataset,
                out_dir=report_dir,
                depth_check="header",
                video_check="metadata",
                video_content_check="none",
                sample_frames=0,
                depth_sample_frames=0,
                duplicate_check="none",
                video_sample_stride=30,
                semantic_evaluator=ProfileAwareSemanticEvaluator(),
                semantic_review_threshold=0.5,
                semantic_pass_threshold=0.9,
                task_profiles=registry,
            )

            self.assertEqual(exit_code, 0)
            with (report_dir / "quality_report.csv").open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["semantic_task_profile"], "pick_up_cups")
            self.assertEqual(float(row["semantic_review_threshold"]), 0.60)
            self.assertEqual(float(row["semantic_pass_threshold"]), 0.95)
            self.assertEqual(row["semantic_status"], "review")
            self.assertEqual(row["semantic_failure_stage"], "grasp")

    def test_semantic_evaluator_uses_failure_uncertainty_and_pass_bands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_rgb_dataset(root, "semantic_dataset", [3, 2, 2], 10)
            report_dir = root / "semantic_quality_report"
            evaluator = load_evaluator(f"{__name__}:DeterministicSemanticEvaluator")

            exit_code = audit_dataset(
                root=dataset,
                out_dir=report_dir,
                depth_check="header",
                video_check="metadata",
                video_content_check="none",
                sample_frames=0,
                depth_sample_frames=0,
                duplicate_check="none",
                video_sample_stride=30,
                semantic_evaluator=evaluator,
                semantic_review_threshold=0.5,
                semantic_pass_threshold=0.9,
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(evaluator.seen, [0, 1, 2])
            with (report_dir / "quality_report.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["status"], "review")
            self.assertEqual(rows[0]["structural_status"], "keep")
            self.assertEqual(rows[0]["semantic_status"], "review")
            self.assertEqual(rows[0]["semantic_evaluator"], evaluator.name)
            self.assertEqual(float(rows[0]["semantic_progress_final"]), 0.27)
            self.assertEqual(float(rows[0]["semantic_success_probability"]), 0.20)
            self.assertEqual(float(rows[0]["semantic_review_threshold"]), 0.5)
            self.assertEqual(float(rows[0]["semantic_pass_threshold"]), 0.9)
            self.assertEqual(rows[0]["semantic_failure_stage"], "early_progress")
            self.assertIn("semantic_low_score_review", rows[0]["review_reasons"])
            self.assertEqual(rows[1]["status"], "review")
            self.assertEqual(rows[1]["semantic_status"], "review")
            self.assertIn("semantic_uncertain_score_review", rows[1]["review_reasons"])
            self.assertEqual(rows[2]["status"], "keep")
            self.assertEqual(rows[2]["semantic_status"], "pass")
            self.assertEqual(
                (report_dir / "review_episodes.txt").read_text(encoding="utf-8").splitlines(),
                ["0", "1"],
            )
            summary = json.loads((report_dir / "quality_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["semantic_evaluation"]["enabled"])
            self.assertEqual(summary["semantic_evaluation"]["pass"], 1)
            self.assertEqual(summary["semantic_evaluation"]["review"], 2)
            self.assertEqual(summary["semantic_evaluation"]["low_score_review"], 1)
            self.assertEqual(summary["semantic_evaluation"]["uncertain_score_review"], 1)

    def test_semantic_reuse_merges_prior_structural_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_rgb_dataset(root, "reuse_dataset", [3, 2, 2], 10)
            report_dir = root / "reuse_quality_report"

            exit_code = audit_dataset(
                root=dataset,
                out_dir=report_dir,
                depth_check="header",
                video_check="metadata",
                video_content_check="none",
                sample_frames=0,
                depth_sample_frames=0,
                duplicate_check="none",
                video_sample_stride=30,
            )
            self.assertEqual(exit_code, 0)

            report_path = report_dir / "quality_report.csv"
            with report_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames
                rows = list(reader)
            rows[1]["review_reasons"] = (
                f"video_file_black_high_review:{RGB_KEY};semantic_low_score_review"
            )
            rows[1]["status"] = "review"
            with report_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            evaluator = load_evaluator(f"{__name__}:DeterministicSemanticEvaluator")
            exit_code = audit_dataset_semantic_reuse(
                root=dataset,
                out_dir=report_dir,
                semantic_evaluator=evaluator,
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(evaluator.seen, [0, 1, 2])
            with report_path.open(newline="", encoding="utf-8") as handle:
                merged = list(csv.DictReader(handle))
            self.assertEqual(merged[0]["status"], "review")
            self.assertEqual(merged[0]["structural_status"], "keep")
            self.assertEqual(merged[0]["semantic_status"], "review")
            self.assertIn("semantic_low_score_review", merged[0]["review_reasons"])
            self.assertIn(
                "structural_checks_reused_from_prior_report", merged[0]["not_checked"]
            )
            self.assertEqual(merged[1]["structural_status"], "review")
            self.assertIn("video_file_black_high_review", merged[1]["review_reasons"])
            self.assertIn("semantic_uncertain_score_review", merged[1]["review_reasons"])
            self.assertNotIn("semantic_low_score_review", merged[1]["review_reasons"])
            self.assertEqual(merged[2]["status"], "keep")
            self.assertEqual(merged[2]["semantic_status"], "pass")
            summary = json.loads(
                (report_dir / "quality_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["audit_mode"], "semantic_reuse")
            self.assertEqual(summary["reused_report"], str(report_path))
            self.assertTrue(summary["semantic_evaluation"]["enabled"])
            self.assertEqual(summary["semantic_evaluation"]["pass"], 1)
            self.assertEqual(summary["semantic_evaluation"]["review"], 2)

    def test_semantic_reuse_rejects_report_from_other_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_a = write_rgb_dataset(root, "reuse_a_dataset", [3, 2, 2], 10)
            dataset_b = write_rgb_dataset(root, "reuse_b_dataset", [3, 2, 2], 40)
            report_a = root / "reuse_a_report"

            exit_code = audit_dataset(
                root=dataset_a,
                out_dir=report_a,
                depth_check="header",
                video_check="metadata",
                video_content_check="none",
                sample_frames=0,
                depth_sample_frames=0,
                duplicate_check="none",
                video_sample_stride=30,
            )
            self.assertEqual(exit_code, 0)

            report_b = root / "reuse_b_report"
            evaluator = load_evaluator(f"{__name__}:DeterministicSemanticEvaluator")
            exit_code = audit_dataset_semantic_reuse(
                root=dataset_b,
                out_dir=report_b,
                semantic_evaluator=evaluator,
                reuse_report=report_a / "quality_report.csv",
            )

            self.assertEqual(exit_code, 1)
            self.assertEqual(evaluator.seen, [])
            summary = json.loads(
                (report_b / "quality_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status"], "fail")
            self.assertTrue(
                any(
                    error.startswith("prior_report_dataset_mismatch")
                    for error in summary["global_errors"]
                )
            )

    def test_merge_audit_and_clean_small_rgbd_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, "a_dataset", [3, 2], 10)
            write_dataset(root, "b_dataset", [2], 300)

            merged = root / "determ_lerobot_rgb_depth_vla"
            merge_datasets(
                src_root=root,
                out_dir=merged,
                overwrite=False,
                dataset_names=["a_dataset", "b_dataset"],
                copy_mode="copy",
            )
            info = json.loads((merged / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["total_episodes"], 3)
            self.assertEqual(info["total_frames"], 7)
            self.assertTrue((merged / "videos" / RGB_KEY / "chunk-000" / "file-002.mp4").exists())
            self.assertTrue((merged / "depth_sidecar" / DEPTH_KEY / "episode-000002" / "frame-000001.png").exists())

            report_dir = root / "quality_report"
            exit_code = audit_dataset(
                root=merged,
                out_dir=report_dir,
                depth_check="header",
                video_check="metadata",
                video_content_check="none",
                sample_frames=2,
                depth_sample_frames=2,
                duplicate_check="none",
                video_sample_stride=30,
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual((report_dir / "drop_episodes.txt").read_text(encoding="utf-8").strip(), "")

            drop_list = root / "drop_episodes.txt"
            drop_list.write_text("1\n", encoding="utf-8")
            cleaned = root / "cleaned_dataset"
            build_clean_dataset(
                src_dataset=merged,
                out_dir=cleaned,
                drop_list=drop_list,
                overwrite=False,
                video_mode="copy-referenced",
                copy_mode="copy",
                ffmpeg_path=None,
                video_crf=30,
                dry_run=False,
            )
            clean_info = json.loads((cleaned / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(clean_info["total_episodes"], 2)
            self.assertEqual(clean_info["total_frames"], 5)
            manifest = json.loads((cleaned / "cleaning_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["dropped_episodes"], [1])

    def test_audit_accepts_rgb_only_lerobot_dataset_without_depth_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_rgb_dataset(root, "rgb_only_dataset", [3], 10)
            self.assertFalse((dataset / "depth_sidecar").exists())

            report_dir = root / "rgb_quality_report"
            exit_code = audit_dataset(
                root=dataset,
                out_dir=report_dir,
                depth_check="header",
                video_check="metadata",
                video_content_check="none",
                sample_frames=0,
                depth_sample_frames=0,
                duplicate_check="none",
                video_sample_stride=30,
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual((report_dir / "drop_episodes.txt").read_text(encoding="utf-8").strip(), "")
            summary = json.loads((report_dir / "quality_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["depth_keys"], [])


if __name__ == "__main__":
    unittest.main()
