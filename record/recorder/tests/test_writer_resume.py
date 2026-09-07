import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from recorder.dataset.features import build_features
from recorder.dataset.writer import LeRobotDatasetWriter
from recorder.display import FramePreview, depth_to_gray

MOTORS = ["shoulder_pan", "gripper"]


def make_writer(root: Path, *, resume: bool = False) -> LeRobotDatasetWriter:
    # No cameras: the writer then needs neither ffmpeg nor depth sidecars.
    features = build_features(MOTORS, {}, fps=30, use_video=True)
    return LeRobotDatasetWriter(root, "u/d", 30, "so100_follower", features, MOTORS, resume=resume)


def record_episode(writer: LeRobotDatasetWriter, frames: int, offset: float) -> None:
    for frame in range(frames):
        obs = {f"{motor}.pos": offset + frame + index for index, motor in enumerate(MOTORS)}
        action = {f"{motor}.pos": offset + frame + index + 0.5 for index, motor in enumerate(MOTORS)}
        writer.add_frame(obs, action, "pick")
    writer.save_episode()


class WriterResumeTests(unittest.TestCase):
    def test_resume_appends_episodes_to_an_existing_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            first = make_writer(root)
            first.create()
            record_episode(first, 3, 0.0)
            first.finalize()

            second = make_writer(root, resume=True)
            second.create()
            self.assertEqual(second.total_episodes, 1)
            self.assertEqual(second.total_frames, 3)
            record_episode(second, 2, 100.0)
            second.finalize()

            info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual((info["total_episodes"], info["total_frames"], info["total_tasks"]), (2, 5, 1))
            episodes = pq.read_table(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pylist()
            self.assertEqual([row["episode_index"] for row in episodes], [0, 1])
            self.assertEqual([(row["dataset_from_index"], row["dataset_to_index"]) for row in episodes], [(0, 3), (3, 5)])
            data = pq.read_table(root / "data" / "chunk-000" / "file-001.parquet")
            self.assertEqual(data["index"].to_pylist(), [3, 4])
            self.assertEqual(data["episode_index"].to_pylist(), [1, 1])

    def test_new_run_refuses_to_overwrite_an_existing_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            writer = make_writer(root)
            writer.create()
            record_episode(writer, 2, 0.0)
            writer.finalize()
            with self.assertRaisesRegex(FileExistsError, "resume"):
                make_writer(root).create()

    def test_unfinished_recording_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            (root / "data" / "chunk-000").mkdir(parents=True)
            (root / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"partial")
            with self.assertRaisesRegex(FileExistsError, "unfinished"):
                make_writer(root).create()

    def test_resume_rejects_a_different_feature_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            writer = make_writer(root)
            writer.create()
            record_episode(writer, 2, 0.0)
            writer.finalize()
            features = build_features(["shoulder_pan"], {}, fps=30)
            other = LeRobotDatasetWriter(root, "u/d", 30, "so100_follower", features, ["shoulder_pan"], resume=True)
            with self.assertRaisesRegex(ValueError, "features differ"):
                other.create()


class DisplayTests(unittest.TestCase):
    def test_depth_to_gray_maps_range_and_marks_invalid(self) -> None:
        depth = np.array([[0, 300], [4000, 2150]], dtype=np.uint16)
        gray = depth_to_gray(depth, vmin=300, vmax=4000)
        self.assertEqual(gray.dtype, np.uint8)
        self.assertEqual(gray[0, 0], 0)
        self.assertEqual(gray[0, 1], 0)
        self.assertEqual(gray[1, 0], 255)
        self.assertTrue(100 < gray[1, 1] < 160)

    def test_disabled_preview_is_a_no_op(self) -> None:
        preview = FramePreview(["front"], enabled=False)
        preview.show({"front": np.zeros((4, 4, 3), dtype=np.uint8)})
        preview.close()

    def test_enabled_preview_never_raises_even_without_a_display(self) -> None:
        preview = FramePreview(["front"], enabled=True)
        preview.show({"front": np.zeros((8, 8, 3), dtype=np.uint8), "front_depth": np.full((8, 8, 1), 1000, np.uint16)})
        preview.close()


if __name__ == "__main__":
    unittest.main()
