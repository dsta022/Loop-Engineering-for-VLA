from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.data_quality_audit import audit_dataset
from pipeline.dataset_clean_build import build_clean_dataset
from pipeline.dataset_merge import merge_datasets
from pipeline.feedback_dataset_build import dataset_episode_lengths
from tests.test_dataset_etl_pipeline import (
    RGB_KEY,
    write_depth_png,
    write_pattern_then_black_video,
    write_rgb_dataset,
)
from utils.dataset_metadata import episode_metadata_paths, read_episode_metadata
from utils.hf_upload_utils import validate_lerobot_dataset_for_upload
from utils.media_utils import ffmpeg_available, read_png_array, read_png_pure_python
from utils.version import __version__


def split_episode_metadata(dataset: Path, first_rows: int) -> None:
    """Rewrite meta/episodes as two shards: rows [0, first_rows) and the rest."""
    original = dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(original)
    pq.write_table(table.slice(0, first_rows), original)
    second = dataset / "meta" / "episodes" / "chunk-000" / "file-001.parquet"
    pq.write_table(table.slice(first_rows), second)


def run_audit(dataset: Path, out_dir: Path, **kwargs) -> int:
    return audit_dataset(
        root=dataset,
        out_dir=out_dir,
        depth_check="header",
        video_check="metadata",
        video_content_check="none",
        sample_frames=0,
        depth_sample_frames=0,
        duplicate_check="none",
        video_sample_stride=30,
        **kwargs,
    )


class MultiShardMetadataTests(unittest.TestCase):
    def test_reader_concatenates_shards_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = write_rgb_dataset(Path(tmp), "sharded", [3, 2, 2], 10)
            split_episode_metadata(dataset, 2)
            self.assertEqual(len(episode_metadata_paths(dataset)), 2)
            table = read_episode_metadata(dataset, columns=["episode_index", "length"])
            self.assertEqual(table["episode_index"].to_pylist(), [0, 1, 2])
            self.assertEqual(dataset_episode_lengths(dataset), [3, 2, 2])

    def test_audit_merge_and_clean_accept_sharded_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_rgb_dataset(root, "sharded_dataset", [3, 2, 2], 10)
            split_episode_metadata(dataset, 2)
            report_dir = root / "report"
            self.assertEqual(run_audit(dataset, report_dir, workers=2), 0)
            summary = json.loads((report_dir / "quality_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["episodes"], 3)
            self.assertEqual(summary["audit_script_version"], __version__)
            self.assertIn("audit_time_utc", summary)
            self.assertEqual(summary["workers"], 2)

            merged = root / "merged"
            merge_datasets(src_root=root, out_dir=merged, overwrite=False, dataset_names=["sharded_dataset"], copy_mode="copy")
            info = json.loads((merged / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["total_episodes"], 3)

            with (report_dir / "quality_report.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[1]["status"] = "review"
            rows[1]["review_reasons"] = "low_motion_review"
            with (report_dir / "quality_report.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (report_dir / "drop_episodes.txt").write_text("2\n", encoding="utf-8")
            cleaned = root / "cleaned"
            build_clean_dataset(
                src_dataset=dataset,
                out_dir=cleaned,
                drop_list=report_dir / "drop_episodes.txt",
                overwrite=False,
                video_mode="copy-referenced",
                copy_mode="copy",
                ffmpeg_path=None,
                video_crf=30,
                dry_run=False,
            )
            manifest = json.loads((cleaned / "cleaning_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["audit_script_version"], __version__)
            self.assertEqual(manifest["audit_time"], summary["audit_time_utc"])
            self.assertEqual(manifest["thresholds"], summary["thresholds"])
            self.assertEqual(manifest["reviewed_episodes"], [1])
            self.assertEqual(manifest["dropped_episodes"], [2])
            self.assertEqual(manifest["build_tool_version"], __version__)
            self.assertEqual(len(episode_metadata_paths(cleaned)), 1)

    def test_upload_validation_accepts_sharded_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = write_rgb_dataset(Path(tmp), "upload_set", [2], 10)
            (dataset / "depth_sidecar").mkdir()
            split_episode_metadata(dataset, 1)
            info = validate_lerobot_dataset_for_upload(dataset, strict_no_reparse=True)
            self.assertEqual(info["total_episodes"], 1)

    def test_missing_metadata_shards_fail_the_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = write_rgb_dataset(Path(tmp), "no_shards", [2], 10)
            (dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet").unlink()
            report_dir = Path(tmp) / "report"
            self.assertEqual(run_audit(dataset, report_dir), 1)
            summary = json.loads((report_dir / "quality_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(any("chunk-*" in error for error in summary["global_errors"]))


class PngFastPathTests(unittest.TestCase):
    def test_cv2_and_pure_python_decoders_agree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "depth.png"
            depth = (np.arange(6 * 4, dtype=np.uint16).reshape(6, 4) * 300 + 7).astype(np.uint16)
            write_depth_png(path, depth)
            fast = read_png_array(path)
            slow = read_png_pure_python(path)
            self.assertEqual(fast.shape, (6, 4, 1))
            self.assertTrue(np.array_equal(fast[:, :, 0].astype(np.uint16), depth))
            self.assertTrue(np.array_equal(slow[:, :, 0].astype(np.uint16), depth))

    def test_corrupt_png_still_raises_a_structured_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.png"
            path.write_bytes(b"not a png at all")
            with self.assertRaisesRegex(ValueError, "bad_png_signature"):
                read_png_array(path)


@unittest.skipUnless(ffmpeg_available(), "ffmpeg is not available")
class ParallelVideoPrewarmTests(unittest.TestCase):
    def test_workers_produce_the_same_verdicts_as_the_sequential_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_rgb_dataset(root, "shared_video_set", [30, 30], 10)
            video_dir = dataset / "videos" / RGB_KEY / "chunk-000"
            (video_dir / "file-001.mp4").unlink()
            write_pattern_then_black_video(video_dir / "file-000.mp4", ffmpeg_available())
            episode_path = dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
            rows = pq.read_table(episode_path).to_pylist()
            for row in rows:
                episode = int(row["episode_index"])
                row[f"videos/{RGB_KEY}/file_index"] = 0
                row[f"videos/{RGB_KEY}/from_timestamp"] = float(episode)
                row[f"videos/{RGB_KEY}/to_timestamp"] = float(episode) + 1.0
            pq.write_table(pa.table({key: [row[key] for row in rows] for key in rows[0]}), episode_path)
            # The synthetic info.json declares a 2x2 placeholder video; the real test
            # video is 64x48, so declare that and drop the codec expectation so that
            # ffprobe (when installed) validates the file instead of rejecting it.
            info_path = dataset / "meta" / "info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["features"][RGB_KEY]["shape"] = [48, 64, 3]
            info["features"][RGB_KEY]["info"].pop("video.codec", None)
            info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

            verdicts = {}
            for workers in (1, 3):
                report_dir = root / f"report_{workers}"
                audit_dataset(
                    root=dataset,
                    out_dir=report_dir,
                    depth_check="none",
                    video_check="decode",
                    video_content_check="sample",
                    sample_frames=4,
                    depth_sample_frames=0,
                    duplicate_check="none",
                    video_sample_stride=8,
                    motion_check="none",
                    workers=workers,
                )
                with (report_dir / "quality_report.csv").open(newline="", encoding="utf-8") as handle:
                    verdicts[workers] = {
                        int(row["episode_index"]): (row["status"], row["review_reasons"])
                        for row in csv.DictReader(handle)
                    }
            self.assertEqual(verdicts[1], verdicts[3])
            self.assertTrue(all(status != "drop" for status, _ in verdicts[3].values()), verdicts[3])
            self.assertIn("video_file_black_high_review", verdicts[3][1][1])
            self.assertNotIn("video_file_black_high_review", verdicts[3][0][1])


if __name__ == "__main__":
    unittest.main()
