"""Streaming writer that emits a LeRobot **v3.0** dataset without lerobot.

Layout produced (one file per episode, indexed by episode number), matching what
the toolkit's ``pipeline/dataset_merge.py`` + ``pipeline/data_quality_audit.py``
read:

    meta/info.json                          features / fps / totals / templates
    meta/tasks.parquet                      task_index, task
    meta/episodes/chunk-000/file-000.parquet   per-episode metadata + stats
    meta/stats.json                         aggregated stats (build_stats_json)
    data/chunk-000/file-000.parquet         per-frame state/action/scalars (+depth struct)
    videos/<key>/chunk-000/file-000.mp4     per-episode RGB video
    depth_sidecar/<key>/episode-XXXXXX/frame-XXXXXX.png   uint16 depth
    meta/rgbd_vla_depth_recording.json      depth manifest (when depth present)

Frames are streamed to ffmpeg and the depth sidecar as they arrive; only the
small numeric columns and running image-stat accumulators are kept in memory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from utils.dataset_io import write_json
from utils.dataset_metadata import (
    build_stats_json,
    chunk_index_for,
    file_index_in_chunk,
    read_episode_metadata,
)

from .depth_sidecar import (
    build_depth_recording_manifest,
    depth_camera_name_from_key,
    encode_depth_png,
    save_depth_sidecar_frame,
)
from .features import (
    ACTION_KEY,
    IMAGE_PREFIX,
    STATE_KEY,
    action_vector,
    image_keys,
    state_vector,
    video_keys,
)
from .stats import ImageStatAccumulator, scalar_stats, vector_stats
from .video_encoder import VideoStream

logger = logging.getLogger(__name__)

CODEBASE_VERSION = "v3.0"
CHUNKS_SIZE = 1000
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
DEPTH_STRUCT_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
SCALAR_STAT_FEATURES = ("timestamp", "frame_index", "episode_index", "index", "task_index")


class LeRobotDatasetWriter:
    def __init__(
        self,
        root: str | Path,
        repo_id: str,
        fps: int,
        robot_type: str,
        features: dict[str, dict[str, Any]],
        motor_names: list[str],
        *,
        vcodec: str = "libsvtav1",
        encoder_threads: int | None = None,
        resume: bool = False,
    ):
        self.root = Path(root)
        self.resume = bool(resume)
        self.repo_id = repo_id
        self.fps = int(fps)
        self.robot_type = robot_type
        self.features = features
        self.motor_names = list(motor_names)
        self.vcodec = vcodec
        self.encoder_threads = encoder_threads

        self.video_keys = video_keys(features)
        self.depth_keys = [k for k in image_keys(features) if depth_camera_name_from_key(k)]
        self._video_cam = {k: k[len(IMAGE_PREFIX):] for k in self.video_keys}
        self._depth_obs_key = {k: f"{depth_camera_name_from_key(k)}_depth" for k in self.depth_keys}

        self.total_episodes = 0
        self.total_frames = 0
        self._tasks: dict[str, int] = {}
        self._episode_rows: list[dict[str, Any]] = []
        self._reset_episode()

    @property
    def current_episode_length(self) -> int:
        """Number of frames buffered in the in-progress (not yet saved) episode."""
        return self._len

    # -- lifecycle --------------------------------------------------------- #
    def create(self) -> None:
        info_path = self.root / "meta" / "info.json"
        if info_path.exists():
            if not self.resume:
                raise FileExistsError(
                    f"dataset already exists at {self.root}; pass --resume=true to append "
                    "episodes to it, or choose another dataset.root"
                )
            self._load_existing(info_path)
        elif any((self.root / "data").rglob("*.parquet")):
            raise FileExistsError(
                f"{self.root} holds an unfinished recording (data files without meta/info.json); "
                "remove it or choose another dataset.root"
            )
        for sub in ("data", "videos", "meta"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        if self.depth_keys:
            (self.root / "depth_sidecar").mkdir(parents=True, exist_ok=True)

    def _load_existing(self, info_path: Path) -> None:
        """Continue an existing dataset: episode numbering, tasks, and metadata rows."""
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("features") != self.features:
            raise ValueError(
                "cannot resume: the existing dataset's features differ from the current "
                "camera/robot configuration"
            )
        if int(info.get("fps", -1)) != self.fps:
            raise ValueError(f"cannot resume: existing fps {info.get('fps')} != configured {self.fps}")
        tasks = pq.read_table(self.root / "meta" / "tasks.parquet").to_pylist()
        self._tasks = {str(row["task"]): int(row["task_index"]) for row in tasks}
        self._episode_rows = read_episode_metadata(self.root).to_pylist()
        self.total_episodes = int(info["total_episodes"])
        self.total_frames = int(info["total_frames"])
        if len(self._episode_rows) != self.total_episodes:
            raise ValueError(
                f"cannot resume: {len(self._episode_rows)} episode metadata rows but "
                f"info.json declares {self.total_episodes} episodes"
            )
        logger.info(
            "resuming %s at episode %d (%d frames, %d tasks)",
            self.root, self.total_episodes, self.total_frames, len(self._tasks),
        )

    def _reset_episode(self) -> None:
        self._started = False
        self._len = 0
        self._states: list[list[float]] = []
        self._actions: list[list[float]] = []
        self._timestamps: list[float] = []
        self._task_indices: list[int] = []
        self._episode_tasks: list[str] = []
        self._depth_structs: dict[str, list[dict]] = {k: [] for k in self.depth_keys}
        self._video_streams: dict[str, VideoStream] = {}
        self._image_accum: dict[str, ImageStatAccumulator] = {}
        self._depth_accum: dict[str, ImageStatAccumulator] = {}

    def _task_index(self, task: str) -> int:
        if task not in self._tasks:
            self._tasks[task] = len(self._tasks)
        return self._tasks[task]

    def _start_episode(self) -> None:
        episode = self.total_episodes
        chunk = chunk_index_for(episode, CHUNKS_SIZE)
        file_index = file_index_in_chunk(episode, CHUNKS_SIZE)
        for vkey in self.video_keys:
            spec = self.features[vkey]
            height, width = int(spec["shape"][0]), int(spec["shape"][1])
            out = self.root / "videos" / vkey / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
            self._video_streams[vkey] = VideoStream(
                out, fps=self.fps, width=width, height=height,
                vcodec=self.vcodec, encoder_threads=self.encoder_threads,
            )
            self._image_accum[vkey] = ImageStatAccumulator(3, divisor=255.0)
        for dkey in self.depth_keys:
            self._depth_accum[dkey] = ImageStatAccumulator(1, divisor=1.0)
        self._started = True

    def add_frame(self, observation: dict[str, Any], action: dict[str, Any], task: str) -> None:
        if not self._started:
            self._start_episode()
        episode = self.total_episodes
        frame_index = self._len

        self._states.append(state_vector(observation, self.motor_names))
        self._actions.append(action_vector(action, self.motor_names))
        self._timestamps.append(frame_index / self.fps)
        self._task_indices.append(self._task_index(task))
        if task not in self._episode_tasks:
            self._episode_tasks.append(task)

        for vkey in self.video_keys:
            rgb = np.asarray(observation[self._video_cam[vkey]])
            self._video_streams[vkey].write(rgb)
            self._image_accum[vkey].update(rgb)

        for dkey in self.depth_keys:
            depth = np.asarray(observation[self._depth_obs_key[dkey]])
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            png_bytes = encode_depth_png(depth.astype(np.uint16))
            save_depth_sidecar_frame(self.root, dkey, episode, frame_index, depth.astype(np.uint16))
            self._depth_structs[dkey].append({"bytes": png_bytes, "path": f"frame-{frame_index:06d}.png"})
            self._depth_accum[dkey].update(depth)

        self._len += 1

    def save_episode(self) -> None:
        if self._len == 0:
            self._reset_episode()
            return
        episode = self.total_episodes
        frame_counts = {vkey: stream.close() for vkey, stream in self._video_streams.items()}
        for vkey, count in frame_counts.items():
            if count != self._len:
                logger.warning("video %s wrote %d frames but episode length is %d", vkey, count, self._len)

        self._write_data_parquet(episode)
        self._episode_rows.append(self._build_episode_row(episode))

        self.total_frames += self._len
        self.total_episodes += 1
        self._reset_episode()

    def clear_episode_buffer(self) -> None:
        """Discard the in-progress episode (re-record): abort video + delete sidecars."""
        for stream in self._video_streams.values():
            stream.abort()
        episode = self.total_episodes
        for dkey in self.depth_keys:
            ep_dir = self.root / "depth_sidecar" / dkey / f"episode-{episode:06d}"
            if ep_dir.exists():
                for png in ep_dir.glob("*.png"):
                    png.unlink(missing_ok=True)
        self._reset_episode()

    # -- parquet writers --------------------------------------------------- #
    def _write_data_parquet(self, episode: int) -> None:
        chunk = chunk_index_for(episode, CHUNKS_SIZE)
        file_index = file_index_in_chunk(episode, CHUNKS_SIZE)
        path = self.root / "data" / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)

        global_from = self.total_frames
        fields = [
            (STATE_KEY, pa.array(self._states, type=pa.list_(pa.float32()))),
            (ACTION_KEY, pa.array(self._actions, type=pa.list_(pa.float32()))),
            ("timestamp", pa.array(self._timestamps, type=pa.float32())),
            ("frame_index", pa.array(list(range(self._len)), type=pa.int64())),
            ("episode_index", pa.array([episode] * self._len, type=pa.int64())),
            ("index", pa.array(list(range(global_from, global_from + self._len)), type=pa.int64())),
            ("task_index", pa.array(self._task_indices, type=pa.int64())),
        ]
        for dkey in self.depth_keys:
            fields.append((dkey, pa.array(self._depth_structs[dkey], type=DEPTH_STRUCT_TYPE)))

        table = pa.table({name: array for name, array in fields})
        pq.write_table(table, path)

    def _build_episode_row(self, episode: int) -> dict[str, Any]:
        chunk = chunk_index_for(episode, CHUNKS_SIZE)
        file_index = file_index_in_chunk(episode, CHUNKS_SIZE)
        global_from = self.total_frames
        row: dict[str, Any] = {
            "episode_index": episode,
            "tasks": list(self._episode_tasks),
            "length": self._len,
            "data/chunk_index": chunk,
            "data/file_index": file_index,
            "dataset_from_index": global_from,
            "dataset_to_index": global_from + self._len,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        duration = self._len / self.fps
        for vkey in self.video_keys:
            row[f"videos/{vkey}/chunk_index"] = chunk
            row[f"videos/{vkey}/file_index"] = file_index
            row[f"videos/{vkey}/from_timestamp"] = 0.0
            row[f"videos/{vkey}/to_timestamp"] = duration
        self._attach_stats(row)
        return row

    def _attach_stats(self, row: dict[str, Any]) -> None:
        n = self._len
        states = np.asarray(self._states, dtype=np.float64)
        actions = np.asarray(self._actions, dtype=np.float64)
        feature_stats: dict[str, dict[str, Any]] = {
            STATE_KEY: vector_stats(states, n),
            ACTION_KEY: vector_stats(actions, n),
            "timestamp": scalar_stats(np.asarray(self._timestamps), n),
            "frame_index": scalar_stats(np.arange(n), n),
            "episode_index": scalar_stats(np.full(n, self.total_episodes), n),
            "index": scalar_stats(np.arange(self.total_frames, self.total_frames + n), n),
            "task_index": scalar_stats(np.asarray(self._task_indices), n),
        }
        for vkey in self.video_keys:
            feature_stats[vkey] = self._image_accum[vkey].result()
        for dkey in self.depth_keys:
            feature_stats[dkey] = self._depth_accum[dkey].result()

        for feature, stats in feature_stats.items():
            for stat, value in stats.items():
                row[f"stats/{feature}/{stat}"] = value

    # -- finalize ---------------------------------------------------------- #
    def finalize(self) -> None:
        if self.total_episodes == 0:
            raise RuntimeError("no episodes were recorded; nothing to finalize")
        self._write_tasks()
        episodes_table = self._write_episodes()
        self._write_info()
        write_json(self.root / "meta" / "stats.json", build_stats_json(episodes_table))
        if self.depth_keys:
            write_json(
                self.root / "meta" / "rgbd_vla_depth_recording.json",
                build_depth_recording_manifest(self.features),
            )

    def _write_tasks(self) -> None:
        ordered = sorted(self._tasks.items(), key=lambda kv: kv[1])
        table = pa.table(
            {
                "task_index": pa.array([idx for _, idx in ordered], type=pa.int64()),
                "task": pa.array([task for task, _ in ordered], type=pa.string()),
            }
        )
        pq.write_table(table, self.root / "meta" / "tasks.parquet")

    def _write_episodes(self) -> pa.Table:
        table = pa.Table.from_pylist(self._episode_rows)
        out = self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out)
        return table

    def _write_info(self) -> None:
        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "total_episodes": self.total_episodes,
            "total_frames": self.total_frames,
            "total_tasks": len(self._tasks),
            "total_videos": self.total_episodes * len(self.video_keys),
            "chunks_size": CHUNKS_SIZE,
            "fps": self.fps,
            "splits": {"train": f"0:{self.total_episodes}"},
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH,
            "features": self.features,
        }
        write_json(self.root / "meta" / "info.json", info)
