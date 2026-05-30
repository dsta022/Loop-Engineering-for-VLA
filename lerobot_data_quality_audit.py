#!/usr/bin/env python3
"""Full conservative data quality audit for LeRobot-style RGB and RGB-D VLA datasets.

The policy is conservative:
  - objective, deterministic failures are marked drop
  - sampled content, distribution, and motion anomalies are marked review
  - semantic task-success checks are reported as not checked unless a detector
    or VLM module is added later

This script does not delete source data directly. Use the emitted keep/review/
drop lists to build a cleaned dataset after review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from utils.lerobot_io import discover_reparse_paths, is_reparse_path, read_json, safe_rel_path
from utils.lerobot_metadata import feature_names_by_dtype, feature_shape, video_expected_codec
from utils.media_utils import (
    depth_content_metrics,
    ffmpeg_available,
    ffmpeg_decode_video,
    ffprobe_available,
    ffprobe_video,
    sample_video_file,
    validate_png,
    video_content_metrics,
)
from utils.quality_utils import (
    EpisodeQuality,
    build_episode_columns,
    column_matrix,
    default_audit_output_dir,
    evenly_spaced_indices,
    hash_array,
    robust_upper_threshold,
    write_quality_outputs,
)


@dataclass(frozen=True)
class AuditConfig:
    root: Path
    out_dir: Path
    depth_check: str
    video_check: str
    video_content_check: str
    sample_frames: int
    depth_sample_frames: int
    duplicate_check: str
    video_sample_stride: int


@dataclass(frozen=True)
class DatasetMetadata:
    info: dict[str, Any]
    tasks: pa.Table
    episodes: pa.Table
    episode_cols: dict[str, list[Any]]
    video_keys: list[str]
    image_keys: list[str]
    depth_keys: list[str]
    action_dim: int | None
    state_dim: int | None
    total_episodes: int
    total_frames: int
    total_tasks: int


@dataclass(frozen=True)
class EpisodeRecord:
    row: int
    episode_index: int
    length: int
    data_chunk: int
    data_file: int
    dataset_from: int
    dataset_to: int

    @classmethod
    def from_columns(cls, row: int, columns: dict[str, list[Any]]) -> "EpisodeRecord":
        return cls(
            row=row,
            episode_index=int(columns["episode_index"][row]),
            length=int(columns["length"][row]),
            data_chunk=int(columns["data/chunk_index"][row]),
            data_file=int(columns["data/file_index"][row]),
            dataset_from=int(columns["dataset_from_index"][row]),
            dataset_to=int(columns["dataset_to_index"][row]),
        )


@dataclass
class AuditRuntime:
    video_check: str
    video_content_check: str
    ffprobe_path: str | None
    ffmpeg_path: str | None
    global_not_checked: list[str] = field(default_factory=list)


@dataclass
class AuditCaches:
    video_probe: dict[Path, tuple[bool, str]] = field(default_factory=dict)
    video_decode: dict[Path, tuple[bool, str]] = field(default_factory=dict)
    video_content: dict[Path, tuple[dict[str, float] | None, str]] = field(default_factory=dict)


@dataclass
class AuditState:
    audits: list[EpisodeQuality] = field(default_factory=list)
    audits_by_episode: dict[int, EpisodeQuality] = field(default_factory=dict)
    action_jump_values: list[float] = field(default_factory=list)
    state_jump_values: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    trajectory_hashes: dict[str, list[int]] = field(default_factory=dict)
    running_global_index: int = 0

    def register(self, audit: EpisodeQuality) -> None:
        self.audits.append(audit)
        self.audits_by_episode[audit.episode_index] = audit
        self.lengths.append(audit.length)


def required_dataset_paths(root: Path) -> list[Path]:
    return [
        root / "meta" / "info.json",
        root / "meta" / "stats.json",
        root / "meta" / "tasks.parquet",
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        root / "data",
        root / "videos",
    ]


def write_failure(out_dir: Path, root: Path, errors: list[str], not_checked: list[str] | None = None) -> int:
    summary = {
        "dataset": str(root),
        "status": "fail",
        "global_errors": errors,
        "global_not_checked": not_checked or [],
        "episodes": 0,
    }
    write_quality_outputs(out_dir, [], summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1


def load_metadata(config: AuditConfig) -> DatasetMetadata:
    info = read_json(config.root / "meta" / "info.json")
    video_keys = feature_names_by_dtype(info, "video")
    image_keys = feature_names_by_dtype(info, "image")
    depth_keys = depth_sidecar_keys(config.root, image_keys)
    action_shape = feature_shape(info, "action")
    state_shape = feature_shape(info, "observation.state")
    tasks = pq.read_table(config.root / "meta" / "tasks.parquet")
    episodes = pq.read_table(
        config.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        columns=build_episode_columns(video_keys),
    )
    return DatasetMetadata(
        info=info,
        tasks=tasks,
        episodes=episodes,
        episode_cols=episodes.to_pydict(),
        video_keys=video_keys,
        image_keys=image_keys,
        depth_keys=depth_keys,
        action_dim=int(action_shape[0]) if action_shape else None,
        state_dim=int(state_shape[0]) if state_shape else None,
        total_episodes=int(info.get("total_episodes", -1)),
        total_frames=int(info.get("total_frames", -1)),
        total_tasks=int(info.get("total_tasks", -1)),
    )


def depth_sidecar_keys(root: Path, image_keys: list[str]) -> list[str]:
    """Return image features that use this project's depth_sidecar layout.

    Standard LeRobot RGB datasets commonly contain only video features and no
    depth_sidecar directory. Those datasets should still pass RGB/action/state
    auditing instead of failing a depth-specific path check.
    """
    if not image_keys:
        return []

    manifest = root / "meta" / "rgbd_vla_depth_recording.json"
    if manifest.exists():
        try:
            payload = read_json(manifest)
            streams = payload.get("depth_streams", [])
            if isinstance(streams, list):
                return [str(key) for key in streams if str(key) in image_keys]
        except Exception:
            pass

    sidecar_root = root / "depth_sidecar"
    if not sidecar_root.exists():
        return []
    return [key for key in image_keys if (sidecar_root / key).exists()]


def validate_global_metadata(root: Path, metadata: DatasetMetadata) -> list[str]:
    errors: list[str] = []
    if metadata.episodes.num_rows != metadata.total_episodes:
        errors.append(f"metadata_episode_rows_mismatch:{metadata.episodes.num_rows}!={metadata.total_episodes}")
    if metadata.tasks.num_rows != metadata.total_tasks:
        errors.append(f"tasks_rows_mismatch:{metadata.tasks.num_rows}!={metadata.total_tasks}")

    episode_indexes = [int(v) for v in metadata.episode_cols["episode_index"]]
    if sorted(episode_indexes) != list(range(metadata.total_episodes)):
        errors.append("episode_index_metadata_not_contiguous")
    if sum(int(v) for v in metadata.episode_cols["length"]) != metadata.total_frames:
        errors.append("episode_lengths_sum_mismatch_total_frames")

    reparse_paths = discover_reparse_paths(root)
    if reparse_paths:
        errors.append(f"reparse_paths_present:{[str(p) for p in reparse_paths[:20]]}")
    return errors


def resolve_runtime(config: AuditConfig) -> AuditRuntime:
    runtime = AuditRuntime(
        video_check=config.video_check,
        video_content_check=config.video_content_check,
        ffprobe_path=ffprobe_available(),
        ffmpeg_path=ffmpeg_available(),
    )
    if runtime.video_check in {"ffprobe", "decode"} and runtime.ffprobe_path is None:
        runtime.global_not_checked.append("video_ffprobe_missing")
        runtime.video_check = "metadata"
    if runtime.video_check == "decode" and runtime.ffmpeg_path is None:
        runtime.global_not_checked.append("video_ffmpeg_missing")
        runtime.video_check = "ffprobe" if runtime.ffprobe_path else "metadata"
    if runtime.video_content_check != "none" and runtime.ffmpeg_path is None:
        runtime.global_not_checked.append("video_content_ffmpeg_missing")
        runtime.video_content_check = "none"
    return runtime


def requested_data_columns(depth_keys: list[str]) -> list[str]:
    columns = [
        "action",
        "observation.state",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    columns.extend(key for key in depth_keys if key not in columns)
    return columns


def read_episode_data(
    root: Path,
    record: EpisodeRecord,
    depth_keys: list[str],
    audit: EpisodeQuality,
) -> pa.Table | None:
    data_path = root / "data" / f"chunk-{record.data_chunk:03d}" / f"file-{record.data_file:03d}.parquet"
    if not data_path.exists():
        audit.drop(f"missing_data_parquet:{data_path}")
        return None
    if data_path.is_symlink() or is_reparse_path(data_path):
        audit.drop(f"data_parquet_is_link:{data_path}")
        return None
    try:
        data = pq.read_table(data_path, columns=requested_data_columns(depth_keys))
    except Exception as exc:
        audit.drop(f"cannot_read_data_parquet:{type(exc).__name__}")
        return None
    audit.data_rows = data.num_rows
    return data


def validate_episode_ranges(record: EpisodeRecord, audit: EpisodeQuality, state: AuditState) -> None:
    if record.dataset_to - record.dataset_from != record.length:
        audit.drop("dataset_range_length_mismatch")
    expected_to = state.running_global_index + record.length
    if record.dataset_from != state.running_global_index or record.dataset_to != expected_to:
        audit.drop("dataset_global_range_not_contiguous")
    state.running_global_index = record.dataset_to


def validate_index_columns(
    data: pa.Table,
    record: EpisodeRecord,
    total_tasks: int,
    audit: EpisodeQuality,
) -> None:
    if data.num_rows != record.length:
        audit.drop("data_rows_mismatch_length")

    try:
        ep_values = np.asarray(data["episode_index"].to_pylist(), dtype=np.int64)
        if ep_values.shape[0] != record.length or not np.all(ep_values == record.episode_index):
            audit.drop("episode_index_values_mismatch")
    except Exception:
        audit.drop("cannot_validate_episode_index")

    try:
        frame_values = np.asarray(data["frame_index"].to_pylist(), dtype=np.int64)
        if frame_values.shape[0] != record.length or not np.array_equal(frame_values, np.arange(record.length)):
            audit.drop("frame_index_not_contiguous")
    except Exception:
        audit.drop("cannot_validate_frame_index")

    try:
        global_values = np.asarray(data["index"].to_pylist(), dtype=np.int64)
        expected_global = np.arange(record.dataset_from, record.dataset_to, dtype=np.int64)
        if global_values.shape[0] != record.length or not np.array_equal(global_values, expected_global):
            audit.drop("global_index_not_contiguous")
    except Exception:
        audit.drop("cannot_validate_global_index")

    try:
        task_values = np.asarray(data["task_index"].to_pylist(), dtype=np.int64)
        if task_values.shape[0] != record.length:
            audit.drop("task_index_length_mismatch")
        elif np.any(task_values < 0) or np.any(task_values >= total_tasks):
            audit.drop("task_index_out_of_range")
    except Exception:
        audit.drop("cannot_validate_task_index")

    timestamps = column_matrix(data, "timestamp", np.float64)
    if timestamps is None:
        audit.drop("cannot_validate_timestamp")
        return
    timestamps = timestamps.reshape(-1)
    if timestamps.shape[0] != record.length:
        audit.drop("timestamp_length_mismatch")
    if not np.isfinite(timestamps).all():
        audit.drop("timestamp_nan_or_inf")
    elif np.any(np.diff(timestamps) < 0):
        audit.drop("timestamp_not_monotonic")


def validate_numeric_array(
    name: str,
    values: np.ndarray | None,
    expected_dim: int | None,
    audit: EpisodeQuality,
) -> bool:
    if values is None:
        audit.drop(f"missing_or_unreadable_numeric:{name}")
        return False
    if values.ndim != 2:
        audit.drop(f"bad_numeric_rank:{name}:{values.ndim}")
        return False
    if expected_dim is not None and values.shape[1] != expected_dim:
        audit.drop(f"bad_numeric_dim:{name}:{values.shape[1]}!={expected_dim}")
    if not np.isfinite(values).all():
        audit.drop(f"nan_or_inf:{name}")
        return False
    return True


def track_numeric_metrics(
    action: np.ndarray | None,
    state_values: np.ndarray | None,
    record: EpisodeRecord,
    audit: EpisodeQuality,
    audit_state: AuditState,
) -> None:
    if action is not None and action.ndim == 2 and np.isfinite(action).all():
        action_range = float(np.max(np.ptp(action, axis=0))) if action.size else 0.0
        action_jump = float(np.max(np.abs(np.diff(action, axis=0)))) if len(action) > 1 else 0.0
        audit.metrics["action_range_max"] = action_range
        audit.metrics["action_jump_max"] = action_jump
        audit_state.action_jump_values.append(action_jump)

    if state_values is not None and state_values.ndim == 2 and np.isfinite(state_values).all():
        state_range = float(np.max(np.ptp(state_values, axis=0))) if state_values.size else 0.0
        state_jump = float(np.max(np.abs(np.diff(state_values, axis=0)))) if len(state_values) > 1 else 0.0
        audit.metrics["state_range_max"] = state_range
        audit.metrics["state_jump_max"] = state_jump
        audit_state.state_jump_values.append(state_jump)

    if action is not None and state_values is not None and action.shape[0] == state_values.shape[0] == record.length:
        digest = hashlib.sha256()
        digest.update(hash_array(action).encode("ascii"))
        digest.update(hash_array(state_values).encode("ascii"))
        digest.update(str(record.length).encode("ascii"))
        audit_state.trajectory_hashes.setdefault(digest.hexdigest(), []).append(record.episode_index)


def audit_numeric_data(
    data: pa.Table,
    metadata: DatasetMetadata,
    record: EpisodeRecord,
    audit: EpisodeQuality,
    audit_state: AuditState,
) -> None:
    validate_index_columns(data, record, metadata.total_tasks, audit)
    action = column_matrix(data, "action", np.float64)
    state_values = column_matrix(data, "observation.state", np.float64)
    validate_numeric_array("action", action, metadata.action_dim, audit)
    validate_numeric_array("observation.state", state_values, metadata.state_dim, audit)
    track_numeric_metrics(action, state_values, record, audit, audit_state)


def audit_depth_sidecars(
    config: AuditConfig,
    metadata: DatasetMetadata,
    record: EpisodeRecord,
    data: pa.Table | None,
    audit: EpisodeQuality,
) -> None:
    if not metadata.depth_keys:
        if config.depth_check != "none":
            audit.skip("depth_sidecar_not_present")
        return

    for image_key in metadata.depth_keys:
        expected_shape = feature_shape(metadata.info, image_key)
        sidecar_dir = config.root / "depth_sidecar" / image_key / f"episode-{record.episode_index:06d}"
        if not sidecar_dir.exists():
            audit.drop(f"missing_image_sidecar_dir:{image_key}")
            continue
        if is_reparse_path(sidecar_dir):
            audit.drop(f"image_sidecar_dir_is_link:{image_key}")
            continue
        if data is None or image_key not in data.column_names:
            audit.drop(f"missing_image_column:{image_key}")
            continue
        path_values = read_image_paths(data, image_key, audit)
        if path_values is None:
            continue
        seen = validate_depth_paths(sidecar_dir, image_key, path_values, expected_shape, config.depth_check, audit)
        audit.depth_files += len(set(seen))
        if len(seen) != record.length:
            audit.drop(f"image_path_count_mismatch:{image_key}:{len(seen)}!={record.length}")
        review_depth_content(image_key, seen, config.depth_sample_frames, audit)


def read_image_paths(data: pa.Table, image_key: str, audit: EpisodeQuality) -> list[Path | None] | None:
    try:
        image_col = data[image_key].combine_chunks()
        return [safe_rel_path(v.as_py()) for v in image_col.field("path")]
    except Exception:
        audit.drop(f"cannot_read_image_paths:{image_key}")
        return None


def validate_depth_paths(
    sidecar_dir: Path,
    image_key: str,
    path_values: list[Path | None],
    expected_shape: list[int] | None,
    depth_check: str,
    audit: EpisodeQuality,
) -> list[Path]:
    seen: list[Path] = []
    for rel_path in path_values:
        if rel_path is None:
            audit.drop(f"missing_image_path_value:{image_key}")
            continue
        image_path = sidecar_dir / rel_path
        seen.append(image_path)
        if not image_path.exists():
            audit.drop(f"missing_image_file:{image_key}:{rel_path}")
            continue
        if not image_path.is_file() or image_path.is_symlink() or is_reparse_path(image_path):
            audit.drop(f"image_file_not_real_file:{image_key}:{rel_path}")
            continue
        try:
            validate_png(image_path, expected_shape, depth_check)
        except Exception as exc:
            audit.drop(f"bad_png:{image_key}:{rel_path}:{exc}")
    return seen


def review_depth_content(
    image_key: str,
    seen: list[Path],
    depth_sample_frames: int,
    audit: EpisodeQuality,
) -> None:
    sampled_metrics: list[dict[str, float | bool]] = []
    for idx in evenly_spaced_indices(len(seen), depth_sample_frames):
        image_path = seen[idx]
        if image_path.exists() and image_path.is_file():
            try:
                sampled_metrics.append(depth_content_metrics(image_path))
            except Exception as exc:
                audit.drop(f"depth_decode_failed:{image_key}:{image_path.name}:{exc}")
    if not sampled_metrics:
        return
    invalid_ratios = [float(m["invalid_ratio"]) for m in sampled_metrics]
    invalid_frame_ratio = float(np.mean(np.asarray(invalid_ratios) > 0.95))
    constant_ratio = float(np.mean([bool(m["constant"]) for m in sampled_metrics]))
    audit.metrics[f"{image_key}.sample_invalid_ratio_mean"] = float(np.mean(invalid_ratios))
    audit.metrics[f"{image_key}.sample_invalid_frame_ratio"] = invalid_frame_ratio
    audit.metrics[f"{image_key}.sample_constant_frame_ratio"] = constant_ratio
    if invalid_frame_ratio > 0.90:
        audit.review(f"depth_invalid_high_review:{image_key}")
    if constant_ratio > 0.90:
        audit.review(f"depth_constant_high_review:{image_key}")


def audit_video_files(
    config: AuditConfig,
    metadata: DatasetMetadata,
    runtime: AuditRuntime,
    caches: AuditCaches,
    record: EpisodeRecord,
    audit: EpisodeQuality,
) -> None:
    for video_key in metadata.video_keys:
        chunk = int(metadata.episode_cols[f"videos/{video_key}/chunk_index"][record.row])
        file_index = int(metadata.episode_cols[f"videos/{video_key}/file_index"][record.row])
        video_path = config.root / "videos" / video_key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
        if not validate_video_path(video_key, video_path, audit):
            continue
        if not probe_video_if_requested(video_key, video_path, metadata, runtime, caches, audit):
            continue
        if not decode_video_if_requested(video_key, video_path, runtime, caches, audit):
            continue
        review_video_content_if_requested(video_key, video_path, config, runtime, caches, audit)


def validate_video_path(video_key: str, video_path: Path, audit: EpisodeQuality) -> bool:
    if not video_path.exists():
        audit.drop(f"missing_video:{video_key}:{video_path}")
        return False
    if not video_path.is_file() or video_path.is_symlink() or is_reparse_path(video_path):
        audit.drop(f"video_not_real_file:{video_key}:{video_path}")
        return False
    if video_path.stat().st_size <= 0:
        audit.drop(f"empty_video:{video_key}:{video_path}")
        return False
    return True


def probe_video_if_requested(
    video_key: str,
    video_path: Path,
    metadata: DatasetMetadata,
    runtime: AuditRuntime,
    caches: AuditCaches,
    audit: EpisodeQuality,
) -> bool:
    if runtime.video_check in {"ffprobe", "decode"} and runtime.ffprobe_path is not None:
        if video_path not in caches.video_probe:
            caches.video_probe[video_path] = ffprobe_video(
                video_path,
                runtime.ffprobe_path,
                feature_shape(metadata.info, video_key),
                video_expected_codec(metadata.info, video_key),
            )
        ok, message = caches.video_probe[video_path]
        if not ok:
            audit.drop(f"video_ffprobe_failed:{video_key}:{message}")
            return False
    elif runtime.video_check in {"ffprobe", "decode"}:
        audit.skip("video_ffprobe_not_checked")
    return True


def decode_video_if_requested(
    video_key: str,
    video_path: Path,
    runtime: AuditRuntime,
    caches: AuditCaches,
    audit: EpisodeQuality,
) -> bool:
    if runtime.video_check == "decode" and runtime.ffmpeg_path is not None:
        if video_path not in caches.video_decode:
            caches.video_decode[video_path] = ffmpeg_decode_video(video_path, runtime.ffmpeg_path)
        ok, message = caches.video_decode[video_path]
        if not ok:
            audit.drop(f"video_ffmpeg_decode_failed:{video_key}:{message}")
            return False
    elif runtime.video_check == "decode":
        audit.skip("video_decode_not_checked")
    return True


def review_video_content_if_requested(
    video_key: str,
    video_path: Path,
    config: AuditConfig,
    runtime: AuditRuntime,
    caches: AuditCaches,
    audit: EpisodeQuality,
) -> None:
    if runtime.video_content_check == "sample" and runtime.ffmpeg_path is not None:
        if video_path not in caches.video_content:
            frames, message = sample_video_file(
                video_path,
                runtime.ffmpeg_path,
                sample_frames=config.sample_frames,
                stride=config.video_sample_stride,
            )
            caches.video_content[video_path] = (
                (None, message) if frames is None else (video_content_metrics(frames), "")
            )
        metrics, message = caches.video_content[video_path]
        if metrics is None:
            audit.review(f"video_content_sample_failed_review:{video_key}:{message}")
            return
        for key, value in metrics.items():
            audit.metrics[f"{video_key}.file_{key}"] = value
        if metrics["black_ratio"] > 0.95:
            audit.review(f"video_file_black_high_review:{video_key}")
        if metrics["white_ratio"] > 0.95:
            audit.review(f"video_file_white_high_review:{video_key}")
        if metrics["constant_ratio"] > 0.95:
            audit.review(f"video_file_near_constant_high_review:{video_key}")
        if metrics["sampled_frames"] >= 4 and metrics["freeze_ratio"] > 0.90:
            audit.review(f"video_file_frozen_high_review:{video_key}")
    elif runtime.video_content_check != "none":
        audit.skip("video_content_not_checked")


def audit_episode(
    config: AuditConfig,
    metadata: DatasetMetadata,
    runtime: AuditRuntime,
    caches: AuditCaches,
    audit_state: AuditState,
    row: int,
) -> None:
    record = EpisodeRecord.from_columns(row, metadata.episode_cols)
    audit = EpisodeQuality(episode_index=record.episode_index, length=record.length)
    audit_state.register(audit)
    validate_episode_ranges(record, audit, audit_state)
    data = read_episode_data(config.root, record, metadata.depth_keys, audit)
    if data is not None:
        audit_numeric_data(data, metadata, record, audit, audit_state)
    audit_depth_sidecars(config, metadata, record, data, audit)
    audit_video_files(config, metadata, runtime, caches, record, audit)
    audit.skip("semantic_task_success_not_checked")
    audit.skip("cup_detector_not_configured")


def apply_distribution_reviews(audits: list[EpisodeQuality], state: AuditState) -> dict[str, float | None]:
    action_threshold = robust_upper_threshold(state.action_jump_values)
    state_threshold = robust_upper_threshold(state.state_jump_values)
    length_threshold = robust_upper_threshold([float(v) for v in state.lengths])
    for audit in audits:
        action_jump = audit.metrics.get("action_jump_max")
        state_jump = audit.metrics.get("state_jump_max")
        action_range = audit.metrics.get("action_range_max")
        state_range = audit.metrics.get("state_range_max")
        if action_threshold is not None and isinstance(action_jump, float) and action_jump > action_threshold:
            audit.review("action_jump_outlier_review")
        if state_threshold is not None and isinstance(state_jump, float) and state_jump > state_threshold:
            audit.review("state_jump_outlier_review")
        if length_threshold is not None and float(audit.length) > length_threshold:
            audit.review("episode_length_high_outlier_review")
        if isinstance(action_range, float) and isinstance(state_range, float) and action_range < 1e-5 and state_range < 1e-5:
            audit.review("low_motion_review")
    return {
        "action_jump_review": action_threshold,
        "state_jump_review": state_threshold,
        "episode_length_high_review": length_threshold,
    }


def apply_duplicate_reviews(state: AuditState, duplicate_check: str) -> None:
    if duplicate_check == "none":
        return
    duplicate_groups = [episodes for episodes in state.trajectory_hashes.values() if len(episodes) > 1]
    for group_idx, group in enumerate(duplicate_groups):
        for episode_index in group:
            audit = state.audits_by_episode.get(episode_index)
            if audit is None:
                continue
            audit.metrics["duplicate_group"] = group_idx
            audit.review("exact_action_state_duplicate_review")


def build_summary(
    config: AuditConfig,
    metadata: DatasetMetadata,
    runtime: AuditRuntime,
    audits: list[EpisodeQuality],
    global_errors: list[str],
    thresholds: dict[str, float | None],
) -> dict[str, Any]:
    keep = [a.episode_index for a in audits if a.status == "keep"]
    review = [a.episode_index for a in audits if a.status == "review"]
    drop = [a.episode_index for a in audits if a.status == "drop"]
    return {
        "dataset": str(config.root),
        "status": "fail" if global_errors or drop else "pass_with_review" if review else "pass",
        "policy": "full_conservative_quality_audit",
        "global_errors": global_errors,
        "global_not_checked": sorted(set(runtime.global_not_checked)),
        "episodes": len(audits),
        "keep": len(keep),
        "review": len(review),
        "drop": len(drop),
        "depth_check": config.depth_check,
        "depth_sample_frames": config.depth_sample_frames,
        "depth_keys": metadata.depth_keys,
        "video_check": runtime.video_check,
        "video_content_check": runtime.video_content_check,
        "sample_frames": config.sample_frames,
        "video_sample_stride": config.video_sample_stride,
        "duplicate_check": config.duplicate_check,
        "ffmpeg": runtime.ffmpeg_path,
        "ffprobe": runtime.ffprobe_path,
        "thresholds": {
            **thresholds,
            "video_black_ratio_review": 0.95,
            "video_white_ratio_review": 0.95,
            "video_constant_ratio_review": 0.95,
            "video_freeze_ratio_review": 0.90,
            "depth_invalid_frame_ratio_review": 0.90,
            "low_motion_range_review": 1e-5,
        },
    }


def print_audit_summary(config: AuditConfig, audits: list[EpisodeQuality], summary: dict[str, Any]) -> None:
    print(f"Dataset: {config.root}")
    print(f"Output: {config.out_dir}")
    print(f"Status: {summary['status']}")
    print(f"Episodes: {len(audits)} keep={summary['keep']} review={summary['review']} drop={summary['drop']}")
    print(f"Global errors: {len(summary['global_errors'])}")
    print(f"Report: {config.out_dir / 'quality_report.csv'}")
    for error in summary["global_errors"][:20]:
        print(f"GLOBAL_ERROR: {error}")
    for audit in audits:
        if audit.drop_reasons:
            print(f"DROP episode={audit.episode_index} reasons={';'.join(audit.drop_reasons[:8])}")


def run_audit(config: AuditConfig) -> int:
    missing = [f"missing_required_path:{path}" for path in required_dataset_paths(config.root) if not path.exists()]
    if missing:
        return write_failure(config.out_dir, config.root, missing)

    try:
        metadata = load_metadata(config)
    except Exception as exc:
        return write_failure(config.out_dir, config.root, [f"cannot_read_metadata:{type(exc).__name__}:{exc}"])

    global_errors = validate_global_metadata(config.root, metadata)
    runtime = resolve_runtime(config)
    caches = AuditCaches()
    state = AuditState()

    for row in range(metadata.episodes.num_rows):
        audit_episode(config, metadata, runtime, caches, state, row)
    if state.running_global_index != metadata.total_frames:
        global_errors.append(f"dataset_global_range_end_mismatch:{state.running_global_index}!={metadata.total_frames}")

    thresholds = apply_distribution_reviews(state.audits, state)
    apply_duplicate_reviews(state, config.duplicate_check)
    summary = build_summary(config, metadata, runtime, state.audits, global_errors, thresholds)
    write_quality_outputs(config.out_dir, state.audits, summary)
    print_audit_summary(config, state.audits, summary)
    return 1 if global_errors or summary["drop"] else 0


def audit_dataset(
    root: Path,
    out_dir: Path,
    depth_check: str,
    video_check: str,
    video_content_check: str,
    sample_frames: int,
    depth_sample_frames: int,
    duplicate_check: str,
    video_sample_stride: int,
) -> int:
    """Run the quality audit; kept for compatibility with existing utility scripts."""
    config = AuditConfig(
        root=root.resolve(),
        out_dir=out_dir,
        depth_check=depth_check,
        video_check=video_check,
        video_content_check=video_content_check,
        sample_frames=sample_frames,
        depth_sample_frames=depth_sample_frames,
        duplicate_check=duplicate_check,
        video_sample_stride=video_sample_stride,
    )
    return run_audit(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full conservative quality audit on a LeRobot dataset.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--depth-check", choices=["none", "header", "full"], default="header")
    parser.add_argument("--video-check", choices=["metadata", "ffprobe", "decode"], default="decode")
    parser.add_argument("--video-content-check", choices=["none", "sample"], default="sample")
    parser.add_argument("--sample-frames", type=int, default=8)
    parser.add_argument("--video-sample-stride", type=int, default=300)
    parser.add_argument("--depth-sample-frames", type=int, default=8)
    parser.add_argument("--duplicate-check", choices=["none", "trajectory"], default="trajectory")
    args = parser.parse_args()

    out_dir = args.out_dir or default_audit_output_dir(args.dataset)
    raise SystemExit(
        audit_dataset(
            root=args.dataset,
            out_dir=out_dir,
            depth_check=args.depth_check,
            video_check=args.video_check,
            video_content_check=args.video_content_check,
            sample_frames=args.sample_frames,
            depth_sample_frames=args.depth_sample_frames,
            duplicate_check=args.duplicate_check,
            video_sample_stride=args.video_sample_stride,
        )
    )


if __name__ == "__main__":
    main()
