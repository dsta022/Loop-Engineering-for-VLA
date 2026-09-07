#!/usr/bin/env python3
"""Full conservative data quality audit for LeRobot-style RGB and RGB-D VLA datasets.

The policy is conservative:
  - objective, deterministic failures are marked drop
  - sampled content, distribution, and motion anomalies are marked review
  - semantic task-success checks are optional and run only when an evaluator
    plugin is explicitly configured
  - model-free consistency checks (progress-curve shape, stage ordering, and
    gripper actuation against the task verb) are review-only and skip themselves
    explicitly when their preconditions are absent

This script does not delete source data directly. Use the emitted keep/review/
drop lists to build a cleaned dataset after review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow `python ./pipeline/data_quality_audit.py` to run as a plain script:
# repo-root packages (utils, policy_improvement, semantic_backends) must be
# importable, including semantic evaluator plugins loaded via module:attribute.
if __package__ in (None, ""):
    _REPO_ROOT = str(Path(__file__).resolve().parents[1])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from utils.dataset_io import discover_reparse_paths, is_reparse_path, read_json, safe_rel_path
from utils.dataset_metadata import (
    episode_metadata_paths,
    feature_names_by_dtype,
    feature_shape,
    read_episode_metadata,
    video_expected_codec,
)
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
from utils.semantic_consistency import (
    GripperTrace,
    analyze_progress_curve,
    build_stage_timeline,
    channel_traces,
    dwell_outliers,
    gripper_pattern_findings,
    progress_curve_findings,
    resolve_actuation,
    resolve_motion_signature,
    select_gripper_index,
    stage_timeline_findings,
    vertical_direction_findings,
    vertical_excursions,
)
from utils.semantic_evaluator import (
    EpisodeMotion,
    SemanticEpisode,
    SemanticEvaluator,
    SemanticResult,
    VideoSegment,
    evaluator_name,
    infer_failure_stage,
    load_evaluator,
    load_evaluator_config,
    normalize_result,
)
from utils.task_semantic_profile import TaskProfileRegistry, TaskSemanticProfile
from utils.version import __version__
from policy_improvement.contracts import ControlPhase, FeedbackSidecar
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
    semantic_evaluator: SemanticEvaluator | None = None
    semantic_review_threshold: float = 0.5
    task_profiles: TaskProfileRegistry | None = None
    semantic_pass_threshold: float = 0.9
    feedback_sidecar: FeedbackSidecar | None = None
    motion_check: str = "signature"
    motion_gripper_index: int | None = None
    motion_vertical_index: int | None = None
    semantic_per_view: bool = False
    semantic_view_disagreement_threshold: float = 0.30
    semantic_uncertainty_threshold: float = 0.15
    semantic_counterfactual_margin_threshold: float = 0.10
    workers: int = 1

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if not 0.0 <= self.semantic_review_threshold <= 1.0:
            raise ValueError("semantic review threshold must be within [0, 1]")
        if not 0.0 <= self.semantic_pass_threshold <= 1.0:
            raise ValueError("semantic pass threshold must be within [0, 1]")
        if self.semantic_review_threshold > self.semantic_pass_threshold:
            raise ValueError("semantic review threshold must not exceed semantic pass threshold")
        if self.motion_check not in {"none", "signature"}:
            raise ValueError("motion check must be one of: none, signature")
        for name in ("motion_gripper_index", "motion_vertical_index"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "semantic_view_disagreement_threshold",
            "semantic_uncertainty_threshold",
            "semantic_counterfactual_margin_threshold",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")


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
    # Keyed by (path, episode segment): episodes sharing one aggregated video
    # file must each be sampled inside their own time range, not once per file.
    video_content: dict[
        tuple[Path, tuple[float, float] | None], tuple[dict[str, float] | None, str]
    ] = field(default_factory=dict)
    # Aggregated-layout data files hold many episodes; keep only the most recent
    # file in memory since episodes are audited in file order.
    episode_data: dict[Path, pa.Table] = field(default_factory=dict)


@dataclass
class AuditState:
    audits: list[EpisodeQuality] = field(default_factory=list)
    audits_by_episode: dict[int, EpisodeQuality] = field(default_factory=dict)
    action_jump_values: list[float] = field(default_factory=list)
    state_jump_values: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    trajectory_hashes: dict[str, list[int]] = field(default_factory=dict)
    running_global_index: int = 0
    channel_traces: dict[int, tuple[GripperTrace, ...]] = field(default_factory=dict)
    episode_vertical: dict[int, tuple[float, float]] = field(default_factory=dict)
    episode_tasks: dict[int, tuple[str, ...]] = field(default_factory=dict)
    stage_dwell_samples: dict[tuple[str, str], list[tuple[int, float]]] = field(
        default_factory=dict
    )

    def register(self, audit: EpisodeQuality) -> None:
        self.audits.append(audit)
        self.audits_by_episode[audit.episode_index] = audit
        self.lengths.append(audit.length)


def required_dataset_paths(root: Path) -> list[Path]:
    return [
        root / "meta" / "info.json",
        root / "meta" / "stats.json",
        root / "meta" / "tasks.parquet",
        root / "data",
        root / "videos",
    ]


def missing_required_paths(root: Path) -> list[str]:
    missing = [f"missing_required_path:{path}" for path in required_dataset_paths(root) if not path.exists()]
    if not episode_metadata_paths(root):
        missing.append(f"missing_required_path:{root / 'meta' / 'episodes' / 'chunk-*' / 'file-*.parquet'}")
    return missing


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
    episodes = read_episode_metadata(config.root, columns=build_episode_columns(video_keys))
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


def validate_feedback_global(
    config: AuditConfig, metadata: DatasetMetadata
) -> list[str]:
    sidecar = config.feedback_sidecar
    if sidecar is None:
        return []
    errors: list[str] = []
    source_dataset = sidecar.manifest.get("source_dataset")
    if source_dataset and Path(str(source_dataset)).resolve() != config.root.resolve():
        errors.append("feedback_sidecar_dataset_mismatch")
    expected = set(range(metadata.total_episodes))
    actual = {episode.episode_index for episode in sidecar.episodes}
    missing = sorted(expected.difference(actual))
    extra = sorted(actual.difference(expected))
    if missing:
        errors.append(f"feedback_sidecar_missing_episodes:{missing[:20]}")
    if extra:
        errors.append(f"feedback_sidecar_unknown_episodes:{extra[:20]}")
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


def slice_episode_rows(data: pa.Table, record: EpisodeRecord) -> pa.Table:
    """Select one episode's rows from an aggregated LeRobot v3 data file.

    Recorded datasets pack many episodes into each parquet file; only merge
    outputs keep one file per episode. When the file holds more rows than the
    episode, its rows are selected by the episode's global frame range using the
    `index` column. The slice is returned as-is: validate_index_columns still
    verifies its length and contiguity, so a corrupt index column surfaces as
    the usual structural drops instead of being silently accepted.
    """
    if data.num_rows == record.length or "index" not in data.column_names:
        return data
    try:
        mask = pc.and_(
            pc.greater_equal(data["index"], record.dataset_from),
            pc.less(data["index"], record.dataset_to),
        )
        return data.filter(mask)
    except Exception:
        return data


def read_episode_data(
    root: Path,
    record: EpisodeRecord,
    depth_keys: list[str],
    audit: EpisodeQuality,
    caches: AuditCaches | None = None,
) -> pa.Table | None:
    data_path = root / "data" / f"chunk-{record.data_chunk:03d}" / f"file-{record.data_file:03d}.parquet"
    if not data_path.exists():
        audit.drop(f"missing_data_parquet:{data_path}")
        return None
    if data_path.is_symlink() or is_reparse_path(data_path):
        audit.drop(f"data_parquet_is_link:{data_path}")
        return None
    data = caches.episode_data.get(data_path) if caches is not None else None
    if data is None:
        try:
            data = pq.read_table(data_path, columns=requested_data_columns(depth_keys))
        except Exception as exc:
            audit.drop(f"cannot_read_data_parquet:{type(exc).__name__}")
            return None
        if caches is not None:
            caches.episode_data.clear()
            caches.episode_data[data_path] = data
    data = slice_episode_rows(data, record)
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


def episode_video_reference(
    config: AuditConfig, metadata: DatasetMetadata, row: int, video_key: str
) -> tuple[Path, tuple[float, float] | None]:
    """Resolve one episode's video file and its (start, duration) segment."""
    cols = metadata.episode_cols
    chunk = int(cols[f"videos/{video_key}/chunk_index"][row])
    file_index = int(cols[f"videos/{video_key}/file_index"][row])
    video_path = config.root / "videos" / video_key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
    from_ts = float(cols[f"videos/{video_key}/from_timestamp"][row])
    to_ts = float(cols[f"videos/{video_key}/to_timestamp"][row])
    segment = (from_ts, to_ts - from_ts) if 0.0 <= from_ts < to_ts else None
    return video_path, segment


def prewarm_video_caches(
    config: AuditConfig, metadata: DatasetMetadata, runtime: AuditRuntime, caches: AuditCaches
) -> None:
    """Run ffprobe, ffmpeg decode, and content sampling for unique videos in parallel.

    The episode loop stays sequential and deterministic; it simply finds its
    results already cached. Every ffmpeg call is a subprocess, so threads are
    enough to overlap them. Verdict semantics are unchanged: a file is decoded
    only when its probe passed, and sampled only when probe and decode passed.
    """
    if config.workers <= 1 or not metadata.video_keys:
        return
    key_by_path: dict[Path, str] = {}
    segment_keys: dict[tuple[Path, tuple[float, float] | None], None] = {}
    for row in range(metadata.episodes.num_rows):
        for video_key in metadata.video_keys:
            video_path, segment = episode_video_reference(config, metadata, row, video_key)
            if not video_path.is_file():
                continue
            key_by_path.setdefault(video_path, video_key)
            segment_keys.setdefault((video_path, segment), None)

    def probe_ok(path: Path) -> bool:
        return caches.video_probe.get(path, (True, ""))[0]

    def decode_ok(path: Path) -> bool:
        return caches.video_decode.get(path, (True, ""))[0]

    def sample(key: tuple[Path, tuple[float, float] | None]) -> tuple[dict[str, float] | None, str]:
        path, segment = key
        frames, message = sample_video_file(
            path,
            runtime.ffmpeg_path or "",
            sample_frames=config.sample_frames,
            stride=config.video_sample_stride,
            start=None if segment is None else segment[0],
            duration=None if segment is None else segment[1],
        )
        return (None, message) if frames is None else (video_content_metrics(frames), "")

    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        if runtime.video_check in {"ffprobe", "decode"} and runtime.ffprobe_path is not None:
            futures = {
                path: pool.submit(
                    ffprobe_video,
                    path,
                    runtime.ffprobe_path,
                    feature_shape(metadata.info, key),
                    video_expected_codec(metadata.info, key),
                )
                for path, key in key_by_path.items()
                if path not in caches.video_probe
            }
            for path, future in futures.items():
                caches.video_probe[path] = future.result()
        if runtime.video_check == "decode" and runtime.ffmpeg_path is not None:
            futures = {
                path: pool.submit(ffmpeg_decode_video, path, runtime.ffmpeg_path)
                for path in key_by_path
                if probe_ok(path) and path not in caches.video_decode
            }
            for path, future in futures.items():
                caches.video_decode[path] = future.result()
        if runtime.video_content_check == "sample" and runtime.ffmpeg_path is not None:
            futures = {
                key: pool.submit(sample, key)
                for key in segment_keys
                if probe_ok(key[0]) and decode_ok(key[0]) and key not in caches.video_content
            }
            for key, future in futures.items():
                caches.video_content[key] = future.result()


def audit_video_files(
    config: AuditConfig,
    metadata: DatasetMetadata,
    runtime: AuditRuntime,
    caches: AuditCaches,
    record: EpisodeRecord,
    audit: EpisodeQuality,
) -> None:
    for video_key in metadata.video_keys:
        video_path, segment = episode_video_reference(config, metadata, record.row, video_key)
        if not validate_video_path(video_key, video_path, audit):
            continue
        if not probe_video_if_requested(video_key, video_path, metadata, runtime, caches, audit):
            continue
        if not decode_video_if_requested(video_key, video_path, runtime, caches, audit):
            continue
        review_video_content_if_requested(
            video_key, video_path, segment, config, runtime, caches, audit
        )


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
    segment: tuple[float, float] | None,
    config: AuditConfig,
    runtime: AuditRuntime,
    caches: AuditCaches,
    audit: EpisodeQuality,
) -> None:
    if runtime.video_content_check == "sample" and runtime.ffmpeg_path is not None:
        cache_key = (video_path, segment)
        if cache_key not in caches.video_content:
            frames, message = sample_video_file(
                video_path,
                runtime.ffmpeg_path,
                sample_frames=config.sample_frames,
                stride=config.video_sample_stride,
                start=None if segment is None else segment[0],
                duration=None if segment is None else segment[1],
            )
            caches.video_content[cache_key] = (
                (None, message) if frames is None else (video_content_metrics(frames), "")
            )
        metrics, message = caches.video_content[cache_key]
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


TASK_TEXT_COLUMNS = ("task", "__index_level_0__")


def task_text_column(tasks: pa.Table) -> str | None:
    """Locate the task-text column in tasks.parquet.

    LeRobot writes the task string either as an explicit `task` column or, when
    the table was saved from a pandas frame indexed by task text, as the pandas
    index artifact `__index_level_0__`. Both forms appear in real datasets, so
    both are accepted, with the explicit column preferred.
    """
    for name in TASK_TEXT_COLUMNS:
        if name in tasks.column_names:
            return name
    return None


def semantic_tasks_for_episode(metadata: DatasetMetadata, data: pa.Table | None) -> tuple[str, ...]:
    if data is None or "task_index" not in data.column_names:
        return ()
    text_column = task_text_column(metadata.tasks)
    if "task_index" not in metadata.tasks.column_names or text_column is None:
        return ()
    task_map = {
        int(index.as_py()): str(text.as_py())
        for index, text in zip(metadata.tasks["task_index"], metadata.tasks[text_column])
    }
    ordered: list[str] = []
    for value in data["task_index"]:
        text = task_map.get(int(value.as_py()))
        if text is not None and text not in ordered:
            ordered.append(text)
    return tuple(ordered)


def semantic_videos_for_episode(
    config: AuditConfig,
    metadata: DatasetMetadata,
    record: EpisodeRecord,
    profile: TaskSemanticProfile | None = None,
) -> tuple[VideoSegment, ...]:
    videos: list[VideoSegment] = []
    preferred_keys = set(profile.preferred_video_keys) if profile is not None else set()
    for video_key in metadata.video_keys:
        if preferred_keys and video_key not in preferred_keys:
            continue
        chunk = int(metadata.episode_cols[f"videos/{video_key}/chunk_index"][record.row])
        file_index = int(metadata.episode_cols[f"videos/{video_key}/file_index"][record.row])
        from_timestamp = float(metadata.episode_cols[f"videos/{video_key}/from_timestamp"][record.row])
        to_timestamp = float(metadata.episode_cols[f"videos/{video_key}/to_timestamp"][record.row])
        videos.append(
            VideoSegment(
                key=video_key,
                path=config.root / "videos" / video_key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4",
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp,
            )
        )
    return tuple(videos)


def apply_progress_shape_checks(
    result_curve: list[float], audit: EpisodeQuality
) -> None:
    """Read the shape of the progress curve, not only its peak and final value."""
    metrics = analyze_progress_curve(result_curve)
    if metrics is None:
        audit.skip("semantic_progress_shape_not_checked")
        return
    audit.semantic_progress_max_drop = metrics.max_drop
    audit.semantic_progress_idle_tail_ratio = metrics.idle_tail_ratio
    audit.metrics["semantic_progress_metrics"] = metrics.to_dict()
    for reason in progress_curve_findings(metrics):
        audit.review(reason)


def apply_stage_timeline_checks(
    profile: TaskSemanticProfile | None,
    result_curve: list[float],
    details: dict[str, Any],
    audit: EpisodeQuality,
    audit_state: AuditState | None,
) -> None:
    """Assign progress points to profile stages and check their ordering."""
    if profile is None:
        audit.skip("semantic_stage_timeline_requires_task_profile")
        return
    timeline = build_stage_timeline(profile, result_curve, details)
    if timeline is None:
        audit.skip("semantic_stage_timeline_not_checked")
        return
    audit.semantic_stage_timeline_source = timeline.source
    audit.metrics["semantic_stage_timeline"] = timeline.to_dict()
    for reason in stage_timeline_findings(timeline):
        audit.review(reason)
    if audit_state is not None:
        for stage, ratio in timeline.dwell_ratios.items():
            audit_state.stage_dwell_samples.setdefault(
                (profile.task_key, stage), []
            ).append((audit.episode_index, ratio))


def apply_score_reliability_checks(
    config: AuditConfig, result: SemanticResult, audit: EpisodeQuality
) -> None:
    """Judge how much the score itself can be trusted, not only its value.

    Both signals are contracts a backend may or may not fill: the dispersion of
    repeated scorings, and the margin over deliberately corrupted instructions.
    A high score that is unstable across samples, or that a distractor
    instruction nearly ties, is demoted to review.
    """
    uncertainty = result.uncertainty
    if uncertainty is None:
        audit.skip("semantic_score_stability_not_checked")
    else:
        audit.semantic_score_uncertainty = uncertainty
        audit.metrics["semantic_score_uncertainty"] = uncertainty
        if result.score_samples:
            audit.metrics["semantic_score_samples"] = list(result.score_samples)
        if uncertainty >= config.semantic_uncertainty_threshold:
            audit.semantic_status = "review"
            audit.review("semantic_score_unstable_review")

    margin = result.counterfactual_margin
    if margin is None:
        audit.skip("semantic_counterfactual_not_checked")
    else:
        audit.semantic_counterfactual_margin = margin
        if margin < config.semantic_counterfactual_margin_threshold:
            audit.semantic_status = "review"
            audit.review("semantic_counterfactual_margin_low_review")


def apply_per_view_evaluation(
    config: AuditConfig,
    evaluator: SemanticEvaluator,
    episode: SemanticEpisode,
    audit: EpisodeQuality,
) -> None:
    """Score each camera view separately and compare the verdicts.

    A large gap between views means occlusion or hallucination: one camera saw
    something the other did not, so neither single score should be trusted. The
    combined result stays authoritative; this is a diagnostic that costs one
    extra evaluator call per view.
    """
    if not config.semantic_per_view:
        return
    if len(episode.videos) < 2:
        audit.skip("semantic_view_disagreement_single_view")
        return
    view_scores: dict[str, float] = {}
    for video in episode.videos:
        try:
            view_result = normalize_result(evaluator.evaluate(replace(episode, videos=(video,))))
            view_scores[video.key] = view_result.score
        except Exception as exc:
            audit.metrics["semantic_view_error"] = (
                f"{video.key}:{type(exc).__name__}:{str(exc)[:200]}"
            )
            audit.skip("semantic_view_disagreement_error")
            return
    disagreement = max(view_scores.values()) - min(view_scores.values())
    audit.semantic_view_disagreement = disagreement
    audit.metrics["semantic_view_scores"] = view_scores
    if disagreement >= config.semantic_view_disagreement_threshold:
        audit.semantic_status = "review"
        audit.review("semantic_view_disagreement_review")


def apply_semantic_evaluation(
    config: AuditConfig,
    metadata: DatasetMetadata,
    record: EpisodeRecord,
    data: pa.Table | None,
    audit: EpisodeQuality,
    audit_state: AuditState | None = None,
) -> None:
    evaluator = config.semantic_evaluator
    audit.structural_status = audit.status
    if evaluator is None:
        audit.skip("semantic_task_success_not_checked")
        return

    audit.semantic_evaluator = evaluator_name(evaluator)
    if audit.status == "drop":
        audit.semantic_status = "skipped_structural_drop"
        audit.skip("semantic_skipped_structural_drop")
        return

    tasks = semantic_tasks_for_episode(metadata, data)
    if not tasks:
        audit.semantic_status = "error"
        audit.review("semantic_task_text_missing_review")
        return
    if len(tasks) > 1:
        audit.semantic_status = "error"
        audit.metrics["semantic_episode_tasks"] = list(tasks)
        audit.review("semantic_multi_task_episode_requires_segmentation_review")
        return

    profile: TaskSemanticProfile | None = None
    if config.task_profiles is not None:
        try:
            profile = config.task_profiles.match_any(tasks)
        except Exception as exc:
            audit.semantic_status = "error"
            audit.metrics["semantic_profile_error"] = f"{type(exc).__name__}:{str(exc)[:500]}"
            audit.review("semantic_task_profile_ambiguous_review")
            return
        if profile is None:
            audit.semantic_status = "error"
            audit.review("semantic_task_profile_missing_review")
            return
        audit.semantic_task_profile = profile.task_key

    review_threshold = (
        profile.effective_review_threshold(config.semantic_review_threshold)
        if profile is not None
        else config.semantic_review_threshold
    )
    pass_threshold = (
        profile.effective_pass_threshold(config.semantic_pass_threshold)
        if profile is not None
        else config.semantic_pass_threshold
    )
    if review_threshold > pass_threshold:
        audit.semantic_status = "error"
        audit.metrics["semantic_threshold_error"] = (
            f"review_threshold={review_threshold}>pass_threshold={pass_threshold}"
        )
        audit.review("semantic_threshold_configuration_invalid_review")
        return
    audit.semantic_review_threshold = review_threshold
    audit.semantic_pass_threshold = pass_threshold
    videos = semantic_videos_for_episode(config, metadata, record, profile)
    if profile is not None and profile.preferred_video_keys:
        available_keys = {video.key for video in videos}
        missing_keys = sorted(set(profile.preferred_video_keys).difference(available_keys))
        if missing_keys:
            audit.semantic_status = "error"
            audit.metrics["semantic_missing_preferred_video_keys"] = missing_keys
            audit.review("semantic_preferred_video_missing_review")
            return

    episode = SemanticEpisode(
        dataset_root=config.root,
        episode_index=record.episode_index,
        length=record.length,
        tasks=tasks,
        videos=videos,
        task_profile=profile.to_dict() if profile is not None else None,
        motion=episode_motion(config, metadata, data),
    )
    try:
        result = normalize_result(evaluator.evaluate(episode))
    except Exception as exc:
        audit.semantic_status = "error"
        audit.metrics["semantic_error"] = f"{type(exc).__name__}:{str(exc)[:500]}"
        audit.review(f"semantic_evaluation_failed_review:{type(exc).__name__}")
        return

    audit.semantic_progress_curve = list(result.progress_curve)
    audit.semantic_progress_final = result.final_progress
    audit.semantic_success_probability = result.success_probability
    audit.semantic_score = result.score
    if result.details:
        audit.metrics["semantic_details"] = result.details
    if result.progress_curve:
        audit.metrics["semantic_progress_peak"] = max(result.progress_curve)
        audit.metrics["semantic_progress_points"] = len(result.progress_curve)

    apply_progress_shape_checks(audit.semantic_progress_curve, audit)
    apply_stage_timeline_checks(
        profile, audit.semantic_progress_curve, dict(result.details), audit, audit_state
    )

    if result.score < review_threshold:
        audit.semantic_status = "review"
        audit.semantic_failure_stage = (
            result.failure_stage
            or (profile.infer_failure_stage(result.progress_curve) if profile is not None else None)
            or infer_failure_stage(result.progress_curve)
        )
        if result.progress_curve and max(result.progress_curve) - result.progress_curve[-1] >= 0.15:
            audit.metrics["semantic_failure_type"] = "regressed_after_progress"
        audit.review("semantic_low_score_review")
    elif result.score < pass_threshold:
        audit.semantic_status = "review"
        audit.semantic_failure_stage = result.failure_stage or ""
        audit.review("semantic_uncertain_score_review")
    else:
        audit.semantic_status = "pass"
        audit.semantic_failure_stage = result.failure_stage or ""

    apply_score_reliability_checks(config, result, audit)
    apply_per_view_evaluation(config, evaluator, episode, audit)


def episode_motion(
    config: AuditConfig, metadata: DatasetMetadata, data: pa.Table | None
) -> EpisodeMotion:
    """Expose the action/state streams so consistency checks can read them."""
    action = column_matrix(data, "action", np.float64) if data is not None else None
    state_values = (
        column_matrix(data, "observation.state", np.float64) if data is not None else None
    )
    fps = metadata.info.get("fps")
    return EpisodeMotion(
        action=action if action is not None and action.ndim == 2 else None,
        state=state_values if state_values is not None and state_values.ndim == 2 else None,
        fps=float(fps) if isinstance(fps, (int, float)) else None,
        gripper_index=config.motion_gripper_index,
    )


def collect_motion_signals(
    config: AuditConfig,
    metadata: DatasetMetadata,
    record: EpisodeRecord,
    data: pa.Table | None,
    audit: EpisodeQuality,
    audit_state: AuditState,
) -> None:
    """Record per-episode motion traces for the dataset-level signature check.

    Findings are deferred to the distribution pass because deciding which action
    dimension is the gripper, and whether it actuated at all, needs the whole
    dataset rather than one episode.
    """
    if config.motion_check == "none":
        audit.skip("semantic_motion_check_disabled")
        return
    audit_state.episode_tasks[record.episode_index] = semantic_tasks_for_episode(metadata, data)
    motion = episode_motion(config, metadata, data)
    if motion.action is None:
        audit.skip("semantic_motion_action_unavailable")
        return
    audit_state.channel_traces[record.episode_index] = channel_traces(motion.action)
    if config.motion_vertical_index is not None and (
        0 <= config.motion_vertical_index < motion.action.shape[1]
    ):
        audit_state.episode_vertical[record.episode_index] = vertical_excursions(
            motion.action[:, config.motion_vertical_index]
        )


def apply_motion_reviews(config: AuditConfig, state: AuditState) -> dict[str, Any]:
    """Compare each episode's gripper actuation against its instruction verb."""
    summary: dict[str, Any] = {
        "enabled": config.motion_check != "none",
        "gripper_index": None,
        "gripper_index_source": "unresolved",
        "vertical_index": config.motion_vertical_index,
        "episodes_checked": 0,
        "verb_unmatched": 0,
        "signature_conflicts": 0,
    }
    if config.motion_check == "none" or not state.channel_traces:
        return summary

    gripper_index = config.motion_gripper_index
    if gripper_index is not None:
        summary["gripper_index_source"] = "configured"
    else:
        gripper_index = select_gripper_index(state.channel_traces)
        if gripper_index is not None:
            summary["gripper_index_source"] = "detected"
    summary["gripper_index"] = gripper_index
    if gripper_index is None:
        for audit in state.audits:
            audit.skip("semantic_motion_gripper_channel_unresolved")
        return summary

    spans = [
        trace.span
        for traces in state.channel_traces.values()
        for trace in traces
        if trace.index == gripper_index
    ]
    median_span = float(np.median(spans)) if spans else 0.0

    for audit in state.audits:
        episode_index = audit.episode_index
        traces = state.channel_traces.get(episode_index)
        if traces is None:
            continue
        tasks = state.episode_tasks.get(episode_index, ())
        signature = resolve_motion_signature(tasks)
        if signature is None:
            audit.skip("semantic_motion_verb_unmatched")
            summary["verb_unmatched"] += 1
            continue
        audit.semantic_motion_verbs = ";".join(signature.names)
        if signature.conflicted:
            audit.skip("semantic_motion_signature_conflict")
            audit.metrics["semantic_motion"] = {"signature": signature.to_dict()}
            summary["signature_conflicts"] += 1
            continue
        trace = resolve_actuation(traces, gripper_index, median_span)
        if trace is None:
            audit.skip("semantic_motion_gripper_channel_unresolved")
            continue

        summary["episodes_checked"] += 1
        audit.semantic_motion_gripper_transitions = trace.transitions
        motion_metrics: dict[str, Any] = {
            "signature": signature.to_dict(),
            "gripper": trace.to_dict(),
            "gripper_index_source": summary["gripper_index_source"],
        }
        pattern_reasons, pattern_metrics = gripper_pattern_findings(signature, trace)
        motion_metrics.update(pattern_metrics)
        excursions = state.episode_vertical.get(episode_index)
        vertical_reasons, vertical_metrics = vertical_direction_findings(
            signature,
            excursions,
            config.motion_vertical_index if excursions is not None else -1,
        )
        if vertical_metrics:
            motion_metrics["vertical"] = vertical_metrics
        elif signature.vertical_direction:
            audit.skip("semantic_motion_vertical_axis_not_configured")
        audit.metrics["semantic_motion"] = motion_metrics
        for reason in (*pattern_reasons, *vertical_reasons):
            audit.review(reason)
    return summary


def apply_stage_dwell_reviews(state: AuditState) -> int:
    """Flag stage durations that are far from their task/stage median."""
    flagged = dwell_outliers(state.stage_dwell_samples)
    for episode_index, deviations in flagged.items():
        audit = state.audits_by_episode.get(episode_index)
        if audit is None:
            continue
        audit.metrics["semantic_stage_dwell_outliers"] = deviations
        audit.review("semantic_stage_dwell_outlier_review")
    return len(flagged)


def apply_feedback_validation(
    config: AuditConfig, record: EpisodeRecord, audit: EpisodeQuality
) -> None:
    sidecar = config.feedback_sidecar
    if sidecar is None:
        return
    alignment_errors = sidecar.validate_episode_alignment(record.episode_index, record.length)
    for error in alignment_errors:
        audit.drop(error)
    episode_map = {episode.episode_index: episode for episode in sidecar.episodes}
    episode = episode_map.get(record.episode_index)
    if episode is None:
        return
    frames = sidecar.frames_by_episode().get(record.episode_index, ())
    segments = sidecar.interventions_by_episode().get(record.episode_index, ())
    phase_counts: dict[str, int] = {}
    for frame in frames:
        phase = ControlPhase(frame.phase).value
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    capture_mode = str(
        episode.capture_mode.value
        if hasattr(episode.capture_mode, "value")
        else episode.capture_mode
    )
    review_status = str(
        episode.review_status.value
        if hasattr(episode.review_status, "value")
        else episode.review_status
    )
    audit.metrics.update(
        {
            "feedback_capture_mode": capture_mode,
            "feedback_source_policy": episode.source_policy,
            "feedback_outcome": episode.outcome,
            "feedback_review_status": review_status,
            "feedback_intervention_count": len(segments),
            "feedback_training_eligible_frames": sum(frame.training_eligible for frame in frames),
            "feedback_phase_counts": phase_counts,
        }
    )
    if episode.outcome in {"aborted", "pending_review", "failure"}:
        audit.review("feedback_outcome_requires_review")
    if review_status == "pending":
        audit.review("feedback_human_approval_required")
    elif review_status == "rejected":
        audit.review("feedback_human_rejected")
    if capture_mode == "corrections_only" and not any(
        ControlPhase(frame.phase) == ControlPhase.CORRECTION for frame in frames
    ):
        audit.review("feedback_correction_missing_review")


def refresh_structural_status(audits: list[EpisodeQuality]) -> None:
    for audit in audits:
        if audit.drop_reasons:
            audit.structural_status = "drop"
        elif any(not reason.startswith("semantic_") for reason in audit.review_reasons):
            audit.structural_status = "review"
        else:
            audit.structural_status = "keep"


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
    data = read_episode_data(config.root, record, metadata.depth_keys, audit, caches)
    if data is not None:
        audit_numeric_data(data, metadata, record, audit, audit_state)
    audit_depth_sidecars(config, metadata, record, data, audit)
    audit_video_files(config, metadata, runtime, caches, record, audit)
    apply_feedback_validation(config, record, audit)
    collect_motion_signals(config, metadata, record, data, audit, audit_state)
    apply_semantic_evaluation(config, metadata, record, data, audit, audit_state)


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
    fingerprint: str | None = None,
    motion_summary: dict[str, Any] | None = None,
    stage_dwell_outliers: int = 0,
) -> dict[str, Any]:
    keep = [a.episode_index for a in audits if a.status == "keep"]
    review = [a.episode_index for a in audits if a.status == "review"]
    drop = [a.episode_index for a in audits if a.status == "drop"]
    return {
        "dataset": str(config.root),
        "dataset_fingerprint": fingerprint,
        "audit_script_version": __version__,
        "audit_time_utc": datetime.now(timezone.utc).isoformat(),
        "workers": config.workers,
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
        "semantic_evaluation": {
            "enabled": config.semantic_evaluator is not None,
            "evaluator": (
                evaluator_name(config.semantic_evaluator) if config.semantic_evaluator is not None else None
            ),
            "review_threshold": config.semantic_review_threshold,
            "pass_threshold": config.semantic_pass_threshold,
            "per_view": config.semantic_per_view,
            "view_disagreement_threshold": config.semantic_view_disagreement_threshold,
            "uncertainty_threshold": config.semantic_uncertainty_threshold,
            "counterfactual_margin_threshold": config.semantic_counterfactual_margin_threshold,
            "task_profiles_enabled": config.task_profiles is not None,
            "task_profile_count": len(config.task_profiles.profiles) if config.task_profiles else 0,
            "pass": sum(a.semantic_status == "pass" for a in audits),
            "review": sum(a.semantic_status == "review" for a in audits),
            "low_score_review": sum(
                "semantic_low_score_review" in a.review_reasons for a in audits
            ),
            "uncertain_score_review": sum(
                "semantic_uncertain_score_review" in a.review_reasons for a in audits
            ),
            "errors": sum(a.semantic_status == "error" for a in audits),
            "skipped": sum(a.semantic_status.startswith("skipped") for a in audits),
        },
        "semantic_consistency": {
            "progress_regression_review": sum(
                "semantic_progress_regression_review" in a.review_reasons for a in audits
            ),
            "progress_idle_tail_review": sum(
                "semantic_progress_idle_tail_review" in a.review_reasons for a in audits
            ),
            "progress_idle_head_review": sum(
                "semantic_progress_idle_head_review" in a.review_reasons for a in audits
            ),
            "stage_order_violation_review": sum(
                "semantic_stage_order_violation_review" in a.review_reasons for a in audits
            ),
            "stage_skipped_review": sum(
                "semantic_stage_skipped_review" in a.review_reasons for a in audits
            ),
            "stage_unknown_review": sum(
                "semantic_stage_unknown_review" in a.review_reasons for a in audits
            ),
            "stage_dwell_outlier_review": stage_dwell_outliers,
            "motion_gripper_pattern_review": sum(
                "semantic_motion_gripper_pattern_review" in a.review_reasons for a in audits
            ),
            "motion_vertical_direction_review": sum(
                "semantic_motion_vertical_direction_review" in a.review_reasons for a in audits
            ),
            "score_unstable_review": sum(
                "semantic_score_unstable_review" in a.review_reasons for a in audits
            ),
            "view_disagreement_review": sum(
                "semantic_view_disagreement_review" in a.review_reasons for a in audits
            ),
            "counterfactual_margin_low_review": sum(
                "semantic_counterfactual_margin_low_review" in a.review_reasons for a in audits
            ),
            "motion": motion_summary
            or {"enabled": config.motion_check != "none", "episodes_checked": 0},
        },
        "feedback_validation": {
            "enabled": config.feedback_sidecar is not None,
            "sidecar": str(config.feedback_sidecar.root) if config.feedback_sidecar else None,
            "episodes_with_intervention": sum(
                int(a.metrics.get("feedback_intervention_count", 0)) > 0 for a in audits
            ),
            "training_eligible_frames": sum(
                int(a.metrics.get("feedback_training_eligible_frames", 0)) for a in audits
            ),
        },
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
    missing = missing_required_paths(config.root)
    if missing:
        return write_failure(config.out_dir, config.root, missing)

    try:
        metadata = load_metadata(config)
    except Exception as exc:
        return write_failure(config.out_dir, config.root, [f"cannot_read_metadata:{type(exc).__name__}:{exc}"])

    global_errors = validate_global_metadata(config.root, metadata)
    global_errors.extend(validate_feedback_global(config, metadata))
    runtime = resolve_runtime(config)
    caches = AuditCaches()
    prewarm_video_caches(config, metadata, runtime, caches)
    state = AuditState()

    for row in range(metadata.episodes.num_rows):
        audit_episode(config, metadata, runtime, caches, state, row)
    if state.running_global_index != metadata.total_frames:
        global_errors.append(f"dataset_global_range_end_mismatch:{state.running_global_index}!={metadata.total_frames}")

    thresholds = apply_distribution_reviews(state.audits, state)
    apply_duplicate_reviews(state, config.duplicate_check)
    motion_summary = apply_motion_reviews(config, state)
    stage_dwell_outliers = apply_stage_dwell_reviews(state)
    refresh_structural_status(state.audits)
    summary = build_summary(
        config,
        metadata,
        runtime,
        state.audits,
        global_errors,
        thresholds,
        fingerprint=dataset_fingerprint(config.root),
        motion_summary=motion_summary,
        stage_dwell_outliers=stage_dwell_outliers,
    )
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
    semantic_evaluator: SemanticEvaluator | None = None,
    semantic_review_threshold: float = 0.5,
    task_profiles: TaskProfileRegistry | None = None,
    semantic_pass_threshold: float = 0.9,
    feedback_sidecar: FeedbackSidecar | Path | None = None,
    motion_check: str = "signature",
    motion_gripper_index: int | None = None,
    motion_vertical_index: int | None = None,
    semantic_per_view: bool = False,
    semantic_view_disagreement_threshold: float = 0.30,
    semantic_uncertainty_threshold: float = 0.15,
    semantic_counterfactual_margin_threshold: float = 0.10,
    workers: int = 1,
) -> int:
    """Run the quality audit; kept for compatibility with existing utility scripts."""
    if not 0.0 <= semantic_review_threshold <= 1.0:
        raise ValueError("semantic review threshold must be within [0, 1]")
    if not 0.0 <= semantic_pass_threshold <= 1.0:
        raise ValueError("semantic pass threshold must be within [0, 1]")
    if semantic_review_threshold > semantic_pass_threshold:
        raise ValueError("semantic review threshold must not exceed semantic pass threshold")
    resolved_feedback = (
        FeedbackSidecar.load(feedback_sidecar)
        if isinstance(feedback_sidecar, Path)
        else feedback_sidecar
    )
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
        semantic_evaluator=semantic_evaluator,
        semantic_review_threshold=semantic_review_threshold,
        semantic_pass_threshold=semantic_pass_threshold,
        task_profiles=task_profiles,
        feedback_sidecar=resolved_feedback,
        motion_check=motion_check,
        motion_gripper_index=motion_gripper_index,
        motion_vertical_index=motion_vertical_index,
        semantic_per_view=semantic_per_view,
        semantic_view_disagreement_threshold=semantic_view_disagreement_threshold,
        semantic_uncertainty_threshold=semantic_uncertainty_threshold,
        semantic_counterfactual_margin_threshold=semantic_counterfactual_margin_threshold,
        workers=workers,
    )
    return run_audit(config)


SEMANTIC_REASON_PREFIX = "semantic_"


def dataset_fingerprint(root: Path) -> str:
    """Hash audit-relevant file identities without loading large media payloads.

    Metadata files are content-hashed. Dataset payloads use relative path, size,
    and modification time so a replaced parquet/video/depth file invalidates a
    reused structural report at low cost.
    """
    root = root.resolve()
    digest = hashlib.sha256()
    metadata_files = [
        root / "meta" / "info.json",
        root / "meta" / "stats.json",
        root / "meta" / "tasks.parquet",
        *episode_metadata_paths(root),
    ]
    for path in metadata_files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
    for directory_name in ("data", "videos", "depth_sidecar"):
        directory = root / directory_name
        if not directory.exists():
            continue
        for path in sorted(
            (item for item in directory.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(root).as_posix(),
        ):
            stat = path.stat()
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def load_prior_audits(report_path: Path) -> dict[int, EpisodeQuality]:
    """Rebuild structural findings from a prior quality report.

    Semantic reasons and metrics are stripped so a semantic-only rerun replaces
    the prior semantic verdicts instead of accumulating them.
    """
    audits: dict[int, EpisodeQuality] = {}
    with report_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episode_text = (row.get("episode_index") or "").strip()
            if not episode_text:
                continue
            audit = EpisodeQuality(episode_index=int(episode_text))
            audit.length = int(row.get("length") or 0)
            audit.data_rows = int(row.get("data_rows") or 0)
            audit.depth_files = int(row.get("depth_files") or 0)
            audit.drop_reasons = [
                reason for reason in (row.get("drop_reasons") or "").split(";") if reason
            ]
            audit.review_reasons = [
                reason
                for reason in (row.get("review_reasons") or "").split(";")
                if reason and not reason.startswith(SEMANTIC_REASON_PREFIX)
            ]
            audit.not_checked = [
                reason
                for reason in (row.get("not_checked") or "").split(";")
                if reason and not reason.startswith(SEMANTIC_REASON_PREFIX)
            ]
            try:
                metrics = json.loads(row.get("metrics_json") or "{}")
            except json.JSONDecodeError:
                metrics = {}
            if isinstance(metrics, dict):
                audit.metrics = {
                    key: value
                    for key, value in metrics.items()
                    if not key.startswith(SEMANTIC_REASON_PREFIX)
                }
            audit.status = (
                "drop"
                if audit.drop_reasons
                else "review"
                if audit.review_reasons
                else "keep"
            )
            audit.structural_status = audit.status
            audits[audit.episode_index] = audit
    return audits


def read_semantic_columns(root: Path, record: EpisodeRecord) -> pa.Table | None:
    """Read only the columns semantic and motion checks need.

    Action and state are included so a semantic-only rerun still performs the
    action/language consistency checks; media payloads are never touched.
    """
    data_path = (
        root / "data" / f"chunk-{record.data_chunk:03d}" / f"file-{record.data_file:03d}.parquet"
    )
    wanted = ["task_index", "action", "observation.state", "index"]
    try:
        available = pq.read_schema(data_path).names
    except Exception:
        return None
    columns = [name for name in wanted if name in available]
    if not columns:
        return None
    try:
        return slice_episode_rows(pq.read_table(data_path, columns=columns), record)
    except Exception:
        return None


def run_semantic_reuse(config: AuditConfig, report_path: Path) -> int:
    """Reuse a fingerprint-matched structural report and rerun only semantics."""
    if config.semantic_evaluator is None:
        raise ValueError("semantic reuse mode requires a configured semantic evaluator")
    missing = missing_required_paths(config.root)
    if missing:
        return write_failure(config.out_dir, config.root, missing)
    if not report_path.exists():
        return write_failure(config.out_dir, config.root, [f"missing_prior_report:{report_path}"])
    try:
        metadata = load_metadata(config)
    except Exception as exc:
        return write_failure(
            config.out_dir, config.root, [f"cannot_read_metadata:{type(exc).__name__}:{exc}"]
        )
    global_errors = validate_global_metadata(config.root, metadata)
    current_fingerprint = dataset_fingerprint(config.root)
    prior_thresholds: dict[str, Any] = {}
    prior_summary_path = report_path.parent / "quality_summary.json"
    if not prior_summary_path.exists():
        return write_failure(
            config.out_dir, config.root, [f"prior_report_summary_missing:{prior_summary_path}"]
        )
    try:
        prior_summary = json.loads(prior_summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return write_failure(
            config.out_dir, config.root, [f"cannot_read_prior_summary:{type(exc).__name__}:{exc}"]
        )
    prior_dataset = prior_summary.get("dataset")
    if prior_dataset and Path(str(prior_dataset)).resolve() != config.root.resolve():
        return write_failure(
            config.out_dir, config.root, [f"prior_report_dataset_mismatch:{prior_dataset}"]
        )
    prior_fingerprint = str(prior_summary.get("dataset_fingerprint") or "")
    if not prior_fingerprint:
        return write_failure(config.out_dir, config.root, ["prior_report_fingerprint_missing"])
    if prior_fingerprint != current_fingerprint:
        return write_failure(
            config.out_dir,
            config.root,
            [f"prior_report_fingerprint_mismatch:{prior_fingerprint}!={current_fingerprint}"],
        )
    if isinstance(prior_summary.get("thresholds"), dict):
        prior_thresholds = prior_summary["thresholds"]
    try:
        prior_audits = load_prior_audits(report_path)
    except Exception as exc:
        return write_failure(
            config.out_dir, config.root, [f"cannot_read_prior_report:{type(exc).__name__}:{exc}"]
        )
    expected_episodes = {int(value) for value in metadata.episode_cols["episode_index"]}
    if set(prior_audits) != expected_episodes:
        missing_episodes = sorted(expected_episodes.difference(prior_audits))
        extra_episodes = sorted(set(prior_audits).difference(expected_episodes))
        return write_failure(
            config.out_dir,
            config.root,
            [
                "prior_report_episode_mismatch:"
                f"missing={missing_episodes[:20]}:extra={extra_episodes[:20]}"
            ],
        )
    state = AuditState()
    for row in range(metadata.episodes.num_rows):
        record = EpisodeRecord.from_columns(row, metadata.episode_cols)
        audit = prior_audits[record.episode_index]
        if audit.length != record.length:
            return write_failure(
                config.out_dir,
                config.root,
                [
                    "prior_report_length_mismatch:"
                    f"episode={record.episode_index}:{audit.length}!={record.length}"
                ],
            )
        state.register(audit)
        audit.skip("structural_checks_reused_from_prior_report")
        data = read_semantic_columns(config.root, record)
        collect_motion_signals(config, metadata, record, data, audit, state)
        apply_semantic_evaluation(config, metadata, record, data, audit, state)
    motion_summary = apply_motion_reviews(config, state)
    stage_dwell_outliers = apply_stage_dwell_reviews(state)
    refresh_structural_status(state.audits)
    runtime = AuditRuntime(
        video_check="reused_prior_report",
        video_content_check="reused_prior_report",
        ffprobe_path=None,
        ffmpeg_path=None,
    )
    summary = build_summary(
        config,
        metadata,
        runtime,
        state.audits,
        global_errors,
        {},
        fingerprint=current_fingerprint,
        motion_summary=motion_summary,
        stage_dwell_outliers=stage_dwell_outliers,
    )
    summary["audit_mode"] = "semantic_reuse"
    summary["reused_report"] = str(report_path)
    summary["depth_check"] = "reused_prior_report"
    if prior_thresholds:
        summary["thresholds"] = prior_thresholds
    write_quality_outputs(config.out_dir, state.audits, summary)
    print_audit_summary(config, state.audits, summary)
    return 1 if global_errors or summary["drop"] else 0


def audit_dataset_semantic_reuse(
    root: Path,
    out_dir: Path,
    semantic_evaluator: SemanticEvaluator,
    reuse_report: Path | None = None,
    semantic_review_threshold: float = 0.5,
    semantic_pass_threshold: float = 0.9,
    task_profiles: TaskProfileRegistry | None = None,
    motion_check: str = "signature",
    motion_gripper_index: int | None = None,
    motion_vertical_index: int | None = None,
    semantic_per_view: bool = False,
    semantic_view_disagreement_threshold: float = 0.30,
    semantic_uncertainty_threshold: float = 0.15,
    semantic_counterfactual_margin_threshold: float = 0.10,
) -> int:
    """Rerun only semantic evaluation against an existing structural report."""
    config = AuditConfig(
        root=root.resolve(),
        out_dir=out_dir,
        depth_check="none",
        video_check="metadata",
        video_content_check="none",
        sample_frames=0,
        depth_sample_frames=0,
        duplicate_check="none",
        video_sample_stride=1,
        semantic_evaluator=semantic_evaluator,
        semantic_review_threshold=semantic_review_threshold,
        semantic_pass_threshold=semantic_pass_threshold,
        task_profiles=task_profiles,
        motion_check=motion_check,
        motion_gripper_index=motion_gripper_index,
        motion_vertical_index=motion_vertical_index,
        semantic_per_view=semantic_per_view,
        semantic_view_disagreement_threshold=semantic_view_disagreement_threshold,
        semantic_uncertainty_threshold=semantic_uncertainty_threshold,
        semantic_counterfactual_margin_threshold=semantic_counterfactual_margin_threshold,
    )
    report_path = reuse_report or (out_dir / "quality_report.csv")
    return run_semantic_reuse(config, report_path)


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
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="parallel ffprobe/ffmpeg calls for unique video files before the episode loop",
    )
    parser.add_argument(
        "--motion-check",
        choices=["none", "signature"],
        default="signature",
        help=(
            "compare gripper actuation against the verb in the task text; "
            "needs no model and skips itself when the verb is unknown"
        ),
    )
    parser.add_argument(
        "--motion-gripper-index",
        type=int,
        default=None,
        help=(
            "action dimension holding the gripper; "
            "auto-detected from bimodal channel behaviour when omitted"
        ),
    )
    parser.add_argument(
        "--motion-vertical-index",
        type=int,
        default=None,
        help=(
            "action dimension that moves vertically; enables direction checks "
            "for verbs such as lift and press, which are skipped when omitted"
        ),
    )
    parser.add_argument(
        "--semantic-evaluator",
        default=None,
        help="optional evaluator plugin using module:attribute syntax",
    )
    parser.add_argument(
        "--semantic-config",
        type=Path,
        default=None,
        help="optional JSON configuration passed to the evaluator plugin",
    )
    parser.add_argument(
        "--semantic-failure-threshold",
        "--semantic-review-threshold",
        dest="semantic_review_threshold",
        metavar="SEMANTIC_FAILURE_THRESHOLD",
        type=float,
        default=0.5,
        help="score below this value is reviewed as a likely semantic failure",
    )
    parser.add_argument(
        "--semantic-pass-threshold",
        type=float,
        default=0.9,
        help="only scores at or above this value receive automatic semantic pass",
    )
    parser.add_argument(
        "--semantic-per-view",
        action="store_true",
        help=(
            "additionally score each camera view separately and review episodes "
            "whose views disagree; costs one extra evaluator call per view"
        ),
    )
    parser.add_argument(
        "--semantic-view-disagreement-threshold",
        type=float,
        default=0.30,
        help="per-view score gap at or above this value is reviewed",
    )
    parser.add_argument(
        "--semantic-uncertainty-threshold",
        type=float,
        default=0.15,
        help=(
            "score dispersion (std of score_samples or explicit score_std) at or "
            "above this value demotes a pass to review"
        ),
    )
    parser.add_argument(
        "--semantic-counterfactual-margin-threshold",
        type=float,
        default=0.10,
        help=(
            "details.counterfactual_margin below this value is reviewed: a "
            "corrupted instruction scored nearly as well as the true one"
        ),
    )
    parser.add_argument(
        "--task-profile-dir",
        type=Path,
        default=None,
        help="optional directory containing validated task semantic profile JSON files",
    )
    parser.add_argument(
        "--feedback-sidecar",
        type=Path,
        default=None,
        help="optional feedback sidecar directory to validate against dataset frames",
    )
    parser.add_argument(
        "--semantic-only",
        action="store_true",
        help=(
            "reuse the structural results of a prior fingerprint-matched report and "
            "rerun only the semantic evaluation; requires --semantic-evaluator"
        ),
    )
    parser.add_argument(
        "--reuse-report",
        type=Path,
        default=None,
        help=(
            "prior quality_report.csv to reuse with --semantic-only; "
            "defaults to <out-dir>/quality_report.csv"
        ),
    )
    args = parser.parse_args()

    out_dir = args.out_dir or default_audit_output_dir(args.dataset)
    semantic_evaluator = (
        load_evaluator(args.semantic_evaluator, load_evaluator_config(args.semantic_config))
        if args.semantic_evaluator
        else None
    )
    task_profiles = (
        TaskProfileRegistry.from_directory(args.task_profile_dir)
        if args.task_profile_dir is not None
        else None
    )
    if args.semantic_only:
        if semantic_evaluator is None:
            parser.error("--semantic-only requires --semantic-evaluator")
        raise SystemExit(
            audit_dataset_semantic_reuse(
                root=args.dataset,
                out_dir=out_dir,
                semantic_evaluator=semantic_evaluator,
                reuse_report=args.reuse_report,
                semantic_review_threshold=args.semantic_review_threshold,
                semantic_pass_threshold=args.semantic_pass_threshold,
                task_profiles=task_profiles,
                motion_check=args.motion_check,
                motion_gripper_index=args.motion_gripper_index,
                motion_vertical_index=args.motion_vertical_index,
                semantic_per_view=args.semantic_per_view,
                semantic_view_disagreement_threshold=args.semantic_view_disagreement_threshold,
                semantic_uncertainty_threshold=args.semantic_uncertainty_threshold,
                semantic_counterfactual_margin_threshold=args.semantic_counterfactual_margin_threshold,
            )
        )
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
            semantic_evaluator=semantic_evaluator,
            semantic_review_threshold=args.semantic_review_threshold,
            semantic_pass_threshold=args.semantic_pass_threshold,
            task_profiles=task_profiles,
            feedback_sidecar=args.feedback_sidecar,
            motion_check=args.motion_check,
            motion_gripper_index=args.motion_gripper_index,
            motion_vertical_index=args.motion_vertical_index,
            semantic_per_view=args.semantic_per_view,
            semantic_view_disagreement_threshold=args.semantic_view_disagreement_threshold,
            semantic_uncertainty_threshold=args.semantic_uncertainty_threshold,
            semantic_counterfactual_margin_threshold=args.semantic_counterfactual_margin_threshold,
            workers=args.workers,
        )
    )


if __name__ == "__main__":
    main()
