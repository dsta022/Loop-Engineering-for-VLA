#!/usr/bin/env python3
"""Build a real cleaned LeRobot dataset from an audited drop_episodes.txt.

The source dataset is never modified. The output dataset rewrites contiguous
episode indices, global frame indices, task indices, metadata, data parquet
files, depth sidecars, and video references.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from utils.dataset_io import copy_file, copy_tree, find_env_executable, prepare_output_dir, read_json, safe_rel_path, write_json
from utils.dataset_metadata import (
    build_stats_json,
    chunk_index_for,
    data_path_for_episode,
    feature_names_by_dtype,
    file_index_in_chunk,
    make_episode_row,
    rewrite_data_table,
    validate_full_dataset,
    video_expected_codec,
)
from utils.media_utils import run_ffmpeg_trim
from utils.quality_utils import default_drop_list, load_quality_rows, parse_episode_list


DEFAULT_VIDEO_CRF = 30


@dataclass(frozen=True)
class CleanConfig:
    src_dataset: Path
    out_dir: Path
    drop_list: Path
    overwrite: bool
    video_mode: str
    copy_mode: str
    ffmpeg_path: str | None
    video_crf: int
    dry_run: bool


@dataclass(frozen=True)
class CleanPlan:
    info: dict[str, Any]
    chunks_size: int
    total_episodes: int
    drop_episodes: list[int]
    keep_episodes: list[int]
    episodes_table: pa.Table
    episode_schema: pa.Schema
    episode_columns: dict[str, list[Any]]
    tasks_table: pa.Table
    task_schema: pa.Schema
    task_columns: dict[str, list[Any]]
    kept_old_tasks: list[int]
    task_map: dict[int, int]
    old_episode_rows: dict[int, dict[str, Any]]


@dataclass
class CleanState:
    video_keys: list[str]
    image_keys: list[str]
    episode_schema: pa.Schema
    chunks_size: int
    next_video_file: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    copied_video_map: dict[tuple[str, int, int], tuple[int, int]] = field(default_factory=dict)
    new_episode_columns: dict[str, list[Any]] = field(init=False)
    new_global_from: int = 0
    materialized_depth: int = 0

    def __post_init__(self) -> None:
        self.new_episode_columns = {name: [] for name in self.episode_schema.names}


def copy_image_sidecar(
    src_dataset: Path,
    out_dir: Path,
    image_key: str,
    old_episode: int,
    new_episode: int,
    data_table: pa.Table,
    copy_mode: str,
) -> tuple[int, int]:
    src_dir = src_dataset / "depth_sidecar" / image_key / f"episode-{old_episode:06d}"
    dst_dir = out_dir / "depth_sidecar" / image_key / f"episode-{new_episode:06d}"
    if dst_dir.exists():
        raise FileExistsError(f"depth sidecar target already exists: {dst_dir}")
    if src_dir.exists():
        copy_tree(src_dir, dst_dir, copy_mode)
    else:
        dst_dir.mkdir(parents=True, exist_ok=True)

    if image_key not in data_table.column_names:
        return 0, 0

    image_col = data_table[image_key].combine_chunks()
    path_col = image_col.field("path")
    struct_field_names = {field.name for field in image_col.type}
    bytes_col = image_col.field("bytes") if "bytes" in struct_field_names else None

    checked = 0
    materialized = 0
    for row_index, rel_value in enumerate(path_col):
        rel = safe_rel_path(rel_value.as_py())
        if rel is None:
            continue
        checked += 1
        dst_file = dst_dir / rel
        if dst_file.exists() or bytes_col is None:
            continue
        payload = bytes_col[row_index].as_py()
        if payload is None:
            continue
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        dst_file.write_bytes(payload)
        materialized += 1
    return checked, materialized


def load_clean_plan(config: CleanConfig) -> CleanPlan:
    src_dataset = config.src_dataset.resolve()
    validate_full_dataset(src_dataset)
    info = read_json(src_dataset / "meta" / "info.json")
    chunks_size = int(info["chunks_size"])
    total_episodes = int(info["total_episodes"])
    drop_episodes = parse_episode_list(config.drop_list)
    keep_episodes = kept_episode_indices(drop_episodes, total_episodes)

    episodes_table = pq.read_table(src_dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    if episodes_table.num_rows != total_episodes:
        raise ValueError(f"episode metadata row count mismatch: {episodes_table.num_rows}!={total_episodes}")
    episode_schema = episodes_table.schema
    episode_columns = episodes_table.to_pydict()
    rows_by_episode = {int(value): i for i, value in enumerate(episode_columns["episode_index"])}

    tasks_table = pq.read_table(src_dataset / "meta" / "tasks.parquet")
    task_schema = tasks_table.schema
    task_columns = tasks_table.to_pydict()
    task_rows_by_index = {int(value): i for i, value in enumerate(task_columns["task_index"])}
    old_episode_rows, used_tasks = collect_kept_episode_rows(src_dataset, keep_episodes, episode_schema, episode_columns, rows_by_episode)
    kept_old_tasks, task_map = build_task_map(task_columns, task_rows_by_index, used_tasks)

    return CleanPlan(
        info=info,
        chunks_size=chunks_size,
        total_episodes=total_episodes,
        drop_episodes=drop_episodes,
        keep_episodes=keep_episodes,
        episodes_table=episodes_table,
        episode_schema=episode_schema,
        episode_columns=episode_columns,
        tasks_table=tasks_table,
        task_schema=task_schema,
        task_columns=task_columns,
        kept_old_tasks=kept_old_tasks,
        task_map=task_map,
        old_episode_rows=old_episode_rows,
    )


def kept_episode_indices(drop_episodes: list[int], total_episodes: int) -> list[int]:
    invalid = [episode for episode in drop_episodes if episode < 0 or episode >= total_episodes]
    if invalid:
        raise ValueError(f"drop list contains episode ids outside 0:{total_episodes}: {invalid[:20]}")
    drop_set = set(drop_episodes)
    keep_episodes = [episode for episode in range(total_episodes) if episode not in drop_set]
    if not keep_episodes:
        raise ValueError("drop list removes every episode; refusing to build an empty dataset")
    return keep_episodes


def collect_kept_episode_rows(
    src_dataset: Path,
    keep_episodes: list[int],
    episode_schema: pa.Schema,
    episode_columns: dict[str, list[Any]],
    rows_by_episode: dict[int, int],
) -> tuple[dict[int, dict[str, Any]], set[int]]:
    used_tasks: set[int] = set()
    old_episode_rows: dict[int, dict[str, Any]] = {}
    for old_episode in keep_episodes:
        row_idx = rows_by_episode[old_episode]
        row = {name: episode_columns[name][row_idx] for name in episode_schema.names}
        old_episode_rows[old_episode] = row
        task_table = pq.read_table(data_path_for_episode(src_dataset, row), columns=["task_index"])
        used_tasks.update(int(value.as_py()) for value in task_table["task_index"])
    return old_episode_rows, used_tasks


def build_task_map(
    task_columns: dict[str, list[Any]],
    task_rows_by_index: dict[int, int],
    used_tasks: set[int],
) -> tuple[list[int], dict[int, int]]:
    missing_task_rows = sorted(used_tasks.difference(task_rows_by_index))
    if missing_task_rows:
        raise ValueError(f"task indices appear in data but not meta/tasks.parquet: {missing_task_rows}")
    kept_old_tasks = [int(value) for value in task_columns["task_index"] if int(value) in used_tasks]
    return kept_old_tasks, {old_task: new_task for new_task, old_task in enumerate(kept_old_tasks)}


def print_dry_run(config: CleanConfig, plan: CleanPlan) -> None:
    print(f"Source: {config.src_dataset}")
    print(f"Output: {config.out_dir}")
    print(f"Drop list: {config.drop_list}")
    print(f"Episodes: keep={len(plan.keep_episodes)} drop={len(plan.drop_episodes)} total={plan.total_episodes}")
    print(f"Tasks: keep={len(plan.kept_old_tasks)} total={len(plan.task_columns['task_index'])}")
    print(f"Video mode: {config.video_mode}")


def resolve_ffmpeg(config: CleanConfig) -> str | None:
    if config.video_mode not in {"trim-reencode", "trim-copy"}:
        return config.ffmpeg_path
    ffmpeg_path = config.ffmpeg_path or find_env_executable("ffmpeg")
    if ffmpeg_path is None:
        raise FileNotFoundError("ffmpeg not found; install it or use --video-mode copy-referenced")
    return ffmpeg_path


def write_rewritten_data(
    config: CleanConfig,
    plan: CleanPlan,
    state: CleanState,
    old_row: dict[str, Any],
    new_episode: int,
) -> tuple[pa.Table, list[int]]:
    data_table = pq.read_table(data_path_for_episode(config.src_dataset, old_row))
    rewritten_data, original_task_values = rewrite_data_table(
        table=data_table,
        new_episode=new_episode,
        new_global_from=state.new_global_from,
        task_map=plan.task_map,
    )
    dst_chunk = chunk_index_for(new_episode, plan.chunks_size)
    dst_file = file_index_in_chunk(new_episode, plan.chunks_size)
    dst_data = config.out_dir / "data" / f"chunk-{dst_chunk:03d}" / f"file-{dst_file:03d}.parquet"
    dst_data.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(rewritten_data, dst_data)
    return rewritten_data, original_task_values


def copy_episode_depth_sidecars(
    config: CleanConfig,
    state: CleanState,
    old_episode: int,
    new_episode: int,
    rewritten_data: pa.Table,
) -> None:
    for image_key in state.image_keys:
        _, missing_written = copy_image_sidecar(
            src_dataset=config.src_dataset,
            out_dir=config.out_dir,
            image_key=image_key,
            old_episode=old_episode,
            new_episode=new_episode,
            data_table=rewritten_data,
            copy_mode=config.copy_mode,
        )
        state.materialized_depth += missing_written


def update_video_fields(
    config: CleanConfig,
    plan: CleanPlan,
    state: CleanState,
    old_row: dict[str, Any],
    row_for_meta: dict[str, Any],
    ffmpeg_path: str | None,
) -> None:
    for video_key in state.video_keys:
        old_chunk = int(old_row[f"videos/{video_key}/chunk_index"])
        old_file = int(old_row[f"videos/{video_key}/file_index"])
        old_start = float(old_row[f"videos/{video_key}/from_timestamp"])
        old_stop = float(old_row[f"videos/{video_key}/to_timestamp"])
        src_video = config.src_dataset / "videos" / video_key / f"chunk-{old_chunk:03d}" / f"file-{old_file:03d}.mp4"
        if not src_video.exists():
            raise FileNotFoundError(f"missing source video: {src_video}")

        if config.video_mode == "copy-referenced":
            new_chunk, new_file, new_start, new_stop = copy_referenced_video(
                config, state, video_key, old_chunk, old_file, old_start, old_stop, src_video
            )
        else:
            new_chunk, new_file, new_start, new_stop = trim_episode_video(
                config, plan, state, video_key, old_start, old_stop, src_video, ffmpeg_path
            )
        row_for_meta[f"videos/{video_key}/chunk_index"] = new_chunk
        row_for_meta[f"videos/{video_key}/file_index"] = new_file
        row_for_meta[f"videos/{video_key}/from_timestamp"] = new_start
        row_for_meta[f"videos/{video_key}/to_timestamp"] = new_stop


def copy_referenced_video(
    config: CleanConfig,
    state: CleanState,
    video_key: str,
    old_chunk: int,
    old_file: int,
    old_start: float,
    old_stop: float,
    src_video: Path,
) -> tuple[int, int, float, float]:
    key = (video_key, old_chunk, old_file)
    if key not in state.copied_video_map:
        new_video_global = state.next_video_file[video_key]
        state.next_video_file[video_key] += 1
        new_chunk = chunk_index_for(new_video_global, state.chunks_size)
        new_file = file_index_in_chunk(new_video_global, state.chunks_size)
        state.copied_video_map[key] = (new_chunk, new_file)
        dst_video = config.out_dir / "videos" / video_key / f"chunk-{new_chunk:03d}" / f"file-{new_file:03d}.mp4"
        copy_file(src_video, dst_video, config.copy_mode)
    new_chunk, new_file = state.copied_video_map[key]
    return new_chunk, new_file, old_start, old_stop


def trim_episode_video(
    config: CleanConfig,
    plan: CleanPlan,
    state: CleanState,
    video_key: str,
    old_start: float,
    old_stop: float,
    src_video: Path,
    ffmpeg_path: str | None,
) -> tuple[int, int, float, float]:
    if ffmpeg_path is None:
        raise FileNotFoundError("ffmpeg not found; install it or use --video-mode copy-referenced")
    new_video_global = state.next_video_file[video_key]
    state.next_video_file[video_key] += 1
    new_chunk = chunk_index_for(new_video_global, state.chunks_size)
    new_file = file_index_in_chunk(new_video_global, state.chunks_size)
    dst_video = config.out_dir / "videos" / video_key / f"chunk-{new_chunk:03d}" / f"file-{new_file:03d}.mp4"
    duration = old_stop - old_start
    run_ffmpeg_trim(
        ffmpeg_path=str(ffmpeg_path),
        src_video=src_video,
        dst_video=dst_video,
        start=old_start,
        duration=duration,
        mode=config.video_mode,
        codec=video_expected_codec(plan.info, video_key),
        crf=config.video_crf,
    )
    return new_chunk, new_file, 0.0, duration


def append_episode_metadata_row(state: CleanState, row_for_meta: dict[str, Any]) -> None:
    for name in state.episode_schema.names:
        state.new_episode_columns[name].append(row_for_meta[name])


def rewrite_kept_episode(
    config: CleanConfig,
    plan: CleanPlan,
    state: CleanState,
    new_episode: int,
    old_episode: int,
    ffmpeg_path: str | None,
) -> None:
    old_row = plan.old_episode_rows[old_episode]
    rewritten_data, original_task_values = write_rewritten_data(config, plan, state, old_row, new_episode)
    copy_episode_depth_sidecars(config, state, old_episode, new_episode, rewritten_data)

    row_for_meta = dict(old_row)
    row_for_meta["_chunks_size"] = plan.chunks_size
    row_for_meta = make_episode_row(
        old_row=row_for_meta,
        schema=plan.episode_schema,
        new_episode=new_episode,
        new_global_from=state.new_global_from,
        row_count=rewritten_data.num_rows,
        task_map=plan.task_map,
        original_task_values=original_task_values,
    )
    update_video_fields(config, plan, state, old_row, row_for_meta, ffmpeg_path)
    append_episode_metadata_row(state, row_for_meta)
    state.new_global_from += rewritten_data.num_rows
    if new_episode % 10 == 0 or new_episode == len(plan.keep_episodes) - 1:
        print(f"episode {old_episode:06d}->{new_episode:06d} frames={rewritten_data.num_rows}")


def build_task_columns(plan: CleanPlan) -> dict[str, list[Any]]:
    task_rows_by_index = {int(value): i for i, value in enumerate(plan.task_columns["task_index"])}
    new_task_columns: dict[str, list[Any]] = {name: [] for name in plan.task_schema.names}
    for old_task in plan.kept_old_tasks:
        row_idx = task_rows_by_index[old_task]
        for name in plan.task_schema.names:
            value = plan.task_columns[name][row_idx]
            if name == "task_index":
                value = plan.task_map[old_task]
            new_task_columns[name].append(value)
    return new_task_columns


def write_clean_metadata(config: CleanConfig, plan: CleanPlan, state: CleanState) -> tuple[pa.Table, pa.Table]:
    meta_dir = config.out_dir / "meta"
    episodes_out = meta_dir / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_out.parent.mkdir(parents=True, exist_ok=True)
    episodes_table_out = pa.Table.from_pydict(state.new_episode_columns, schema=plan.episode_schema)
    tasks_table_out = pa.Table.from_pydict(build_task_columns(plan), schema=plan.task_schema)
    pq.write_table(episodes_table_out, episodes_out)
    pq.write_table(tasks_table_out, meta_dir / "tasks.parquet")

    new_info = dict(plan.info)
    new_info["total_episodes"] = len(plan.keep_episodes)
    new_info["total_frames"] = state.new_global_from
    new_info["total_tasks"] = tasks_table_out.num_rows
    new_info["splits"] = {"train": f"0:{len(plan.keep_episodes)}"}
    write_json(meta_dir / "info.json", new_info)
    write_json(meta_dir / "stats.json", build_stats_json(episodes_table_out))

    recording = config.src_dataset / "meta" / "rgbd_vla_depth_recording.json"
    if recording.exists():
        copy_file(recording, meta_dir / "rgbd_vla_depth_recording.json", "copy")
    return episodes_table_out, tasks_table_out


def write_cleaning_manifest(config: CleanConfig, plan: CleanPlan, state: CleanState) -> None:
    quality_rows = load_quality_rows(config.drop_list)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(config.src_dataset),
        "output_dataset": str(config.out_dir.resolve()),
        "drop_list": str(config.drop_list.resolve()),
        "video_mode": config.video_mode,
        "copy_mode": config.copy_mode,
        "source_total_episodes": plan.total_episodes,
        "kept_episodes": plan.keep_episodes,
        "dropped_episodes": plan.drop_episodes,
        "episode_index_mapping": {str(old): new for new, old in enumerate(plan.keep_episodes)},
        "task_index_mapping": {str(old): new for old, new in plan.task_map.items()},
        "drop_reasons": {
            str(ep): {
                "status": quality_rows.get(ep, {}).get("status", ""),
                "drop_reasons": quality_rows.get(ep, {}).get("drop_reasons", ""),
                "review_reasons": quality_rows.get(ep, {}).get("review_reasons", ""),
            }
            for ep in plan.drop_episodes
        },
        "physical_video_policy": (
            "trimmed per kept episode and rewritten to new video files"
            if config.video_mode in {"trim-reencode", "trim-copy"}
            else "copied referenced source video files; unreferenced segments may remain inside copied videos"
        ),
        "materialized_missing_depth_files": state.materialized_depth,
    }
    write_json(config.out_dir / "cleaning_manifest.json", manifest)


def run_clean(config: CleanConfig) -> None:
    config = CleanConfig(**{**config.__dict__, "src_dataset": config.src_dataset.resolve()})
    plan = load_clean_plan(config)
    if config.dry_run:
        print_dry_run(config, plan)
        return

    ffmpeg_path = resolve_ffmpeg(config)
    prepare_output_dir(config.src_dataset, config.out_dir, config.overwrite)
    state = CleanState(
        video_keys=feature_names_by_dtype(plan.info, "video"),
        image_keys=feature_names_by_dtype(plan.info, "image"),
        episode_schema=plan.episode_schema,
        chunks_size=plan.chunks_size,
    )
    for new_episode, old_episode in enumerate(plan.keep_episodes):
        rewrite_kept_episode(config, plan, state, new_episode, old_episode, ffmpeg_path)

    _, tasks_table_out = write_clean_metadata(config, plan, state)
    write_cleaning_manifest(config, plan, state)
    print("=" * 80)
    print("CLEAN DATASET COMPLETE")
    print(f"Output: {config.out_dir}")
    print(f"Episodes: kept={len(plan.keep_episodes)} dropped={len(plan.drop_episodes)}")
    print(f"Frames: {state.new_global_from}")
    print(f"Tasks: {tasks_table_out.num_rows}")
    print(f"Manifest: {config.out_dir / 'cleaning_manifest.json'}")


def build_clean_dataset(
    src_dataset: Path,
    out_dir: Path,
    drop_list: Path,
    overwrite: bool,
    video_mode: str,
    copy_mode: str,
    ffmpeg_path: str | None,
    video_crf: int,
    dry_run: bool,
) -> None:
    """Build a cleaned dataset; kept for compatibility with existing usage."""
    run_clean(
        CleanConfig(
            src_dataset=src_dataset,
            out_dir=out_dir,
            drop_list=drop_list,
            overwrite=overwrite,
            video_mode=video_mode,
            copy_mode=copy_mode,
            ffmpeg_path=ffmpeg_path,
            video_crf=video_crf,
            dry_run=dry_run,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a cleaned real LeRobot dataset from drop_episodes.txt.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--drop-list", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-mode", choices=["copy", "hardlink"], default="copy")
    parser.add_argument(
        "--video-mode",
        choices=["trim-reencode", "trim-copy", "copy-referenced"],
        default="trim-reencode",
        help="trim-reencode physically removes dropped video segments; copy-referenced is faster but may keep unreferenced segments in copied mp4 files",
    )
    parser.add_argument("--ffmpeg", type=Path, default=None)
    parser.add_argument("--video-crf", type=int, default=DEFAULT_VIDEO_CRF)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    drop_list = args.drop_list or default_drop_list(args.dataset)
    out_dir = args.out_dir or args.dataset.parent / f"{args.dataset.name}_clean"

    build_clean_dataset(
        src_dataset=args.dataset,
        out_dir=out_dir,
        drop_list=drop_list,
        overwrite=args.overwrite,
        video_mode=args.video_mode,
        copy_mode=args.copy_mode,
        ffmpeg_path=str(args.ffmpeg) if args.ffmpeg else None,
        video_crf=args.video_crf,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
