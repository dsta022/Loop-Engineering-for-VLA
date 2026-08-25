#!/usr/bin/env python3
"""Merge multiple LeRobot v3 datasets into one real dataset directory.

The output is not a directory of source-dataset subfolders. It rewrites
episode indices, global frame indices, task indices, data/video file indices,
and metadata so the output has the same LeRobot directory structure as each
source dataset.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from utils.dataset_io import copy_file, copy_tree, read_json, remove_tree_safely, safe_rel_path, write_json
from utils.dataset_metadata import (
    CANONICAL_DATA_PATH,
    CANONICAL_VIDEO_PATH,
    EpisodeDataReader,
    add_int_column,
    append_columns,
    build_stats_json,
    chunk_index_for,
    file_index_in_chunk,
    map_int_column,
    table_column_index,
    update_episode_metadata,
    validate_merge_source_dataset,
    video_keys_from_info,
)


SRC_ROOT = Path(".")
OUT_DIR = Path("determ_lerobot_rgb_depth_vla")


@dataclass(frozen=True)
class MergeConfig:
    src_root: Path
    out_dir: Path
    overwrite: bool
    dataset_names: list[str]
    copy_mode: str


@dataclass(frozen=True)
class MergeSource:
    path: Path
    info: dict[str, Any]
    episodes_table: pa.Table
    episode_rows: list[dict[str, Any]]


@dataclass
class MergeState:
    merged_episode_columns: dict[str, list[Any]] = field(default_factory=dict)
    merged_task_columns: dict[str, list[Any]] = field(default_factory=dict)
    episode_schema: pa.Schema | None = None
    task_schema: pa.Schema | None = None
    next_video_file: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    episode_offset: int = 0
    frame_offset: int = 0
    task_offset: int = 0
    total_frames: int = 0
    total_episodes: int = 0
    missing_depth_written_total: int = 0


def copy_depth_sidecar(
    src_dataset: Path,
    out_dir: Path,
    image_key: str,
    old_episode_index: int,
    new_episode_index: int,
    data_table: pa.Table,
    copy_mode: str,
) -> tuple[int, int]:
    """Copy depth PNGs and materialize missing sidecar files from parquet bytes."""
    src_episode_dir = src_dataset / "depth_sidecar" / image_key / f"episode-{old_episode_index:06d}"
    dst_episode_dir = out_dir / "depth_sidecar" / image_key / f"episode-{new_episode_index:06d}"
    if dst_episode_dir.exists():
        raise FileExistsError(f"depth sidecar target already exists: {dst_episode_dir}")
    if src_episode_dir.exists():
        copy_tree(src_episode_dir, dst_episode_dir, copy_mode)
    else:
        dst_episode_dir.mkdir(parents=True, exist_ok=True)

    if table_column_index(data_table, image_key) < 0:
        return 0, 0

    image_col = data_table[image_key].combine_chunks()
    path_col = image_col.field("path")
    field_names = {field.name for field in image_col.type}
    bytes_col = image_col.field("bytes") if "bytes" in field_names else None

    missing_written = 0
    checked = 0
    for row_index, rel_value in enumerate(path_col):
        rel_path = safe_rel_path(rel_value.as_py())
        if rel_path is None:
            continue
        dst_file = dst_episode_dir / rel_path
        checked += 1
        if dst_file.exists() or bytes_col is None:
            continue
        payload = bytes_col[row_index].as_py()
        if payload is None:
            continue
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        dst_file.write_bytes(payload)
        missing_written += 1

    return checked, missing_written


def copy_videos_for_dataset(
    src_dataset: Path,
    out_dir: Path,
    episodes_columns: dict[str, list[Any]],
    video_keys: list[str],
    next_video_file: dict[str, int],
    chunks_size: int,
    copy_mode: str,
) -> dict[tuple[str, int, int], tuple[int, int]]:
    video_map: dict[tuple[str, int, int], tuple[int, int]] = {}

    for video_key in video_keys:
        chunk_col = f"videos/{video_key}/chunk_index"
        file_col = f"videos/{video_key}/file_index"
        if chunk_col not in episodes_columns or file_col not in episodes_columns:
            continue

        unique_files = sorted(
            {
                (int(chunk), int(file))
                for chunk, file in zip(episodes_columns[chunk_col], episodes_columns[file_col])
            }
        )
        for old_chunk, old_file in unique_files:
            new_global_file = next_video_file[video_key]
            next_video_file[video_key] += 1
            new_chunk = chunk_index_for(new_global_file, chunks_size)
            new_file = file_index_in_chunk(new_global_file, chunks_size)
            video_map[(video_key, old_chunk, old_file)] = (new_chunk, new_file)

            src_video = src_dataset / "videos" / video_key / f"chunk-{old_chunk:03d}" / f"file-{old_file:03d}.mp4"
            dst_video = out_dir / "videos" / video_key / f"chunk-{new_chunk:03d}" / f"file-{new_file:03d}.mp4"
            if not src_video.exists():
                raise FileNotFoundError(f"missing video file: {src_video}")
            copy_file(src_video, dst_video, copy_mode)

    return video_map


def discover_dataset_names(src_root: Path, out_dir: Path, dataset_glob: str) -> list[str]:
    out_resolved = out_dir.resolve() if out_dir.exists() else out_dir.absolute()
    names = []
    for path in sorted(src_root.glob(dataset_glob)):
        if not path.is_dir():
            continue
        candidate_resolved = path.resolve()
        if candidate_resolved == out_resolved:
            continue
        if (path / "meta" / "info.json").exists():
            names.append(path.name)
    return names


def image_keys_from_info(info: dict[str, Any]) -> list[str]:
    return [name for name, spec in info.get("features", {}).items() if spec.get("dtype") == "image"]


def prepare_merge_output(config: MergeConfig) -> None:
    if config.out_dir.exists():
        if not config.overwrite:
            raise FileExistsError(f"output already exists, rerun with --overwrite: {config.out_dir}")
        remove_tree_safely(config.out_dir)
    config.out_dir.mkdir(parents=True)
    for dirname in ["data", "videos", "depth_sidecar", "meta"]:
        (config.out_dir / dirname).mkdir(parents=True, exist_ok=True)


def source_dataset_paths(config: MergeConfig) -> list[Path]:
    src_paths = [config.src_root / name for name in config.dataset_names]
    for path in src_paths:
        validate_merge_source_dataset(path)
    return src_paths


def load_merge_source(path: Path) -> MergeSource:
    validate_merge_source_dataset(path)
    info = read_json(path / "meta" / "info.json")
    episode_paths = sorted((path / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if len(episode_paths) != 1:
        raise ValueError(
            f"merge currently requires one episode metadata shard in {path.name}; "
            f"found {len(episode_paths)}"
        )

    episodes_table = pq.read_table(episode_paths[0])
    required_columns = {
        "episode_index",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    }
    missing = sorted(required_columns.difference(episodes_table.column_names))
    if missing:
        raise ValueError(f"episode metadata is missing required columns in {path.name}: {missing}")

    declared_episodes = int(info["total_episodes"])
    if episodes_table.num_rows != declared_episodes:
        raise ValueError(
            f"episode metadata row count mismatch in {path.name}: "
            f"{episodes_table.num_rows}!={declared_episodes}"
        )
    episode_rows = episodes_table.to_pylist()
    episode_indices = [int(row["episode_index"]) for row in episode_rows]
    if episode_indices != list(range(declared_episodes)):
        raise ValueError(
            f"episode metadata indices must be contiguous in {path.name}: "
            f"found={episode_indices[:20]}"
        )

    expected_from = 0
    for row in episode_rows:
        actual_from = int(row["dataset_from_index"])
        if actual_from != expected_from:
            raise ValueError(
                f"episode metadata ranges must start at zero and be contiguous in {path.name}: "
                f"episode={row['episode_index']}, expected={expected_from}, actual={actual_from}"
            )
        expected_from = int(row["dataset_to_index"])

    declared_frames = int(info["total_frames"])
    metadata_frames = sum(int(row["length"]) for row in episode_rows)
    if metadata_frames != declared_frames:
        raise ValueError(
            f"episode metadata frame count mismatch in {path.name}: "
            f"{metadata_frames}!={declared_frames}"
        )
    return MergeSource(path=path, info=info, episodes_table=episodes_table, episode_rows=episode_rows)


def validate_dataset_compatibility(src_dataset: Path, info: dict[str, Any], first_info: dict[str, Any]) -> None:
    if info.get("features") != first_info.get("features"):
        raise ValueError(f"feature schema differs in {src_dataset.name}")
    if int(info["chunks_size"]) != int(first_info["chunks_size"]):
        raise ValueError(f"chunks_size differs in {src_dataset.name}")


def append_tasks(src_dataset: Path, state: MergeState) -> tuple[pa.Table, dict[int, int]]:
    tasks_table = pq.read_table(src_dataset / "meta" / "tasks.parquet")
    if state.task_schema is None:
        state.task_schema = tasks_table.schema
    task_cols = tasks_table.to_pydict()
    task_map = {int(old): state.task_offset + i for i, old in enumerate(task_cols["task_index"])}
    updated_tasks = dict(task_cols)
    updated_tasks["task_index"] = [task_map[int(v)] for v in task_cols["task_index"]]
    append_columns(state.merged_task_columns, updated_tasks, state.task_schema)
    return tasks_table, task_map


def append_episode_metadata(
    config: MergeConfig,
    src_dataset: Path,
    episodes_table: pa.Table,
    state: MergeState,
    video_keys: list[str],
    chunks_size: int,
) -> None:
    if state.episode_schema is None:
        state.episode_schema = episodes_table.schema
    elif episodes_table.schema.names != state.episode_schema.names:
        raise ValueError(f"episode metadata columns differ in {src_dataset.name}")

    episode_cols = episodes_table.to_pydict()
    video_map = copy_videos_for_dataset(
        src_dataset=src_dataset,
        out_dir=config.out_dir,
        episodes_columns=episode_cols,
        video_keys=video_keys,
        next_video_file=state.next_video_file,
        chunks_size=chunks_size,
        copy_mode=config.copy_mode,
    )
    updated_episode_cols = update_episode_metadata(
        columns=episode_cols,
        schema=state.episode_schema,
        episode_offset=state.episode_offset,
        frame_offset=state.frame_offset,
        task_offset=state.task_offset,
        video_map=video_map,
        video_keys=video_keys,
        chunks_size=chunks_size,
        meta_file_index=0,
    )
    append_columns(state.merged_episode_columns, updated_episode_cols, state.episode_schema)


def rewrite_episode_data_file(
    config: MergeConfig,
    src_dataset: Path,
    old_row: dict[str, Any],
    data_reader: EpisodeDataReader,
    task_map: dict[int, int],
    image_keys: list[str],
    chunks_size: int,
    state: MergeState,
) -> int:
    old_episode = int(old_row["episode_index"])
    new_episode = state.episode_offset + old_episode
    dst_chunk = chunk_index_for(new_episode, chunks_size)
    dst_file = file_index_in_chunk(new_episode, chunks_size)

    dst_data = config.out_dir / "data" / f"chunk-{dst_chunk:03d}" / f"file-{dst_file:03d}.parquet"
    data_table = data_reader.read_episode(old_row)
    data_table = add_int_column(data_table, "episode_index", state.episode_offset)
    data_table = add_int_column(data_table, "index", state.frame_offset)
    data_table = map_int_column(data_table, "task_index", task_map)

    dst_data.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(data_table, dst_data)

    checked_depth = 0
    missing_written = 0
    for image_key in image_keys:
        checked_depth, missing_written = copy_depth_sidecar(
            src_dataset=src_dataset,
            out_dir=config.out_dir,
            image_key=image_key,
            old_episode_index=old_episode,
            new_episode_index=new_episode,
            data_table=data_table,
            copy_mode=config.copy_mode,
        )
        state.missing_depth_written_total += missing_written

    if old_episode % 10 == 0:
        print_episode_progress(old_episode, new_episode, data_table.num_rows, checked_depth, missing_written)
    return data_table.num_rows


def print_episode_progress(
    old_episode: int,
    new_episode: int,
    frame_count: int,
    checked_depth: int,
    missing_written: int,
) -> None:
    print(
        f"  episode {old_episode:03d}->{new_episode:03d} "
        f"frames={frame_count} depth_checked={checked_depth} "
        f"missing_depth_written={missing_written}"
    )


def rewrite_dataset_data_files(
    config: MergeConfig,
    src_dataset: Path,
    task_map: dict[int, int],
    image_keys: list[str],
    chunks_size: int,
    episode_rows: list[dict[str, Any]],
    state: MergeState,
) -> int:
    data_reader = EpisodeDataReader(src_dataset)
    written_frames = 0
    for row_no, old_row in enumerate(episode_rows):
        old_episode = int(old_row["episode_index"])
        written_frames += rewrite_episode_data_file(
            config, src_dataset, old_row, data_reader, task_map, image_keys, chunks_size, state
        )
        if row_no == len(episode_rows) - 1 and old_episode % 10 != 0:
            new_episode = state.episode_offset + old_episode
            print(f"  episode {old_episode:03d}->{new_episode:03d} complete")
    return written_frames


def advance_offsets(
    src_dataset: Path,
    info: dict[str, Any],
    tasks_table: pa.Table,
    written_episodes: int,
    written_frames: int,
    state: MergeState,
) -> None:
    declared_episodes = int(info["total_episodes"])
    declared_frames = int(info["total_frames"])
    if written_episodes != declared_episodes or written_frames != declared_frames:
        raise ValueError(
            f"written dataset totals differ in {src_dataset.name}: "
            f"episodes={written_episodes}!={declared_episodes}, "
            f"frames={written_frames}!={declared_frames}"
        )
    state.episode_offset += written_episodes
    state.frame_offset += written_frames
    state.task_offset += tasks_table.num_rows
    state.total_episodes += written_episodes
    state.total_frames += written_frames
    print(
        f"Done {src_dataset.name}: episodes={written_episodes}, frames={written_frames}, "
        f"cumulative episodes={state.total_episodes}, frames={state.total_frames}"
    )


def merge_one_dataset(
    config: MergeConfig,
    source: MergeSource,
    dataset_no: int,
    dataset_count: int,
    first_info: dict[str, Any],
    video_keys: list[str],
    image_keys: list[str],
    state: MergeState,
) -> None:
    src_dataset = source.path
    info = source.info
    validate_dataset_compatibility(src_dataset, info, first_info)
    chunks_size = int(first_info["chunks_size"])

    print("=" * 80)
    print(f"[{dataset_no}/{dataset_count}] {src_dataset.name}")

    tasks_table, task_map = append_tasks(src_dataset, state)
    append_episode_metadata(config, src_dataset, source.episodes_table, state, video_keys, chunks_size)
    written_frames = rewrite_dataset_data_files(
        config=config,
        src_dataset=src_dataset,
        task_map=task_map,
        image_keys=image_keys,
        chunks_size=chunks_size,
        episode_rows=source.episode_rows,
        state=state,
    )
    advance_offsets(
        src_dataset,
        info,
        tasks_table,
        written_episodes=len(source.episode_rows),
        written_frames=written_frames,
        state=state,
    )


def write_merged_metadata(config: MergeConfig, first_info: dict[str, Any], src_paths: list[Path], state: MergeState) -> None:
    if state.episode_schema is None or state.task_schema is None:
        raise RuntimeError("no datasets were merged")

    meta_dir = config.out_dir / "meta"
    episodes_out = meta_dir / "episodes" / "chunk-000" / "file-000.parquet"
    tasks_out = meta_dir / "tasks.parquet"
    episodes_table_out = pa.Table.from_pydict(state.merged_episode_columns, schema=state.episode_schema)
    tasks_table_out = pa.Table.from_pydict(state.merged_task_columns, schema=state.task_schema)

    episodes_out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(episodes_table_out, episodes_out)
    pq.write_table(tasks_table_out, tasks_out)

    merged_info = dict(first_info)
    merged_info["total_episodes"] = state.total_episodes
    merged_info["total_frames"] = state.total_frames
    merged_info["total_tasks"] = tasks_table_out.num_rows
    merged_info["splits"] = {"train": f"0:{state.total_episodes}"}
    merged_info["data_path"] = CANONICAL_DATA_PATH
    merged_info["video_path"] = CANONICAL_VIDEO_PATH
    write_json(meta_dir / "info.json", merged_info)
    write_json(meta_dir / "stats.json", build_stats_json(episodes_table_out))

    recording = src_paths[0] / "meta" / "rgbd_vla_depth_recording.json"
    if recording.exists():
        copy_file(recording, meta_dir / "rgbd_vla_depth_recording.json", "copy")


def run_merge_paths(src_paths: list[Path], out_dir: Path, overwrite: bool, copy_mode: str) -> None:
    """Merge explicitly addressed datasets without staging them under one parent."""
    if not src_paths:
        raise ValueError("at least one source dataset is required")
    resolved_sources = [path.resolve() for path in src_paths]
    if len(set(resolved_sources)) != len(resolved_sources):
        raise ValueError("source dataset paths must be unique")
    out_resolved = out_dir.resolve() if out_dir.exists() else out_dir.absolute()
    if out_resolved in resolved_sources:
        raise ValueError("merge output must not overwrite a source dataset")
    sources = [load_merge_source(path) for path in resolved_sources]
    for source in resolved_sources:
        try:
            source.relative_to(out_resolved)
        except ValueError:
            continue
        raise ValueError(f"merge output must not contain a source dataset: {source}")
 

    config = MergeConfig(
        src_root=Path("."),
        out_dir=out_dir,
        overwrite=overwrite,
        dataset_names=[],
        copy_mode=copy_mode,
    )
    prepare_merge_output(config)
    src_paths = [source.path for source in sources]
    first_info = sources[0].info
    video_keys = video_keys_from_info(first_info)
    image_keys = image_keys_from_info(first_info)
    state = MergeState()

    print(f"Output: {config.out_dir}")
    print(f"Datasets: {len(src_paths)}")
    print(f"Copy mode: {config.copy_mode}")
    print(f"Video keys: {', '.join(video_keys)}")

    for dataset_no, source in enumerate(sources, start=1):
        merge_one_dataset(config, source, dataset_no, len(sources), first_info, video_keys, image_keys, state)

    write_merged_metadata(config, first_info, src_paths, state)
    print("=" * 80)
    print("MERGE COMPLETE")
    print(f"Output: {config.out_dir}")
    print(f"Total episodes: {state.total_episodes}")
    print(f"Total frames: {state.total_frames}")
    print(f"Total tasks: {state.task_offset}")
    print(f"Depth files materialized from parquet bytes: {state.missing_depth_written_total}")


def run_merge(config: MergeConfig) -> None:
    run_merge_paths(
        source_dataset_paths(config),
        out_dir=config.out_dir,
        overwrite=config.overwrite,
        copy_mode=config.copy_mode,
    )


def merge_datasets(
    src_root: Path,
    out_dir: Path,
    overwrite: bool,
    dataset_names: list[str],
    copy_mode: str,
) -> None:
    """Merge datasets; kept for compatibility with existing scripts."""
    run_merge(
        MergeConfig(
            src_root=src_root,
            out_dir=out_dir,
            overwrite=overwrite,
            dataset_names=dataset_names,
            copy_mode=copy_mode,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LeRobot datasets into one real dataset directory.")
    parser.add_argument("--src-root", type=Path, default=SRC_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dataset-glob", default="*_dataset")
    parser.add_argument("--dataset-names", nargs="*", default=None)
    parser.add_argument("--copy-mode", choices=["copy", "hardlink"], default="copy")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.dataset_names:
        dataset_names = args.dataset_names
    else:
        dataset_names = discover_dataset_names(args.src_root, args.out_dir, args.dataset_glob)

    if not dataset_names:
        raise RuntimeError(f"no source datasets found in {args.src_root} using glob {args.dataset_glob!r}")

    merge_datasets(
        src_root=args.src_root,
        out_dir=args.out_dir,
        overwrite=args.overwrite,
        dataset_names=dataset_names,
        copy_mode=args.copy_mode,
    )


if __name__ == "__main__":
    main()
