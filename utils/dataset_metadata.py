#!/usr/bin/env python3
"""Shared metadata and parquet helpers for LeRobot dataset tools."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


STATS_OFFSET_FEATURES = {
    "episode_index": "episode",
    "index": "frame",
    "task_index": "task",
}

OFFSET_STATS = {"min", "max", "mean", "q01", "q10", "q50", "q90", "q99"}


def chunk_index_for(file_index: int, chunks_size: int) -> int:
    return file_index // chunks_size


def file_index_in_chunk(file_index: int, chunks_size: int) -> int:
    return file_index % chunks_size


def table_column_index(table: pa.Table, name: str) -> int:
    return table.schema.get_field_index(name)


def replace_column(table: pa.Table, name: str, values: pa.Array | pa.ChunkedArray) -> pa.Table:
    idx = table_column_index(table, name)
    if idx < 0:
        return table
    field = table.schema.field(idx)
    arr = values.cast(field.type)
    return table.set_column(idx, field, arr)


def replace_column_values(table: pa.Table, name: str, values: list[Any]) -> pa.Table:
    idx = table.schema.get_field_index(name)
    if idx < 0:
        return table
    field = table.schema.field(idx)
    return table.set_column(idx, field, pa.array(values, type=field.type))


def add_int_column(table: pa.Table, name: str, offset: int) -> pa.Table:
    if offset == 0 or table_column_index(table, name) < 0:
        return table
    field = table.schema.field(name)
    updated = pc.add(table[name], pa.scalar(offset, type=field.type))
    return replace_column(table, name, updated)


def map_int_column(table: pa.Table, name: str, mapping: dict[int, int]) -> pa.Table:
    idx = table_column_index(table, name)
    if idx < 0:
        return table
    field = table.schema.field(idx)
    values = [mapping[int(v.as_py())] for v in table[name]]
    return table.set_column(idx, field, pa.array(values, type=field.type))


def rewrite_data_table(
    table: pa.Table,
    new_episode: int,
    new_global_from: int,
    task_map: dict[int, int],
) -> tuple[pa.Table, list[int]]:
    rows = table.num_rows
    original_tasks = [int(value.as_py()) for value in table["task_index"]] if "task_index" in table.column_names else []
    table = replace_column_values(table, "episode_index", [new_episode] * rows)
    table = replace_column_values(table, "index", list(range(new_global_from, new_global_from + rows)))
    if "task_index" in table.column_names:
        table = replace_column_values(table, "task_index", [task_map[value] for value in original_tasks])
    return table, original_tasks


def add_nested_number(value: Any, offset: int) -> Any:
    if offset == 0:
        return value
    if isinstance(value, list):
        return [add_nested_number(v, offset) for v in value]
    if value is None:
        return None
    return value + offset


def append_columns(target: dict[str, list[Any]], source: dict[str, list[Any]], schema: pa.Schema) -> None:
    if not target:
        for field in schema:
            target[field.name] = []
    for field in schema:
        target[field.name].extend(source[field.name])


def recursive_min(values: list[Any]) -> Any:
    arrs = [np.asarray(v, dtype=float) for v in values]
    return np.minimum.reduce(arrs).tolist()


def recursive_max(values: list[Any]) -> Any:
    arrs = [np.asarray(v, dtype=float) for v in values]
    return np.maximum.reduce(arrs).tolist()


def recursive_weighted_average(values: list[Any], weights: list[float]) -> Any:
    arrs = [np.asarray(v, dtype=float) for v in values]
    total = float(sum(weights))
    if total == 0:
        return arrs[0].tolist()
    acc = np.zeros_like(arrs[0], dtype=float)
    for arr, weight in zip(arrs, weights):
        acc += arr * weight
    return (acc / total).tolist()


def recursive_pooled_std(means: list[Any], stds: list[Any], weights: list[float]) -> Any:
    total = float(sum(weights))
    if total == 0:
        return np.asarray(stds[0], dtype=float).tolist()
    mean_arrs = [np.asarray(v, dtype=float) for v in means]
    std_arrs = [np.asarray(v, dtype=float) for v in stds]
    global_mean = np.asarray(recursive_weighted_average(means, weights), dtype=float)
    second_moment = np.zeros_like(global_mean, dtype=float)
    for mean_arr, std_arr, weight in zip(mean_arrs, std_arrs, weights):
        second_moment += weight * (np.square(std_arr) + np.square(mean_arr))
    variance = np.maximum(second_moment / total - np.square(global_mean), 0.0)
    return np.sqrt(variance).tolist()


def count_weight(value: Any) -> float:
    if isinstance(value, list) and value:
        return float(value[0])
    if value is None:
        return 0.0
    return float(value)


def build_stats_json(episodes_table: pa.Table) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, list[Any]]] = defaultdict(dict)

    for name in episodes_table.column_names:
        if not name.startswith("stats/"):
            continue
        _, feature, stat = name.rsplit("/", 2)
        grouped[feature][stat] = episodes_table[name].to_pylist()

    stats: dict[str, dict[str, Any]] = {}
    for feature, stat_values in grouped.items():
        if "count" not in stat_values:
            continue
        weights = [count_weight(v) for v in stat_values["count"]]
        feature_stats: dict[str, Any] = {}

        if "min" in stat_values:
            feature_stats["min"] = recursive_min(stat_values["min"])
        if "max" in stat_values:
            feature_stats["max"] = recursive_max(stat_values["max"])
        if "mean" in stat_values:
            feature_stats["mean"] = recursive_weighted_average(stat_values["mean"], weights)
        if "std" in stat_values and "mean" in stat_values:
            feature_stats["std"] = recursive_pooled_std(stat_values["mean"], stat_values["std"], weights)
        feature_stats["count"] = [int(sum(weights))]

        for q in ["q01", "q10", "q50", "q90", "q99"]:
            if q in stat_values:
                feature_stats[q] = recursive_weighted_average(stat_values[q], weights)

        stats[feature] = feature_stats

    return stats


def feature_shape(info: dict[str, Any], name: str) -> list[int] | None:
    spec = info.get("features", {}).get(name)
    if not isinstance(spec, dict):
        return None
    shape = spec.get("shape")
    return list(shape) if isinstance(shape, list) else None


def feature_names_by_dtype(info: dict[str, Any], dtype: str) -> list[str]:
    return [
        name
        for name, spec in info.get("features", {}).items()
        if isinstance(spec, dict) and spec.get("dtype") == dtype
    ]


def video_keys_from_info(info: dict[str, Any]) -> list[str]:
    return feature_names_by_dtype(info, "video")


def video_expected_codec(info: dict[str, Any], video_key: str) -> str | None:
    feature = info.get("features", {}).get(video_key)
    if not isinstance(feature, dict):
        return None
    video_info = feature.get("info")
    if not isinstance(video_info, dict):
        return None
    codec = video_info.get("video.codec")
    return str(codec) if codec else None


def data_path_for_episode(root: Path, row: dict[str, Any]) -> Path:
    chunk = int(row["data/chunk_index"])
    file_index = int(row["data/file_index"])
    return root / "data" / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.parquet"


def scalar_list_for_field(value: float, field: pa.Field) -> list[Any]:
    value_type = field.type.value_type if hasattr(field.type, "value_type") else None
    if value_type is not None and pa.types.is_integer(value_type):
        return [int(round(value))]
    return [float(value)]


def update_scalar_feature_stats(
    row: dict[str, Any],
    schema: pa.Schema,
    feature: str,
    values: list[int] | np.ndarray,
) -> None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return
    stat_values = {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "count": float(arr.size),
        "q01": float(np.quantile(arr, 0.01)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "q99": float(np.quantile(arr, 0.99)),
    }
    for stat, value in stat_values.items():
        name = f"stats/{feature}/{stat}"
        idx = schema.get_field_index(name)
        if idx >= 0:
            row[name] = scalar_list_for_field(value, schema.field(idx))


def make_episode_row(
    old_row: dict[str, Any],
    schema: pa.Schema,
    new_episode: int,
    new_global_from: int,
    row_count: int,
    task_map: dict[int, int],
    original_task_values: list[int],
) -> dict[str, Any]:
    row = dict(old_row)
    chunks_size = int(row.pop("_chunks_size", 1000))
    row["episode_index"] = new_episode
    row["length"] = row_count
    row["data/chunk_index"] = chunk_index_for(new_episode, chunks_size)
    row["data/file_index"] = file_index_in_chunk(new_episode, chunks_size)
    row["dataset_from_index"] = new_global_from
    row["dataset_to_index"] = new_global_from + row_count
    if "meta/episodes/chunk_index" in row:
        row["meta/episodes/chunk_index"] = 0
    if "meta/episodes/file_index" in row:
        row["meta/episodes/file_index"] = 0

    update_scalar_feature_stats(row, schema, "episode_index", [new_episode] * row_count)
    update_scalar_feature_stats(row, schema, "index", np.arange(new_global_from, new_global_from + row_count))
    if original_task_values:
        update_scalar_feature_stats(row, schema, "task_index", [task_map[value] for value in original_task_values])
    return row


def update_episode_metadata(
    columns: dict[str, list[Any]],
    schema: pa.Schema,
    episode_offset: int,
    frame_offset: int,
    task_offset: int,
    video_map: dict[tuple[str, int, int], tuple[int, int]],
    video_keys: list[str],
    chunks_size: int,
    meta_file_index: int,
) -> dict[str, list[Any]]:
    row_count = len(columns["episode_index"])
    updated: dict[str, list[Any]] = {name: list(values) for name, values in columns.items()}

    for i in range(row_count):
        old_episode = int(updated["episode_index"][i])
        new_episode = old_episode + episode_offset
        updated["episode_index"][i] = new_episode
        updated["data/chunk_index"][i] = chunk_index_for(new_episode, chunks_size)
        updated["data/file_index"][i] = file_index_in_chunk(new_episode, chunks_size)
        updated["dataset_from_index"][i] = int(updated["dataset_from_index"][i]) + frame_offset
        updated["dataset_to_index"][i] = int(updated["dataset_to_index"][i]) + frame_offset

        if "meta/episodes/chunk_index" in updated:
            updated["meta/episodes/chunk_index"][i] = chunk_index_for(meta_file_index, chunks_size)
        if "meta/episodes/file_index" in updated:
            updated["meta/episodes/file_index"][i] = file_index_in_chunk(meta_file_index, chunks_size)

        for video_key in video_keys:
            chunk_col = f"videos/{video_key}/chunk_index"
            file_col = f"videos/{video_key}/file_index"
            if chunk_col not in updated or file_col not in updated:
                continue
            old_video_chunk = int(columns[chunk_col][i])
            old_video_file = int(columns[file_col][i])
            new_video_chunk, new_video_file = video_map[(video_key, old_video_chunk, old_video_file)]
            updated[chunk_col][i] = new_video_chunk
            updated[file_col][i] = new_video_file

    for field in schema:
        name = field.name
        if not name.startswith("stats/"):
            continue
        _, feature, stat = name.rsplit("/", 2)
        offset_kind = STATS_OFFSET_FEATURES.get(feature)
        if offset_kind is None or stat not in OFFSET_STATS:
            continue
        offset = {
            "episode": episode_offset,
            "frame": frame_offset,
            "task": task_offset,
        }[offset_kind]
        updated[name] = [add_nested_number(v, offset) for v in updated[name]]

    return updated


def validate_merge_source_dataset(path: Path) -> None:
    required = [
        path / "meta" / "info.json",
        path / "meta" / "tasks.parquet",
        path / "meta" / "stats.json",
        path / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        path / "data" / "chunk-000",
    ]
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise FileNotFoundError("missing required dataset files:\n" + "\n".join(missing))


def validate_full_dataset(path: Path) -> None:
    required = [
        path / "meta" / "info.json",
        path / "meta" / "stats.json",
        path / "meta" / "tasks.parquet",
        path / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        path / "data",
        path / "videos",
    ]
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise FileNotFoundError("missing required dataset paths:\n" + "\n".join(missing))
