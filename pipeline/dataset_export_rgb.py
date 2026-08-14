#!/usr/bin/env python3
"""Export a pure-RGB LeRobot v3 dataset from an RGB-D dataset.

The source dataset is not modified. The output drops everything depth-related
so the result matches the standard LeRobot v3.0 layout (lerobot >= 0.4,
including 0.6.x): image-dtype features (the embedded depth PNG columns) are
removed from the data parquets, info.json, episode stats columns, and
stats.json; the depth_sidecar directory and the depth recording manifest are
not copied. Videos and tasks are copied unchanged.

Data parquets are read column-selectively, so the embedded depth bytes are
never loaded; the export reads only the numeric columns plus copies videos.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pyarrow as pa
import pyarrow.parquet as pq

from utils.dataset_io import copy_file, copy_tree, read_json, remove_tree_safely, write_json
from utils.dataset_metadata import build_stats_json, validate_merge_source_dataset

HF_METADATA_KEY = b"huggingface"


def image_keys_from_info(info: dict[str, Any]) -> list[str]:
    return [name for name, spec in info.get("features", {}).items() if spec.get("dtype") == "image"]


def strip_hf_schema_metadata(schema: pa.Schema, dropped_keys: list[str]) -> pa.Schema:
    """Remove dropped features from the embedded huggingface datasets metadata."""
    metadata = schema.metadata or {}
    if HF_METADATA_KEY not in metadata:
        return schema
    try:
        payload = json.loads(metadata[HF_METADATA_KEY].decode("utf-8"))
        features = payload["info"]["features"]
        for key in dropped_keys:
            features.pop(key, None)
        updated = dict(metadata)
        updated[HF_METADATA_KEY] = json.dumps(payload).encode("utf-8")
        return schema.with_metadata(updated)
    except (ValueError, KeyError, TypeError):
        updated = {k: v for k, v in metadata.items() if k != HF_METADATA_KEY}
        return schema.with_metadata(updated) if updated else schema.remove_metadata()


def export_data_parquets(src: Path, out_dir: Path, image_keys: list[str]) -> int:
    files = sorted((src / "data").rglob("*.parquet"))
    for count, src_file in enumerate(files):
        parquet = pq.ParquetFile(src_file)
        keep = [name for name in parquet.schema_arrow.names if name not in image_keys]
        table = parquet.read(columns=keep)
        schema = strip_hf_schema_metadata(table.schema, image_keys)
        table = table.replace_schema_metadata(schema.metadata)
        dst_file = out_dir / src_file.relative_to(src)
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dst_file)
        if count % 100 == 0 or count == len(files) - 1:
            print(f"  data parquet {count + 1}/{len(files)}")
    return len(files)


def export_episodes_meta(src: Path, out_dir: Path, image_keys: list[str]) -> pa.Table:
    src_file = src / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(src_file)
    dropped = [
        name
        for name in table.schema.names
        if any(name.startswith(f"stats/{key}/") for key in image_keys)
    ]
    table = table.drop_columns(dropped)
    dst_file = out_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst_file)
    print(f"  episodes metadata: dropped columns {dropped}")
    return table


def run_export(src: Path, out_dir: Path, overwrite: bool) -> None:
    validate_merge_source_dataset(src)
    if out_dir.resolve() == src.resolve():
        raise ValueError("output must not be the source dataset")
    info = read_json(src / "meta" / "info.json")
    image_keys = image_keys_from_info(info)
    if not image_keys:
        raise ValueError("source has no image-dtype features to strip; it already looks RGB-only")

    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists, rerun with --overwrite: {out_dir}")
        remove_tree_safely(out_dir)
    out_dir.mkdir(parents=True)

    print(f"Source: {src}")
    print(f"Output: {out_dir}")
    print(f"Stripping image features: {', '.join(image_keys)}")

    print("Copying videos...")
    copy_tree(src / "videos", out_dir / "videos", "copy")

    print("Rewriting data parquets without depth columns...")
    data_file_count = export_data_parquets(src, out_dir, image_keys)

    print("Writing metadata...")
    episodes_table = export_episodes_meta(src, out_dir, image_keys)
    copy_file(src / "meta" / "tasks.parquet", out_dir / "meta" / "tasks.parquet", "copy")

    rgb_info = dict(info)
    rgb_info["features"] = {k: v for k, v in info["features"].items() if k not in image_keys}
    write_json(out_dir / "meta" / "info.json", rgb_info)
    write_json(out_dir / "meta" / "stats.json", build_stats_json(episodes_table))

    print("=" * 80)
    print("EXPORT COMPLETE")
    print(f"Output: {out_dir}")
    print(f"Episodes: {rgb_info['total_episodes']}, frames: {rgb_info['total_frames']}")
    print(f"Data parquets rewritten: {data_file_count}")
    print("Depth sidecars and depth manifest intentionally not copied")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a pure-RGB copy of an RGB-D LeRobot dataset.")
    parser.add_argument("--src", type=Path, required=True, help="RGB-D source dataset (unchanged)")
    parser.add_argument("--out-dir", type=Path, required=True, help="output directory for the RGB dataset")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_export(args.src, args.out_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
