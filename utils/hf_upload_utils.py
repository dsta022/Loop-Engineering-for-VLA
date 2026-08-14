#!/usr/bin/env python3
"""Shared Hugging Face upload helpers for LeRobot dataset tools."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from utils.dataset_io import discover_reparse_paths, read_json


IGNORE_PATTERNS = [
    ".git",
    ".git/*",
    "__pycache__",
    "__pycache__/*",
    ".cache",
    ".cache/*",
    "*.pyc",
    "*.tmp",
    "*.part",
    "*.lock",
    "Thumbs.db",
    "desktop.ini",
]


def import_hf_api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub\n"
            "Install it in your conda environment first:\n"
            "  conda activate vla_data_check\n"
            "  conda install -c conda-forge huggingface_hub hf-xet -y\n"
            "or:\n"
            "  pip install -U huggingface_hub hf_xet"
        ) from exc
    return HfApi


def require_supported_call(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs supported by the installed huggingface_hub version."""
    signature = inspect.signature(fn)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def validate_lerobot_dataset_for_upload(dataset_dir: Path, strict_no_reparse: bool) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {dataset_dir}")

    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"dataset is incomplete; missing:\n{info_path}")
    info = read_json(info_path)

    required = [
        dataset_dir / "meta" / "stats.json",
        dataset_dir / "meta" / "tasks.parquet",
        dataset_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        dataset_dir / "data",
        dataset_dir / "videos",
    ]
    # depth sidecars only exist for datasets that declare image-dtype (depth) features;
    # RGB-only LeRobot datasets have no depth_sidecar directory
    has_image_features = any(
        isinstance(spec, dict) and spec.get("dtype") == "image" for spec in info.get("features", {}).values()
    )
    if has_image_features:
        required.append(dataset_dir / "depth_sidecar")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("dataset is incomplete; missing:\n" + "\n".join(missing))

    if strict_no_reparse:
        reparse_paths = discover_reparse_paths(dataset_dir)
        if reparse_paths:
            sample = "\n".join(str(path) for path in reparse_paths[:20])
            raise RuntimeError(
                "dataset contains symlink/junction/reparse paths; refusing to upload non-real files:\n" + sample
            )

    for key in ["total_episodes", "total_frames", "total_tasks", "chunks_size", "features"]:
        if key not in info:
            raise ValueError(f"meta/info.json is missing key: {key}")
    return info


def count_files_and_bytes(dataset_dir: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    for path in dataset_dir.rglob("*"):
        if path.is_file():
            file_count += 1
            total_bytes += path.stat().st_size
    return file_count, total_bytes


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def upload_with_large_folder(
    api: Any,
    repo_id: str,
    dataset_dir: Path,
    token: str,
    revision: str,
    private: bool,
    num_workers: int,
) -> None:
    kwargs = {
        "repo_id": repo_id,
        "folder_path": str(dataset_dir),
        "repo_type": "dataset",
        "revision": revision,
        "private": private,
        "token": token,
        "ignore_patterns": IGNORE_PATTERNS,
        "num_workers": num_workers,
        "print_report": True,
        "print_report_every": 60,
    }
    api.upload_large_folder(**require_supported_call(api.upload_large_folder, kwargs))


def upload_with_folder(
    api: Any,
    repo_id: str,
    dataset_dir: Path,
    token: str,
    revision: str,
    commit_message: str,
) -> None:
    kwargs = {
        "repo_id": repo_id,
        "folder_path": str(dataset_dir),
        "repo_type": "dataset",
        "revision": revision,
        "token": token,
        "ignore_patterns": IGNORE_PATTERNS,
        "commit_message": commit_message,
    }
    api.upload_folder(**require_supported_call(api.upload_folder, kwargs))
