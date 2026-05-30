#!/usr/bin/env python3
"""Shared file-system helpers for LeRobot dataset tools."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterator


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def is_reparse_path(path: Path) -> bool:
    if path.is_symlink():
        return True
    checker = getattr(os.path, "isjunction", None)
    return bool(checker(path)) if checker is not None else False


def iter_tree_no_reparse(root: Path) -> Iterator[Path]:
    for current, dirs, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        kept_dirs = []
        for dirname in dirs:
            path = current_path / dirname
            yield path
            if not is_reparse_path(path):
                kept_dirs.append(dirname)
        dirs[:] = kept_dirs
        for filename in files:
            yield current_path / filename


def discover_reparse_paths(root: Path) -> list[Path]:
    return [path for path in iter_tree_no_reparse(root) if is_reparse_path(path)]


def safe_rel_path(value: Any) -> Path | None:
    if value is None:
        return None
    rel = Path(str(value))
    if rel.is_absolute() or ".." in rel.parts:
        return Path(rel.name)
    return rel


def find_env_executable(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for candidate in [
        Path(sys.prefix) / f"{name}.exe",
        Path(sys.prefix) / "Library" / "bin" / f"{name}.exe",
        Path(sys.prefix) / "Scripts" / f"{name}.exe",
    ]:
        if candidate.exists():
            return str(candidate)
    return None


def copy_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return
    raise ValueError(f"unsupported copy mode: {mode}")


def robocopy_tree(src_dir: Path, dst_dir: Path) -> bool:
    if os.name != "nt":
        return False
    result = shutil.which("robocopy")
    if result is None:
        return False
    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        result,
        str(src_dir),
        str(dst_dir),
        "/E",
        "/MT:32",
        "/R:2",
        "/W:1",
        "/COPY:DAT",
        "/DCOPY:DAT",
        "/NFL",
        "/NDL",
        "/NJH",
        "/NJS",
        "/NP",
    ]
    completed = os.spawnv(os.P_WAIT, result, cmd)
    if completed > 7:
        raise RuntimeError(f"robocopy failed with exit code {completed}: {src_dir} -> {dst_dir}")
    return True


def copy_tree(src_dir: Path, dst_dir: Path, mode: str) -> None:
    if dst_dir.exists():
        raise FileExistsError(f"target already exists: {dst_dir}")
    if mode == "copy" and robocopy_tree(src_dir, dst_dir):
        return
    dst_dir.mkdir(parents=True)
    for src_file in src_dir.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src_dir)
        copy_file(src_file, dst_dir / rel, mode)


def count_files(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file())


def remove_tree_safely(path: Path) -> None:
    """Remove a tree without following Windows junctions or symlinks."""
    if path.is_symlink() or is_reparse_path(path):
        path.rmdir() if path.is_dir() else path.unlink()
        return
    if path.is_dir():
        for current, dirs, files in os.walk(path, topdown=True):
            keep_dirs = []
            for dirname in dirs:
                child = Path(current) / dirname
                if child.is_symlink() or is_reparse_path(child):
                    child.rmdir()
                else:
                    keep_dirs.append(dirname)
            dirs[:] = keep_dirs
            for filename in files:
                child = Path(current) / filename
                if child.is_symlink():
                    child.unlink()
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def prepare_output_dir(source: Path, out_dir: Path, overwrite: bool) -> None:
    source_resolved = source.resolve()
    out_resolved = out_dir.resolve() if out_dir.exists() else out_dir.absolute()
    if out_resolved == source_resolved:
        raise ValueError("output dataset must not be the source dataset")
    try:
        out_resolved.relative_to(source_resolved)
    except ValueError:
        pass
    else:
        raise ValueError("output dataset must not be inside the source dataset")

    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists, rerun with --overwrite: {out_dir}")
        remove_tree_safely(out_dir)
    out_dir.mkdir(parents=True)
