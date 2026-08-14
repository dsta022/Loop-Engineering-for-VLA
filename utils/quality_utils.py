#!/usr/bin/env python3
"""Shared quality-audit helpers for LeRobot dataset tools."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa


DEFAULT_REPORT_DIR = "quality_audit_reports"


@dataclass
class EpisodeQuality:
    episode_index: int
    length: int = 0
    data_rows: int = 0
    depth_files: int = 0
    structural_status: str = "keep"
    semantic_status: str = "not_checked"
    semantic_evaluator: str = ""
    semantic_task_profile: str = ""
    semantic_review_threshold: float | None = None
    semantic_pass_threshold: float | None = None
    semantic_progress_curve: list[float] = field(default_factory=list)
    semantic_progress_final: float | None = None
    semantic_success_probability: float | None = None
    semantic_score: float | None = None
    semantic_failure_stage: str = ""
    semantic_progress_max_drop: float | None = None
    semantic_progress_idle_tail_ratio: float | None = None
    semantic_stage_timeline_source: str = ""
    semantic_motion_verbs: str = ""
    semantic_motion_gripper_transitions: int | None = None
    semantic_score_uncertainty: float | None = None
    semantic_view_disagreement: float | None = None
    semantic_counterfactual_margin: float | None = None
    status: str = "keep"
    drop_reasons: list[str] = field(default_factory=list)
    review_reasons: list[str] = field(default_factory=list)
    not_checked: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        if reason not in self.drop_reasons:
            self.drop_reasons.append(reason)
        self.status = "drop"

    def review(self, reason: str) -> None:
        if reason not in self.review_reasons:
            self.review_reasons.append(reason)
        if self.status != "drop":
            self.status = "review"

    def skip(self, reason: str) -> None:
        if reason not in self.not_checked:
            self.not_checked.append(reason)


def default_audit_output_dir(dataset: Path) -> Path:
    return dataset.parent / DEFAULT_REPORT_DIR / dataset.name


def default_drop_list(dataset: Path) -> Path:
    return dataset.parent / DEFAULT_REPORT_DIR / dataset.name / "drop_episodes.txt"


def build_episode_columns(video_keys: list[str]) -> list[str]:
    columns = [
        "episode_index",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    ]
    for video_key in video_keys:
        columns.extend(
            [
                f"videos/{video_key}/chunk_index",
                f"videos/{video_key}/file_index",
                f"videos/{video_key}/from_timestamp",
                f"videos/{video_key}/to_timestamp",
            ]
        )
    return columns


def column_matrix(table: pa.Table, name: str, dtype: Any) -> np.ndarray | None:
    if name not in table.column_names:
        return None
    try:
        values = np.asarray(table[name].to_pylist(), dtype=dtype)
    except Exception:
        return None
    return values


def hash_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if length <= count:
        return list(range(length))
    return sorted({int(round(v)) for v in np.linspace(0, length - 1, count)})


def robust_upper_threshold(values: list[float], min_floor: float = 0.0) -> float | None:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size < 5:
        return None
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    q99 = float(np.quantile(arr, 0.99))
    threshold = max(q99, median + 20.0 * max(mad, 1e-12), min_floor)
    return threshold


def parse_episode_list(path: Path) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"drop list not found: {path}")
    values: list[int] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.lstrip("\ufeff").split("#", 1)[0].replace(",", " ").strip()
        if not line:
            continue
        for token in line.split():
            try:
                values.append(int(token))
            except ValueError as exc:
                raise ValueError(f"invalid episode id in {path}:{line_no}: {token!r}") from exc
    return sorted(set(values))


def load_quality_rows(drop_list: Path) -> dict[int, dict[str, str]]:
    report_path = drop_list.parent / "quality_report.csv"
    if not report_path.exists():
        return {}
    rows: dict[int, dict[str, str]] = {}
    with report_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episode_text = row.get("episode_index", "")
            if episode_text == "":
                continue
            rows[int(episode_text)] = dict(row)
    return rows


def write_lines(path: Path, values: list[int]) -> None:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def write_quality_outputs(out_dir: Path, audits: list[EpisodeQuality], summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "episode_index",
        "status",
        "structural_status",
        "semantic_status",
        "semantic_evaluator",
        "semantic_task_profile",
        "semantic_review_threshold",
        "semantic_pass_threshold",
        "semantic_progress_curve_json",
        "semantic_progress_final",
        "semantic_success_probability",
        "semantic_score",
        "semantic_failure_stage",
        "semantic_progress_max_drop",
        "semantic_progress_idle_tail_ratio",
        "semantic_stage_timeline_source",
        "semantic_motion_verbs",
        "semantic_motion_gripper_transitions",
        "semantic_score_uncertainty",
        "semantic_view_disagreement",
        "semantic_counterfactual_margin",
        "length",
        "data_rows",
        "depth_files",
        "drop_reasons",
        "review_reasons",
        "not_checked",
        "metrics_json",
    ]
    with (out_dir / "quality_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for audit in audits:
            writer.writerow(
                {
                    "structural_status": audit.structural_status,
                    "semantic_status": audit.semantic_status,
                    "semantic_evaluator": audit.semantic_evaluator,
                    "semantic_task_profile": audit.semantic_task_profile,
                    "semantic_review_threshold": (
                        "" if audit.semantic_review_threshold is None else audit.semantic_review_threshold
                    ),
                    "semantic_pass_threshold": (
                        "" if audit.semantic_pass_threshold is None else audit.semantic_pass_threshold
                    ),
                    "semantic_progress_curve_json": json.dumps(audit.semantic_progress_curve),
                    "semantic_progress_final": (
                        "" if audit.semantic_progress_final is None else audit.semantic_progress_final
                    ),
                    "semantic_success_probability": (
                        ""
                        if audit.semantic_success_probability is None
                        else audit.semantic_success_probability
                    ),
                    "semantic_score": "" if audit.semantic_score is None else audit.semantic_score,
                    "semantic_failure_stage": audit.semantic_failure_stage,
                    "semantic_progress_max_drop": (
                        "" if audit.semantic_progress_max_drop is None else audit.semantic_progress_max_drop
                    ),
                    "semantic_progress_idle_tail_ratio": (
                        ""
                        if audit.semantic_progress_idle_tail_ratio is None
                        else audit.semantic_progress_idle_tail_ratio
                    ),
                    "semantic_stage_timeline_source": audit.semantic_stage_timeline_source,
                    "semantic_motion_verbs": audit.semantic_motion_verbs,
                    "semantic_motion_gripper_transitions": (
                        ""
                        if audit.semantic_motion_gripper_transitions is None
                        else audit.semantic_motion_gripper_transitions
                    ),
                    "semantic_score_uncertainty": (
                        ""
                        if audit.semantic_score_uncertainty is None
                        else audit.semantic_score_uncertainty
                    ),
                    "semantic_view_disagreement": (
                        ""
                        if audit.semantic_view_disagreement is None
                        else audit.semantic_view_disagreement
                    ),
                    "semantic_counterfactual_margin": (
                        ""
                        if audit.semantic_counterfactual_margin is None
                        else audit.semantic_counterfactual_margin
                    ),
                    "episode_index": audit.episode_index,
                    "status": audit.status,
                    "length": audit.length,
                    "data_rows": audit.data_rows,
                    "depth_files": audit.depth_files,
                    "drop_reasons": ";".join(audit.drop_reasons),
                    "review_reasons": ";".join(audit.review_reasons),
                    "not_checked": ";".join(audit.not_checked),
                    "metrics_json": json.dumps(audit.metrics, ensure_ascii=False, sort_keys=True),
                }
            )
    keep = [a.episode_index for a in audits if a.status == "keep"]
    review = [a.episode_index for a in audits if a.status == "review"]
    drop = [a.episode_index for a in audits if a.status == "drop"]
    write_lines(out_dir / "keep_episodes.txt", keep)
    write_lines(out_dir / "review_episodes.txt", review)
    write_lines(out_dir / "drop_episodes.txt", drop)
    (out_dir / "quality_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
