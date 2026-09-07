#!/usr/bin/env python3
"""Generate validated language sidecars for quality-approved robot trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import sys

# Allow `python ./pipeline/trajectory_language_enrichment.py` to run as a plain script: the repo-root
# packages must be importable without installing the project first.
if __package__ in (None, ""):
    _REPO_ROOT = str(Path(__file__).resolve().parents[1])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from pipeline.data_quality_audit import (
    AuditConfig,
    DatasetMetadata,
    EpisodeRecord,
    episode_motion,
    load_metadata,
    read_episode_data,
    semantic_tasks_for_episode,
    semantic_videos_for_episode,
)
from utils.dataset_io import prepare_output_dir
from utils.semantic_consistency import (
    boundary_alignment_findings,
    channel_traces,
    kinematic_change_points,
    select_gripper_index,
    subtask_boundary_alignment,
)
from utils.language_annotation_profile import (
    LanguageAnnotationEpisode,
    TrajectoryLanguageAnnotation,
    TrajectoryLanguageAnnotator,
    annotator_name,
    load_annotator,
    load_annotator_config,
    normalize_annotation_result,
)
from utils.quality_utils import EpisodeQuality, default_audit_output_dir
from utils.task_semantic_profile import TaskProfileRegistry, TaskSemanticProfile


DEFAULT_OUTPUT_ROOT = "language_enrichment_outputs"


@dataclass(frozen=True)
class LanguageEnrichmentConfig:
    dataset_root: Path
    quality_report: Path
    out_dir: Path
    annotator: TrajectoryLanguageAnnotator
    task_profiles: TaskProfileRegistry | None = None
    approved_episodes: Path | None = None
    overwrite: bool = False
    boundary_check: str = "kinematic"
    motion_gripper_index: int | None = None
    boundary_tolerance_frames: int = 5

    def __post_init__(self) -> None:
        if self.boundary_check not in {"none", "kinematic"}:
            raise ValueError("boundary check must be one of: none, kinematic")
        if self.boundary_tolerance_frames < 0:
            raise ValueError("boundary tolerance must be non-negative")


def default_enrichment_output_dir(dataset: Path) -> Path:
    return dataset.parent / DEFAULT_OUTPUT_ROOT / dataset.name


def resolve_quality_report(path: Path) -> Path:
    return path / "quality_report.csv" if path.is_dir() else path


def load_quality_rows(path: Path) -> dict[int, dict[str, str]]:
    report_path = resolve_quality_report(path)
    if not report_path.is_file():
        raise FileNotFoundError(f"quality report not found: {report_path}")
    with report_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"episode_index", "status"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("quality report must contain episode_index and status columns")
        rows: dict[int, dict[str, str]] = {}
        for raw_row in reader:
            episode_index = int(raw_row["episode_index"])
            if episode_index in rows:
                raise ValueError(f"duplicate episode_index in quality report: {episode_index}")
            rows[episode_index] = dict(raw_row)
    if not rows:
        raise ValueError("quality report contains no episodes")
    return rows


def verify_quality_report_dataset(report_path: Path, dataset_root: Path) -> bool:
    summary_path = report_path.parent / "quality_summary.json"
    if not summary_path.exists():
        return False
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not str(payload.get("dataset", "")).strip():
        raise ValueError(f"quality summary does not identify its dataset: {summary_path}")
    reported_dataset = Path(str(payload["dataset"])).expanduser().resolve()
    if reported_dataset != dataset_root.resolve():
        raise ValueError(
            "quality report belongs to a different dataset: "
            f"{reported_dataset} != {dataset_root.resolve()}"
        )
    return True


def load_episode_list(path: Path) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(f"approved episode list not found: {path}")
    episodes: list[int] = []
    seen: set[int] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = raw_line.partition("#")[0].strip()
        if not value:
            continue
        try:
            episode_index = int(value)
        except ValueError as exc:
            raise ValueError(f"invalid episode index at {path}:{line_number}: {value!r}") from exc
        if episode_index < 0:
            raise ValueError(f"episode index must be non-negative at {path}:{line_number}")
        if episode_index not in seen:
            episodes.append(episode_index)
            seen.add(episode_index)
    if not episodes:
        raise ValueError(f"approved episode list is empty: {path}")
    return episodes


def select_episode_indices(
    quality_rows: dict[int, dict[str, str]], approved_episodes: Path | None
) -> tuple[list[int], str]:
    if approved_episodes is None:
        selected = sorted(
            episode_index
            for episode_index, row in quality_rows.items()
            if row.get("status", "").strip().casefold() == "keep"
        )
        if not selected:
            raise ValueError("quality report contains no keep episodes to enrich")
        return selected, "quality_report_keep"

    selected = load_episode_list(approved_episodes)
    missing = [episode for episode in selected if episode not in quality_rows]
    if missing:
        raise ValueError(f"approved episodes are absent from quality report: {missing}")
    dropped = [
        episode
        for episode in selected
        if quality_rows[episode].get("status", "").strip().casefold() == "drop"
    ]
    if dropped:
        raise ValueError(f"drop episodes cannot be language-enriched: {dropped}")
    return selected, "explicit_approved_episode_list"


def _optional_float(value: str | None) -> float | None:
    text = str(value or "").strip()
    return None if not text else float(text)


def semantic_context_from_quality_row(row: dict[str, str]) -> dict[str, Any]:
    progress_text = str(row.get("semantic_progress_curve_json") or "").strip()
    metrics_text = str(row.get("metrics_json") or "").strip()
    return {
        "quality_status": row.get("status", ""),
        "structural_status": row.get("structural_status", ""),
        "semantic_status": row.get("semantic_status", ""),
        "semantic_evaluator": row.get("semantic_evaluator", ""),
        "semantic_score": _optional_float(row.get("semantic_score")),
        "semantic_success_probability": _optional_float(
            row.get("semantic_success_probability")
        ),
        "semantic_failure_stage": row.get("semantic_failure_stage", ""),
        "semantic_progress_curve": json.loads(progress_text) if progress_text else [],
        "review_reasons": [
            item for item in str(row.get("review_reasons") or "").split(";") if item
        ],
        "metrics": json.loads(metrics_text) if metrics_text else {},
    }


def _metadata_config(config: LanguageEnrichmentConfig) -> AuditConfig:
    return AuditConfig(
        root=config.dataset_root,
        out_dir=config.out_dir,
        depth_check="none",
        video_check="metadata",
        video_content_check="none",
        sample_frames=0,
        depth_sample_frames=0,
        duplicate_check="none",
        video_sample_stride=30,
        motion_check="none",
        motion_gripper_index=config.motion_gripper_index,
    )


def subtask_boundaries(annotation: TrajectoryLanguageAnnotation) -> tuple[int, ...]:
    """Interior frames where the annotation claims one subtask hands off to the next."""
    subtasks = annotation.subtasks
    if len(subtasks) < 2:
        return ()
    frames: set[int] = set()
    for previous, current in zip(subtasks, subtasks[1:]):
        frames.add(current.start_frame)
        frames.add(previous.end_frame + 1)
    return tuple(sorted(frame for frame in frames if 0 < frame < annotation.length))


def evaluate_boundary_alignment(
    config: LanguageEnrichmentConfig,
    episode: LanguageAnnotationEpisode,
    annotation: TrajectoryLanguageAnnotation,
) -> dict[str, Any]:
    """Check annotated subtask boundaries against real changes in arm motion."""
    row: dict[str, Any] = {
        "episode_index": annotation.episode_index,
        "status": "not_checked",
        "reasons": [],
    }
    if config.boundary_check == "none":
        row["status"] = "disabled"
        return row
    if episode.motion is None or episode.motion.action is None:
        row["status"] = "action_unavailable"
        return row
    boundaries = subtask_boundaries(annotation)
    if not boundaries:
        row["status"] = "no_interior_boundary"
        return row
    motion = episode.motion
    if motion.gripper_index is None:
        detected = select_gripper_index({episode.episode_index: channel_traces(motion.action)})
        if detected is not None:
            motion = replace(motion, gripper_index=detected)
    row["gripper_index"] = motion.gripper_index
    change_points = kinematic_change_points(motion)
    alignment = subtask_boundary_alignment(
        boundaries, change_points, config.boundary_tolerance_frames
    )
    if alignment is None:
        row["status"] = "no_change_point"
        return row
    reasons = boundary_alignment_findings(alignment)
    row.update(alignment.to_dict())
    row["status"] = "review" if reasons else "aligned"
    row["reasons"] = list(reasons)
    return row


def build_annotation_episode(
    config: LanguageEnrichmentConfig,
    metadata_config: AuditConfig,
    metadata: DatasetMetadata,
    rows_by_episode: dict[int, int],
    quality_row: dict[str, str],
    episode_index: int,
) -> tuple[LanguageAnnotationEpisode, str]:
    row = rows_by_episode.get(episode_index)
    if row is None:
        raise ValueError(f"episode is absent from dataset metadata: {episode_index}")
    record = EpisodeRecord.from_columns(row, metadata.episode_cols)
    reported_length = str(quality_row.get("length") or "").strip()
    if reported_length and int(reported_length) != record.length:
        raise ValueError(
            f"quality report length mismatch for episode {episode_index}: "
            f"{reported_length}!={record.length}"
        )
    audit = EpisodeQuality(episode_index=episode_index, length=record.length)
    data = read_episode_data(config.dataset_root, record, metadata.depth_keys, audit)
    if data is None or audit.drop_reasons:
        raise ValueError(f"episode data cannot be read: {audit.drop_reasons}")
    tasks = semantic_tasks_for_episode(metadata, data)
    if len(tasks) != 1:
        raise ValueError(
            f"episode {episode_index} requires exactly one task for language enrichment; got {tasks}"
        )
    profile: TaskSemanticProfile | None = None
    if config.task_profiles is not None:
        profile = config.task_profiles.match_any(tasks)
        if profile is None:
            raise ValueError(f"no Task Profile matches episode {episode_index}: {tasks[0]!r}")
    videos = semantic_videos_for_episode(metadata_config, metadata, record, profile)
    missing_videos = [str(video.path) for video in videos if not video.path.is_file()]
    if not videos or missing_videos:
        raise FileNotFoundError(f"episode videos are missing: {missing_videos}")
    episode = LanguageAnnotationEpisode(
        dataset_root=config.dataset_root,
        episode_index=episode_index,
        length=record.length,
        tasks=tasks,
        videos=videos,
        task_profile=profile.to_dict() if profile is not None else None,
        semantic_context=semantic_context_from_quality_row(quality_row),
        motion=episode_motion(metadata_config, metadata, data),
    )
    return episode, profile.task_key if profile is not None else ""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_enrichment_outputs(
    out_dir: Path,
    annotations: list[TrajectoryLanguageAnnotation],
    manifest: dict[str, Any],
    boundary_rows: list[dict[str, Any]] | None = None,
) -> None:
    annotation_rows: list[dict[str, Any]] = []
    subtask_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    description_rows: list[dict[str, Any]] = []
    for annotation in annotations:
        payload = annotation.to_dict()
        annotation.save(
            out_dir / "annotations" / f"episode-{annotation.episode_index:06d}.json"
        )
        annotation_rows.append(payload)
        subtask_rows.extend(
            {"episode_index": annotation.episode_index, **item.to_dict()}
            for item in annotation.subtasks
        )
        event_rows.extend(
            {"episode_index": annotation.episode_index, **item.to_dict()}
            for item in annotation.events
        )
        description_rows.extend(
            {"episode_index": annotation.episode_index, **item.to_dict()}
            for item in annotation.state_descriptions
        )
    _write_jsonl(out_dir / "language_annotations.jsonl", annotation_rows)
    _write_jsonl(out_dir / "subtasks.jsonl", subtask_rows)
    _write_jsonl(out_dir / "events.jsonl", event_rows)
    _write_jsonl(out_dir / "state_descriptions.jsonl", description_rows)
    _write_jsonl(out_dir / "subtask_boundary_alignment.jsonl", boundary_rows or [])
    _write_json(out_dir / "enrichment_manifest.json", manifest)


def run_language_enrichment(config: LanguageEnrichmentConfig) -> int:
    dataset_root = config.dataset_root.resolve()
    quality_report = resolve_quality_report(config.quality_report).resolve()
    quality_dataset_verified = verify_quality_report_dataset(quality_report, dataset_root)
    quality_rows = load_quality_rows(quality_report)
    selected, selection_mode = select_episode_indices(quality_rows, config.approved_episodes)
    metadata_config = _metadata_config(config)
    metadata = load_metadata(metadata_config)
    rows_by_episode = {
        int(value): row for row, value in enumerate(metadata.episode_cols["episode_index"])
    }
    prepare_output_dir(dataset_root, config.out_dir, config.overwrite)

    source = annotator_name(config.annotator)
    annotations: list[TrajectoryLanguageAnnotation] = []
    boundary_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for episode_index in selected:
        try:
            episode, task_profile = build_annotation_episode(
                config,
                metadata_config,
                metadata,
                rows_by_episode,
                quality_rows[episode_index],
                episode_index,
            )
            result = config.annotator.annotate(episode)
            annotation = normalize_annotation_result(result, episode, source, task_profile)
            annotations.append(annotation)
            boundary_rows.append(evaluate_boundary_alignment(config, episode, annotation))
        except Exception as exc:
            errors.append(
                {
                    "episode_index": episode_index,
                    "error": f"{type(exc).__name__}:{str(exc)[:1000]}",
                }
            )

    manifest = {
        "schema_version": 1,
        "dataset": str(dataset_root),
        "quality_report": str(quality_report),
        "quality_report_dataset_verified": quality_dataset_verified,
        "selection_mode": selection_mode,
        "approved_episode_list": (
            str(config.approved_episodes.resolve())
            if config.approved_episodes is not None
            else None
        ),
        "annotator": source,
        "task_profiles_enabled": config.task_profiles is not None,
        "requested_episodes": selected,
        "annotated_episodes": [item.episode_index for item in annotations],
        "errors": errors,
        "status": "complete" if not errors else "partial_failure",
        "source_dataset_modified": False,
        "subtask_boundary_alignment": {
            "check": config.boundary_check,
            "tolerance_frames": config.boundary_tolerance_frames,
            "checked": sum(row["status"] in {"aligned", "review"} for row in boundary_rows),
            "review": sum(row["status"] == "review" for row in boundary_rows),
            "review_episodes": [
                row["episode_index"] for row in boundary_rows if row["status"] == "review"
            ],
        },
    }
    write_enrichment_outputs(config.out_dir, annotations, manifest, boundary_rows)
    unaligned = manifest["subtask_boundary_alignment"]["review"]
    print(f"Dataset: {dataset_root}")
    print(f"Quality report: {quality_report}")
    print(f"Selection: {selection_mode} requested={len(selected)}")
    print(f"Annotated: {len(annotations)} errors={len(errors)}")
    print(f"Boundary alignment: review={unaligned}")
    print(f"Output: {config.out_dir}")
    for error in errors[:20]:
        print(f"ERROR episode={error['episode_index']} {error['error']}")
    return 1 if errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate validated language sidecars for approved robot trajectories."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=None,
        help="quality_report.csv or its directory; defaults to the dataset audit output",
    )
    parser.add_argument(
        "--approved-episodes",
        type=Path,
        default=None,
        help="optional human-approved episode list; review is allowed but drop is always rejected",
    )
    parser.add_argument(
        "--task-profile-dir",
        type=Path,
        default=None,
        help="optional directory containing Task Profile JSON files",
    )
    parser.add_argument(
        "--annotator",
        required=True,
        help="language annotator plugin using module:attribute syntax",
    )
    parser.add_argument(
        "--annotator-config",
        type=Path,
        default=None,
        help="optional JSON configuration passed to the annotator plugin",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--boundary-check",
        choices=["none", "kinematic"],
        default="kinematic",
        help="check subtask boundaries against gripper events and arm near-stops",
    )
    parser.add_argument(
        "--motion-gripper-index",
        type=int,
        default=None,
        help="action dimension holding the gripper, used by the boundary check",
    )
    parser.add_argument(
        "--boundary-tolerance-frames",
        type=int,
        default=5,
        help="how far a boundary may sit from a change point and still count as aligned",
    )
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    quality_report = args.quality_report or (
        default_audit_output_dir(dataset) / "quality_report.csv"
    )
    task_profiles = (
        TaskProfileRegistry.from_directory(args.task_profile_dir)
        if args.task_profile_dir is not None
        else None
    )
    annotator = load_annotator(
        args.annotator, load_annotator_config(args.annotator_config)
    )
    raise SystemExit(
        run_language_enrichment(
            LanguageEnrichmentConfig(
                dataset_root=dataset,
                quality_report=quality_report,
                out_dir=args.out_dir or default_enrichment_output_dir(dataset),
                annotator=annotator,
                task_profiles=task_profiles,
                approved_episodes=args.approved_episodes,
                overwrite=args.overwrite,
                boundary_check=args.boundary_check,
                motion_gripper_index=args.motion_gripper_index,
                boundary_tolerance_frames=args.boundary_tolerance_frames,
            )
        )
    )


if __name__ == "__main__":
    main()
