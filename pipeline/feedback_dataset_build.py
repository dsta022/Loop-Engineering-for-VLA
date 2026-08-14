#!/usr/bin/env python3
"""Build a train-ready dataset from reviewed human recovery/correction episodes."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.dataset_clean_build import build_clean_dataset
from pipeline.dataset_merge import run_merge_paths
from policy_improvement.contracts import ControlPhase, FeedbackSidecar
from utils.dataset_io import read_json
from utils.quality_utils import parse_episode_list


@dataclass(frozen=True)
class FeedbackBuildConfig:
    base_datasets: tuple[Path, ...]
    feedback_dataset: Path
    feedback_sidecar: Path
    quality_report: Path
    out_dir: Path
    approved_episodes: Path | None = None
    language_enrichment: Path | None = None
    copy_mode: str = "copy"
    overwrite: bool = False
    recovery_weight: float = 1.0
    correction_weight: float = 2.0


def load_quality_report(path: Path) -> dict[int, dict[str, str]]:
    report = path / "quality_report.csv" if path.is_dir() else path
    if not report.exists():
        raise FileNotFoundError(f"quality report not found: {report}")
    with report.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "episode_index" not in rows[0] or "status" not in rows[0]:
        raise ValueError("quality report must contain episode_index and status columns")
    result: dict[int, dict[str, str]] = {}
    for row in rows:
        episode_index = int(row["episode_index"])
        if episode_index in result:
            raise ValueError(f"duplicate episode in quality report: {episode_index}")
        result[episode_index] = dict(row)
    return result


def verify_quality_report(path: Path, dataset: Path) -> None:
    report = path / "quality_report.csv" if path.is_dir() else path
    summary_path = report.parent / "quality_summary.json"
    if not summary_path.exists():
        return
    summary = read_json(summary_path)
    reported = summary.get("dataset")
    if reported and Path(str(reported)).resolve() != dataset.resolve():
        raise ValueError("quality report belongs to a different feedback dataset")


def select_feedback_episodes(
    rows: dict[int, dict[str, str]], approved_episodes: Path | None
) -> tuple[list[int], str]:
    if approved_episodes is None:
        selected = sorted(
            episode
            for episode, row in rows.items()
            if row.get("status", "").strip().casefold() == "keep"
        )
        mode = "quality_report_keep"
    else:
        selected = sorted(parse_episode_list(approved_episodes))
        mode = "explicit_human_approval"
        missing = sorted(set(selected).difference(rows))
        if missing:
            raise ValueError(f"approved feedback episodes are absent from report: {missing}")
        dropped = [
            episode
            for episode in selected
            if rows[episode].get("status", "").strip().casefold() == "drop"
        ]
        if dropped:
            raise ValueError(f"drop episodes cannot enter a feedback training set: {dropped}")
    if not selected:
        raise ValueError("no reviewed feedback episodes were selected")
    return selected, mode


def verify_feedback_sidecar(
    sidecar: FeedbackSidecar,
    feedback_dataset: Path,
    selected: list[int],
    *,
    explicit_human_approval: bool,
) -> None:
    reported = sidecar.manifest.get("source_dataset")
    if reported and Path(str(reported)).resolve() != feedback_dataset.resolve():
        raise ValueError("feedback sidecar belongs to a different dataset")
    episode_map = {episode.episode_index: episode for episode in sidecar.episodes}
    frames_by_episode = sidecar.frames_by_episode()
    missing = sorted(set(selected).difference(episode_map))
    if missing:
        raise ValueError(f"selected episodes are missing feedback metadata: {missing}")
    unsafe: list[int] = []
    without_correction: list[int] = []
    not_approved: list[int] = []
    rejected: list[int] = []
    for episode_index in selected:
        review_status = str(
            episode_map[episode_index].review_status.value
            if hasattr(episode_map[episode_index].review_status, "value")
            else episode_map[episode_index].review_status
        )
        if review_status == "rejected":
            rejected.append(episode_index)
        elif review_status != "approved" and not explicit_human_approval:
            not_approved.append(episode_index)
        frames = frames_by_episode.get(episode_index, ())
        if not frames or any(not frame.training_eligible for frame in frames):
            unsafe.append(episode_index)
        if not any(ControlPhase(frame.phase) == ControlPhase.CORRECTION for frame in frames):
            without_correction.append(episode_index)
    if unsafe:
        raise ValueError(
            "standard feedback build accepts corrections-only episodes; autonomous/event frames "
            f"must be segmented first: {unsafe}"
        )
    if without_correction:
        raise ValueError(f"feedback episodes contain no correction frames: {without_correction}")
    if rejected:
        raise ValueError(f"human-rejected feedback episodes cannot enter training: {rejected}")
    if not_approved:
        raise ValueError(
            "feedback episodes require sidecar approval or --approved-episodes: "
            f"{not_approved}"
        )


def dataset_episode_lengths(dataset: Path) -> list[int]:
    table = pq.read_table(
        dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        columns=["episode_index", "length"],
    )
    rows = sorted(
        zip(table["episode_index"].to_pylist(), table["length"].to_pylist()),
        key=lambda item: int(item[0]),
    )
    return [int(length) for _, length in rows]


def write_training_weights(
    out_dir: Path,
    base_datasets: tuple[Path, ...],
    selected: list[int],
    sidecar: FeedbackSidecar,
    recovery_weight: float,
    correction_weight: float,
) -> None:
    if recovery_weight <= 0 or correction_weight <= 0:
        raise ValueError("training weights must be positive")
    rows: list[dict[str, Any]] = []
    output_episode = 0
    for dataset in base_datasets:
        for length in dataset_episode_lengths(dataset):
            rows.extend(
                {
                    "episode_index": output_episode,
                    "frame_index": frame_index,
                    "weight": 1.0,
                    "source": "base_demonstration",
                    "phase": "expert_demonstration",
                }
                for frame_index in range(length)
            )
            output_episode += 1
    frames_by_episode = sidecar.frames_by_episode()
    for source_episode in selected:
        for frame in frames_by_episode[source_episode]:
            phase = ControlPhase(frame.phase)
            weight = correction_weight if phase == ControlPhase.CORRECTION else recovery_weight
            rows.append(
                {
                    "episode_index": output_episode,
                    "frame_index": frame.frame_index,
                    "weight": float(weight),
                    "source": "human_feedback",
                    "phase": phase.value,
                }
            )
        output_episode += 1
    sidecar_dir = out_dir / "training_sidecars"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("episode_index", pa.int64()),
            ("frame_index", pa.int64()),
            ("weight", pa.float64()),
            ("source", pa.string()),
            ("phase", pa.string()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), sidecar_dir / "training_weights.parquet")


def copy_language_annotations(
    language_root: Path,
    out_dir: Path,
    selected: list[int],
    output_offset: int,
    feedback_dataset: Path,
) -> dict[str, int]:
    manifest_path = language_root / "enrichment_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"language enrichment manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    source_dataset = manifest.get("dataset_root") or manifest.get("dataset")
    if source_dataset and Path(str(source_dataset)).resolve() != feedback_dataset.resolve():
        raise ValueError("language enrichment belongs to a different feedback dataset")
    annotated = {int(value) for value in manifest.get("annotated_episodes", [])}
    missing = sorted(set(selected).difference(annotated))
    if missing:
        raise ValueError(f"selected feedback episodes lack language annotations: {missing}")

    destination = out_dir / "training_sidecars" / "language_annotations.jsonl"
    mapping: dict[str, int] = {}
    with destination.open("w", encoding="utf-8") as handle:
        for local_index, source_episode in enumerate(selected):
            path = language_root / "annotations" / f"episode-{source_episode:06d}.json"
            value = read_json(path)
            output_episode = output_offset + local_index
            value["source_episode_index"] = source_episode
            value["episode_index"] = output_episode
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            mapping[str(source_episode)] = output_episode
    return mapping


def run_feedback_build(config: FeedbackBuildConfig) -> None:
    if not config.base_datasets:
        raise ValueError("at least one base demonstration dataset is required")
    for dataset in (*config.base_datasets, config.feedback_dataset):
        if not (dataset / "meta" / "info.json").exists():
            raise FileNotFoundError(f"dataset metadata not found: {dataset}")
    verify_quality_report(config.quality_report, config.feedback_dataset)
    quality_rows = load_quality_report(config.quality_report)
    selected, selection_mode = select_feedback_episodes(
        quality_rows, config.approved_episodes
    )
    sidecar = FeedbackSidecar.load(config.feedback_sidecar)
    verify_feedback_sidecar(
        sidecar,
        config.feedback_dataset,
        selected,
        explicit_human_approval=config.approved_episodes is not None,
    )

    all_feedback_episodes = set(range(int(read_json(config.feedback_dataset / "meta" / "info.json")["total_episodes"])))
    unknown = sorted(set(selected).difference(all_feedback_episodes))
    if unknown:
        raise ValueError(f"selected feedback episodes are outside the dataset: {unknown}")
    dropped = sorted(all_feedback_episodes.difference(selected))
    config.out_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{config.out_dir.name}-feedback-", dir=config.out_dir.parent
    ) as temporary:
        stage_root = Path(temporary)
        drop_list = stage_root / "drop_episodes.txt"
        drop_list.write_text("".join(f"{episode}\n" for episode in dropped), encoding="utf-8")
        accepted_feedback = stage_root / "accepted_feedback"
        build_clean_dataset(
            src_dataset=config.feedback_dataset,
            out_dir=accepted_feedback,
            drop_list=drop_list,
            overwrite=False,
            video_mode="copy-referenced",
            copy_mode=config.copy_mode,
            ffmpeg_path=None,
            video_crf=30,
            dry_run=False,
        )
        run_merge_paths(
            [*config.base_datasets, accepted_feedback],
            out_dir=config.out_dir,
            overwrite=config.overwrite,
            copy_mode=config.copy_mode,
        )

    write_training_weights(
        config.out_dir,
        config.base_datasets,
        selected,
        sidecar,
        config.recovery_weight,
        config.correction_weight,
    )
    base_episode_count = sum(
        int(read_json(dataset / "meta" / "info.json")["total_episodes"])
        for dataset in config.base_datasets
    )
    language_mapping = None
    if config.language_enrichment is not None:
        language_mapping = copy_language_annotations(
            config.language_enrichment,
            config.out_dir,
            selected,
            base_episode_count,
            config.feedback_dataset,
        )
    manifest = {
        "schema_version": 1,
        "base_datasets": [str(path.resolve()) for path in config.base_datasets],
        "feedback_dataset": str(config.feedback_dataset.resolve()),
        "feedback_sidecar": str(config.feedback_sidecar.resolve()),
        "quality_report": str(config.quality_report.resolve()),
        "selection_mode": selection_mode,
        "selected_feedback_episodes": selected,
        "rejected_feedback_episodes": dropped,
        "output_feedback_episode_mapping": {
            str(source): base_episode_count + index for index, source in enumerate(selected)
        },
        "recovery_weight": config.recovery_weight,
        "correction_weight": config.correction_weight,
        "language_enrichment": (
            str(config.language_enrichment.resolve())
            if config.language_enrichment is not None
            else None
        ),
        "language_episode_mapping": language_mapping,
        "autonomous_failure_actions_used_as_expert_targets": False,
        "source_datasets_modified": False,
    }
    (config.out_dir / "feedback_build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a train-ready dataset from approved human feedback episodes."
    )
    parser.add_argument("--base-dataset", type=Path, action="append", required=True)
    parser.add_argument("--feedback-dataset", type=Path, required=True)
    parser.add_argument("--feedback-sidecar", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--approved-episodes", type=Path, default=None)
    parser.add_argument("--language-enrichment", type=Path, default=None)
    parser.add_argument("--copy-mode", choices=["copy", "hardlink"], default="copy")
    parser.add_argument("--recovery-weight", type=float, default=1.0)
    parser.add_argument("--correction-weight", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_feedback_build(
        FeedbackBuildConfig(
            base_datasets=tuple(path.resolve() for path in args.base_dataset),
            feedback_dataset=args.feedback_dataset.resolve(),
            feedback_sidecar=args.feedback_sidecar.resolve(),
            quality_report=args.quality_report.resolve(),
            out_dir=args.out_dir.resolve(),
            approved_episodes=(
                args.approved_episodes.resolve() if args.approved_episodes is not None else None
            ),
            language_enrichment=(
                args.language_enrichment.resolve() if args.language_enrichment is not None else None
            ),
            copy_mode=args.copy_mode,
            overwrite=args.overwrite,
            recovery_weight=args.recovery_weight,
            correction_weight=args.correction_weight,
        )
    )


if __name__ == "__main__":
    main()
