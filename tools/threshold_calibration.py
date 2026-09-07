#!/usr/bin/env python3
"""Measure review-reason precision and recommend semantic thresholds from human labels.

The audit's semantic bands and review reasons are conservative defaults. This
tool compares a quality report against a human-labelled episode list and
reports, per review reason, how often a flagged episode was actually one a
reviewer rejected (precision) and how many rejected episodes the reason caught
(recall). It also sweeps the semantic score and recommends values for
``--semantic-failure-threshold`` and ``--semantic-pass-threshold``:

- failure threshold: the highest score below which flagged episodes are
  rejected with at least ``--precision-target`` precision;
- pass threshold: the lowest score at or above which no more than
  ``--pass-false-rate`` of the episodes were rejected.

Labels file: CSV with ``episode_index`` and a label column. Labels listed in
``--reject-labels`` mean "a human would not train on this episode"; every other
label means accepted.

Usage::

    python ./tools/threshold_calibration.py \\
      --quality-report ./quality_audit_reports/my_dataset/quality_report.csv \\
      --labels ./my_dataset_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow `python ./tools/threshold_calibration.py` to run as a plain script.
if __package__ in (None, ""):
    _REPO_ROOT = str(Path(__file__).resolve().parents[1])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from utils.quality_utils import DEFAULT_REPORT_DIR  # noqa: E402

DEFAULT_REJECT_LABELS = ("reject", "rejected", "fail", "failed", "bad", "drop", "0", "false", "no")


@dataclass(frozen=True)
class ReasonStats:
    reason: str
    kind: str
    flagged: int
    flagged_rejected: int
    total_rejected: int

    @property
    def precision(self) -> float | None:
        return self.flagged_rejected / self.flagged if self.flagged else None

    @property
    def recall(self) -> float | None:
        return self.flagged_rejected / self.total_rejected if self.total_rejected else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "kind": self.kind,
            "flagged": self.flagged,
            "flagged_rejected": self.flagged_rejected,
            "precision": self.precision,
            "recall": self.recall,
        }


def read_quality_report(path: Path) -> dict[int, dict[str, str]]:
    report = path / "quality_report.csv" if path.is_dir() else path
    if not report.is_file():
        raise FileNotFoundError(f"quality report not found: {report}")
    rows: dict[int, dict[str, str]] = {}
    with report.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "episode_index" not in reader.fieldnames:
            raise ValueError("quality report must contain an episode_index column")
        for row in reader:
            text = (row.get("episode_index") or "").strip()
            if text:
                rows[int(text)] = dict(row)
    if not rows:
        raise ValueError("quality report contains no episodes")
    return rows


def read_labels(path: Path, label_column: str, reject_labels: set[str]) -> dict[int, bool]:
    """Map episode_index -> True when a human rejected the episode."""
    if not path.is_file():
        raise FileNotFoundError(f"labels file not found: {path}")
    labels: dict[int, bool] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "episode_index" not in fields or label_column not in fields:
            raise ValueError(f"labels file must contain episode_index and {label_column!r} columns")
        for line_no, row in enumerate(reader, start=2):
            text = (row.get("episode_index") or "").strip()
            if not text:
                continue
            episode = int(text)
            if episode in labels:
                raise ValueError(f"duplicate episode_index {episode} in labels at line {line_no}")
            labels[episode] = (row.get(label_column) or "").strip().casefold() in reject_labels
    if not labels:
        raise ValueError("labels file contains no episodes")
    return labels


def base_reason(reason: str) -> str:
    return reason.split(":", 1)[0]


def reason_statistics(rows: dict[int, dict[str, str]], labels: dict[int, bool]) -> list[ReasonStats]:
    episodes = sorted(set(rows).intersection(labels))
    total_rejected = sum(labels[episode] for episode in episodes)
    flagged: dict[tuple[str, str], list[int]] = {}
    for episode in episodes:
        row = rows[episode]
        for kind, column in (("review", "review_reasons"), ("drop", "drop_reasons")):
            reasons = {base_reason(item) for item in (row.get(column) or "").split(";") if item}
            for reason in reasons:
                flagged.setdefault((kind, reason), []).append(episode)
    stats = [
        ReasonStats(
            reason=reason,
            kind=kind,
            flagged=len(hits),
            flagged_rejected=sum(labels[episode] for episode in hits),
            total_rejected=total_rejected,
        )
        for (kind, reason), hits in flagged.items()
    ]
    return sorted(stats, key=lambda item: (item.kind, -(item.precision or 0.0), item.reason))


def status_confusion(rows: dict[int, dict[str, str]], labels: dict[int, bool]) -> dict[str, dict[str, int]]:
    confusion: dict[str, dict[str, int]] = {}
    for episode in sorted(set(rows).intersection(labels)):
        status = (rows[episode].get("status") or "").strip() or "unknown"
        bucket = confusion.setdefault(status, {"accepted": 0, "rejected": 0})
        bucket["rejected" if labels[episode] else "accepted"] += 1
    return confusion


def _score(row: dict[str, str], column: str) -> float | None:
    text = (row.get(column) or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def score_sweep(
    rows: dict[int, dict[str, str]],
    labels: dict[int, bool],
    score_column: str,
) -> list[dict[str, Any]]:
    """Precision/recall of "score below t means rejected" for every candidate t."""
    pairs: list[tuple[float, bool]] = []
    for episode, row in rows.items():
        if episode not in labels:
            continue
        score = _score(row, score_column)
        if score is not None:
            pairs.append((score, labels[episode]))
    if not pairs:
        return []
    pairs.sort()
    total_rejected = sum(rejected for _, rejected in pairs)
    candidates = sorted({score for score, _ in pairs})
    sweep: list[dict[str, Any]] = []
    for threshold in candidates + [candidates[-1] + 1e-9]:
        below = [rejected for score, rejected in pairs if score < threshold]
        above = [rejected for score, rejected in pairs if score >= threshold]
        flagged = len(below)
        caught = sum(below)
        precision = caught / flagged if flagged else None
        recall = caught / total_rejected if total_rejected else None
        f1 = None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        sweep.append(
            {
                "threshold": threshold,
                "flagged_below": flagged,
                "precision_below": precision,
                "recall_below": recall,
                "f1_below": f1,
                "episodes_at_or_above": len(above),
                "rejected_rate_at_or_above": (sum(above) / len(above)) if above else None,
            }
        )
    return sweep


def recommend_thresholds(
    sweep: list[dict[str, Any]], precision_target: float, pass_false_rate: float
) -> dict[str, Any]:
    failure = None
    for point in sweep:
        precision = point["precision_below"]
        if point["flagged_below"] > 0 and precision is not None and precision >= precision_target:
            failure = point["threshold"]
    pass_threshold = None
    for point in sweep:
        rate = point["rejected_rate_at_or_above"]
        if point["episodes_at_or_above"] > 0 and rate is not None and rate <= pass_false_rate:
            pass_threshold = point["threshold"]
            break
    consistent = failure is not None and pass_threshold is not None and failure <= pass_threshold
    return {
        "semantic_failure_threshold": failure,
        "semantic_pass_threshold": pass_threshold,
        "consistent": consistent,
        "precision_target": precision_target,
        "pass_false_rate": pass_false_rate,
        "note": (
            "use these with --semantic-failure-threshold / --semantic-pass-threshold"
            if consistent
            else "no threshold pair satisfied both targets; keep the conservative defaults and label more episodes"
        ),
    }


def calibrate(
    rows: dict[int, dict[str, str]],
    labels: dict[int, bool],
    *,
    score_column: str = "semantic_score",
    precision_target: float = 0.9,
    pass_false_rate: float = 0.05,
) -> dict[str, Any]:
    matched = sorted(set(rows).intersection(labels))
    if not matched:
        raise ValueError("no episode appears in both the quality report and the labels file")
    sweep = score_sweep(rows, labels, score_column)
    return {
        "episodes_labelled": len(labels),
        "episodes_matched": len(matched),
        "episodes_rejected": sum(labels[episode] for episode in matched),
        "unlabelled_in_report": sorted(set(rows).difference(labels)),
        "status_confusion": status_confusion(rows, labels),
        "reasons": [item.to_dict() for item in reason_statistics(rows, labels)],
        "score_column": score_column,
        "score_sweep": sweep,
        "recommendation": recommend_thresholds(sweep, precision_target, pass_false_rate),
    }


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:5.2f}"


def print_report(report: dict[str, Any]) -> None:
    print(
        f"Episodes: labelled={report['episodes_labelled']} matched={report['episodes_matched']} "
        f"rejected={report['episodes_rejected']}"
    )
    print("Status vs labels:")
    for status, counts in sorted(report["status_confusion"].items()):
        print(f"  {status:8s} accepted={counts['accepted']:4d} rejected={counts['rejected']:4d}")
    print("Reason precision/recall (fraction of flagged episodes a human rejected):")
    for item in report["reasons"]:
        print(
            f"  {item['kind']:6s} {item['reason']:55s} flagged={item['flagged']:4d} "
            f"precision={_fmt(item['precision'])} recall={_fmt(item['recall'])}"
        )
    rec = report["recommendation"]
    print(
        f"Recommended semantic thresholds: failure={rec['semantic_failure_threshold']} "
        f"pass={rec['semantic_pass_threshold']} ({rec['note']})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate audit review reasons and semantic thresholds against human labels."
    )
    parser.add_argument("--quality-report", type=Path, required=True, help="quality_report.csv or its directory")
    parser.add_argument("--labels", type=Path, required=True, help="CSV with episode_index and a label column")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--reject-labels",
        default=",".join(DEFAULT_REJECT_LABELS),
        help="comma-separated label values meaning the episode was rejected by a human",
    )
    parser.add_argument("--score-column", default="semantic_score")
    parser.add_argument("--precision-target", type=float, default=0.9)
    parser.add_argument("--pass-false-rate", type=float, default=0.05)
    parser.add_argument("--out-dir", type=Path, default=None, help="defaults to the report directory")
    args = parser.parse_args()

    if not 0.0 <= args.precision_target <= 1.0 or not 0.0 <= args.pass_false_rate <= 1.0:
        parser.error("--precision-target and --pass-false-rate must be within [0, 1]")
    report_path = args.quality_report
    rows = read_quality_report(report_path)
    reject = {item.strip().casefold() for item in args.reject_labels.split(",") if item.strip()}
    labels = read_labels(args.labels, args.label_column, reject)
    report = calibrate(
        rows,
        labels,
        score_column=args.score_column,
        precision_target=args.precision_target,
        pass_false_rate=args.pass_false_rate,
    )
    out_dir = args.out_dir or (report_path if report_path.is_dir() else report_path.parent)
    if out_dir.name == DEFAULT_REPORT_DIR:
        out_dir = out_dir / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print_report(report)
    print(f"Report: {out_dir / 'calibration_report.json'}")


if __name__ == "__main__":
    main()
