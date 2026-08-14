"""Validated feedback metadata and external sidecar storage.

The feedback sidecar deliberately stays outside the training dataset schema.
Robot observations and the action actually sent to the robot remain in the
dataset, while control provenance, intervention ranges, and the policy's
proposed action are stored here.  This prevents autonomous failure actions
from silently becoming expert targets.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


SIDECAR_SCHEMA_VERSION = 1
FRAME_FILE = "frame_feedback.parquet"
EPISODE_FILE = "episode_feedback.jsonl"
INTERVENTION_FILE = "intervention_segments.jsonl"
MANIFEST_FILE = "session_manifest.json"


class CaptureMode(str, Enum):
    CORRECTIONS_ONLY = "corrections_only"
    EVENT_BUFFER = "event_buffer"
    CONTINUOUS = "continuous"


class ControlPhase(str, Enum):
    AUTONOMOUS = "autonomous"
    SYNCING = "syncing"
    RECOVERY = "recovery"
    CORRECTION = "correction"
    PAUSED = "paused"


class TriggerSource(str, Enum):
    NONE = "none"
    MANUAL = "manual"
    DETERMINISTIC = "deterministic"
    SEMANTIC = "semantic"
    SCHEDULED = "scheduled"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


def _plain_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json_value(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return _plain_json_value(value.tolist())
    if hasattr(value, "item"):
        return _plain_json_value(value.item())
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _validate_finite_numbers(value: Any, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _validate_finite_numbers(item, label)
        return
    if isinstance(value, Sequence):
        for item in value:
            _validate_finite_numbers(item, label)
        return
    raise TypeError(f"{label} contains unsupported value {type(value).__name__}")


def _enum_value(value: str | Enum, enum_type: type[Enum]) -> str:
    return enum_type(value).value


@dataclass(frozen=True)
class FeedbackFrame:
    episode_index: int
    frame_index: int
    timestamp: float
    phase: ControlPhase | str
    intervention_id: int = -1
    trigger_source: TriggerSource | str = TriggerSource.NONE
    policy_action: Any = None
    executed_action: Any = None
    training_eligible: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.episode_index < 0 or self.frame_index < 0:
            raise ValueError("feedback episode_index and frame_index must be non-negative")
        if not math.isfinite(float(self.timestamp)) or self.timestamp < 0:
            raise ValueError("feedback timestamp must be finite and non-negative")
        phase = ControlPhase(self.phase)
        TriggerSource(self.trigger_source)
        if phase in {ControlPhase.RECOVERY, ControlPhase.CORRECTION}:
            if self.intervention_id < 0:
                raise ValueError("human-controlled frames require a non-negative intervention_id")
            if self.executed_action is None:
                raise ValueError("human-controlled frames require an executed_action")
        elif self.intervention_id != -1:
            raise ValueError("non-intervention frames must use intervention_id=-1")
        if self.training_eligible and phase not in {ControlPhase.RECOVERY, ControlPhase.CORRECTION}:
            raise ValueError("only human recovery/correction frames are training-eligible by default")
        policy_action = _plain_json_value(self.policy_action)
        executed_action = _plain_json_value(self.executed_action)
        details = _plain_json_value(dict(self.details))
        _validate_finite_numbers(policy_action, "policy_action")
        _validate_finite_numbers(executed_action, "executed_action")
        json.dumps(details, ensure_ascii=False)

    def to_row(self) -> dict[str, Any]:
        self.validate()
        return {
            "episode_index": self.episode_index,
            "frame_index": self.frame_index,
            "timestamp": float(self.timestamp),
            "phase": ControlPhase(self.phase).value,
            "intervention_id": self.intervention_id,
            "trigger_source": TriggerSource(self.trigger_source).value,
            "policy_action_json": json.dumps(_plain_json_value(self.policy_action), ensure_ascii=False),
            "executed_action_json": json.dumps(
                _plain_json_value(self.executed_action), ensure_ascii=False
            ),
            "training_eligible": bool(self.training_eligible),
            "details_json": json.dumps(_plain_json_value(dict(self.details)), ensure_ascii=False),
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "FeedbackFrame":
        return cls(
            episode_index=int(row["episode_index"]),
            frame_index=int(row["frame_index"]),
            timestamp=float(row["timestamp"]),
            phase=str(row["phase"]),
            intervention_id=int(row["intervention_id"]),
            trigger_source=str(row["trigger_source"]),
            policy_action=json.loads(str(row.get("policy_action_json") or "null")),
            executed_action=json.loads(str(row.get("executed_action_json") or "null")),
            training_eligible=bool(row.get("training_eligible", False)),
            details=json.loads(str(row.get("details_json") or "{}")),
        )


@dataclass(frozen=True)
class InterventionSegment:
    episode_index: int
    intervention_id: int
    recovery_start_frame: int
    recovery_end_frame: int
    correction_start_frame: int
    correction_end_frame: int
    trigger_source: TriggerSource | str = TriggerSource.MANUAL
    outcome: str = "completed"
    details: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, episode_length: int | None = None) -> None:
        if self.episode_index < 0 or self.intervention_id < 0:
            raise ValueError("intervention identifiers must be non-negative")
        if not (
            0
            <= self.recovery_start_frame
            <= self.recovery_end_frame
            < self.correction_start_frame
            <= self.correction_end_frame
        ):
            raise ValueError("recovery and correction ranges must be ordered and non-overlapping")
        if episode_length is not None and self.correction_end_frame >= episode_length:
            raise ValueError("intervention range exceeds the episode length")
        TriggerSource(self.trigger_source)
        json.dumps(_plain_json_value(dict(self.details)), ensure_ascii=False)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["trigger_source"] = TriggerSource(self.trigger_source).value
        payload["details"] = _plain_json_value(dict(self.details))
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InterventionSegment":
        segment = cls(
            episode_index=int(value["episode_index"]),
            intervention_id=int(value["intervention_id"]),
            recovery_start_frame=int(value["recovery_start_frame"]),
            recovery_end_frame=int(value["recovery_end_frame"]),
            correction_start_frame=int(value["correction_start_frame"]),
            correction_end_frame=int(value["correction_end_frame"]),
            trigger_source=str(value.get("trigger_source") or TriggerSource.MANUAL.value),
            outcome=str(value.get("outcome") or "completed"),
            details=dict(value.get("details") or {}),
        )
        segment.validate()
        return segment


@dataclass(frozen=True)
class EpisodeFeedback:
    episode_index: int
    length: int
    source_policy: str
    capture_mode: CaptureMode | str
    task: str = ""
    outcome: str = "pending_review"
    failure_stage: str = ""
    intervention_count: int = 0
    review_status: ReviewStatus | str = ReviewStatus.PENDING
    notes: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.episode_index < 0 or self.length <= 0:
            raise ValueError("episode feedback requires a non-negative index and positive length")
        if self.intervention_count < 0:
            raise ValueError("intervention_count must be non-negative")
        CaptureMode(self.capture_mode)
        ReviewStatus(self.review_status)
        json.dumps(_plain_json_value(dict(self.details)), ensure_ascii=False)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["capture_mode"] = CaptureMode(self.capture_mode).value
        payload["review_status"] = ReviewStatus(self.review_status).value
        payload["details"] = _plain_json_value(dict(self.details))
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpisodeFeedback":
        episode = cls(
            episode_index=int(value["episode_index"]),
            length=int(value["length"]),
            source_policy=str(value.get("source_policy") or ""),
            capture_mode=str(value["capture_mode"]),
            task=str(value.get("task") or ""),
            outcome=str(value.get("outcome") or "pending_review"),
            failure_stage=str(value.get("failure_stage") or ""),
            intervention_count=int(value.get("intervention_count", 0)),
            review_status=str(value.get("review_status") or ReviewStatus.PENDING.value),
            notes=str(value.get("notes") or ""),
            details=dict(value.get("details") or {}),
        )
        episode.validate()
        return episode


FRAME_SCHEMA = pa.schema(
    [
        ("episode_index", pa.int64()),
        ("frame_index", pa.int64()),
        ("timestamp", pa.float64()),
        ("phase", pa.string()),
        ("intervention_id", pa.int64()),
        ("trigger_source", pa.string()),
        ("policy_action_json", pa.string()),
        ("executed_action_json", pa.string()),
        ("training_eligible", pa.bool_()),
        ("details_json", pa.string()),
    ]
)


def _write_json_line(handle: Any, payload: Mapping[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


class FeedbackSidecarWriter:
    """Append validated feedback episodes without changing the source dataset."""

    def __init__(
        self,
        root: Path,
        *,
        session_id: str,
        source_dataset: Path,
        source_policy: str,
        capture_mode: CaptureMode | str,
        fps: float,
        overwrite: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.capture_mode = CaptureMode(capture_mode)
        if fps <= 0 or not math.isfinite(float(fps)):
            raise ValueError("feedback fps must be finite and positive")
        if self.root.exists() and any(self.root.iterdir()) and not overwrite:
            raise FileExistsError(f"feedback sidecar already contains data: {self.root}")
        if self.root.exists() and overwrite:
            for required in (FRAME_FILE, EPISODE_FILE, INTERVENTION_FILE, MANIFEST_FILE):
                path = self.root / required
                if path.exists():
                    path.unlink()
        self.root.mkdir(parents=True, exist_ok=True)
        self._parquet_writer: pq.ParquetWriter | None = None
        self._episode_handle = (self.root / EPISODE_FILE).open("a", encoding="utf-8")
        self._intervention_handle = (self.root / INTERVENTION_FILE).open("a", encoding="utf-8")
        self._closed = False
        self._episode_indices: list[int] = []
        self._frame_count = 0
        self._manifest = {
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_dataset": str(source_dataset.resolve()),
            "source_policy": source_policy,
            "capture_mode": self.capture_mode.value,
            "fps": float(fps),
            "source_dataset_modified": False,
            "files": {
                "frames": FRAME_FILE,
                "episodes": EPISODE_FILE,
                "interventions": INTERVENTION_FILE,
            },
            "details": _plain_json_value(dict(details or {})),
        }

    def append_episode(
        self,
        frames: Iterable[FeedbackFrame],
        episode: EpisodeFeedback,
        interventions: Iterable[InterventionSegment] = (),
    ) -> None:
        if self._closed:
            raise RuntimeError("feedback sidecar writer is closed")
        episode.validate()
        frame_items = list(frames)
        segment_items = list(interventions)
        if episode.episode_index in self._episode_indices:
            raise ValueError(f"duplicate feedback episode: {episode.episode_index}")
        if len(frame_items) != episode.length:
            raise ValueError(
                f"feedback frame count mismatch for episode {episode.episode_index}: "
                f"{len(frame_items)}!={episode.length}"
            )
        expected = list(range(episode.length))
        actual = [frame.frame_index for frame in frame_items]
        if actual != expected:
            raise ValueError("feedback frame_index values must be contiguous and episode-local")
        for frame in frame_items:
            frame.validate()
            if frame.episode_index != episode.episode_index:
                raise ValueError("feedback frame belongs to a different episode")
        for segment in segment_items:
            segment.validate(episode.length)
            if segment.episode_index != episode.episode_index:
                raise ValueError("intervention segment belongs to a different episode")
        if episode.intervention_count != len(segment_items):
            raise ValueError("episode intervention_count does not match intervention segments")
        human_frame_ids = {
            frame.intervention_id
            for frame in frame_items
            if ControlPhase(frame.phase) in {ControlPhase.RECOVERY, ControlPhase.CORRECTION}
        }
        segment_ids = {segment.intervention_id for segment in segment_items}
        if human_frame_ids != segment_ids:
            raise ValueError("human-controlled frames must have matching intervention segments")
        frame_map = {frame.frame_index: frame for frame in frame_items}
        for segment in segment_items:
            recovery_phases = {
                ControlPhase(frame_map[index].phase)
                for index in range(segment.recovery_start_frame, segment.recovery_end_frame + 1)
            }
            correction_phases = {
                ControlPhase(frame_map[index].phase)
                for index in range(segment.correction_start_frame, segment.correction_end_frame + 1)
            }
            if recovery_phases != {ControlPhase.RECOVERY}:
                raise ValueError("intervention recovery range does not match feedback frame phases")
            if correction_phases != {ControlPhase.CORRECTION}:
                raise ValueError("intervention correction range does not match feedback frame phases")

        table = pa.Table.from_pylist([frame.to_row() for frame in frame_items], schema=FRAME_SCHEMA)
        if self._parquet_writer is None:
            self._parquet_writer = pq.ParquetWriter(self.root / FRAME_FILE, FRAME_SCHEMA)
        self._parquet_writer.write_table(table)
        _write_json_line(self._episode_handle, episode.to_dict())
        for segment in segment_items:
            _write_json_line(self._intervention_handle, segment.to_dict())
        self._episode_indices.append(episode.episode_index)
        self._frame_count += episode.length

    def close(self) -> None:
        if self._closed:
            return
        if self._parquet_writer is None:
            pq.write_table(pa.Table.from_pylist([], schema=FRAME_SCHEMA), self.root / FRAME_FILE)
        else:
            self._parquet_writer.close()
        self._episode_handle.close()
        self._intervention_handle.close()
        manifest = {
            **self._manifest,
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "episodes": self._episode_indices,
            "episode_count": len(self._episode_indices),
            "frame_count": self._frame_count,
        }
        temp_path = self.root / f"{MANIFEST_FILE}.tmp"
        temp_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(self.root / MANIFEST_FILE)
        self._closed = True

    def __enter__(self) -> "FeedbackSidecarWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"feedback sidecar file not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not text.strip():
            continue
        value = json.loads(text)
        if not isinstance(value, dict):
            raise TypeError(f"feedback JSONL row must be an object at {path}:{line_number}")
        rows.append(value)
    return rows


@dataclass(frozen=True)
class FeedbackSidecar:
    root: Path
    manifest: Mapping[str, Any]
    frames: tuple[FeedbackFrame, ...]
    episodes: tuple[EpisodeFeedback, ...]
    interventions: tuple[InterventionSegment, ...]

    @classmethod
    def load(cls, root: Path) -> "FeedbackSidecar":
        root = root.resolve()
        manifest_path = root / MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(f"feedback sidecar manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("schema_version", -1)) != SIDECAR_SCHEMA_VERSION:
            raise ValueError("unsupported feedback sidecar schema_version")
        frame_table = pq.read_table(root / FRAME_FILE)
        frame_rows = frame_table.to_pylist()
        frames = tuple(FeedbackFrame.from_row(row) for row in frame_rows)
        episodes = tuple(
            EpisodeFeedback.from_dict(row) for row in _read_json_lines(root / EPISODE_FILE)
        )
        interventions = tuple(
            InterventionSegment.from_dict(row)
            for row in _read_json_lines(root / INTERVENTION_FILE)
        )
        sidecar = cls(root, manifest, frames, episodes, interventions)
        sidecar.validate()
        return sidecar

    def validate(self) -> None:
        episode_map = {episode.episode_index: episode for episode in self.episodes}
        if len(episode_map) != len(self.episodes):
            raise ValueError("feedback sidecar contains duplicate episode metadata")
        frames_by_episode = self.frames_by_episode()
        segments_by_episode = self.interventions_by_episode()
        for episode_index, episode in episode_map.items():
            errors = self.validate_episode_alignment(episode_index, episode.length)
            if errors:
                raise ValueError("; ".join(errors))
            episode_segments = segments_by_episode.get(episode_index, ())
            if len(episode_segments) != episode.intervention_count:
                raise ValueError(
                    f"feedback intervention count mismatch for episode {episode_index}"
                )
            episode_frames = frames_by_episode.get(episode_index, ())
            frame_ids = {
                frame.intervention_id
                for frame in episode_frames
                if ControlPhase(frame.phase) in {ControlPhase.RECOVERY, ControlPhase.CORRECTION}
            }
            segment_ids = {segment.intervention_id for segment in episode_segments}
            if frame_ids != segment_ids:
                raise ValueError(
                    f"feedback intervention ids do not match segments for episode {episode_index}"
                )
            frame_map = {frame.frame_index: frame for frame in episode_frames}
            for segment in episode_segments:
                for frame_index in range(
                    segment.recovery_start_frame, segment.recovery_end_frame + 1
                ):
                    frame = frame_map.get(frame_index)
                    if frame is None or ControlPhase(frame.phase) != ControlPhase.RECOVERY:
                        raise ValueError(
                            f"feedback recovery range does not match frame phases for episode {episode_index}"
                        )
                for frame_index in range(
                    segment.correction_start_frame, segment.correction_end_frame + 1
                ):
                    frame = frame_map.get(frame_index)
                    if frame is None or ControlPhase(frame.phase) != ControlPhase.CORRECTION:
                        raise ValueError(
                            f"feedback correction range does not match frame phases for episode {episode_index}"
                        )
        unknown_frames = sorted(set(frames_by_episode).difference(episode_map))
        unknown_segments = sorted(set(segments_by_episode).difference(episode_map))
        if unknown_frames or unknown_segments:
            raise ValueError(
                f"feedback rows reference unknown episodes: frames={unknown_frames} "
                f"interventions={unknown_segments}"
            )

    def frames_by_episode(self) -> dict[int, tuple[FeedbackFrame, ...]]:
        grouped: dict[int, list[FeedbackFrame]] = {}
        for frame in self.frames:
            grouped.setdefault(frame.episode_index, []).append(frame)
        return {
            episode: tuple(sorted(rows, key=lambda item: item.frame_index))
            for episode, rows in grouped.items()
        }

    def interventions_by_episode(self) -> dict[int, tuple[InterventionSegment, ...]]:
        grouped: dict[int, list[InterventionSegment]] = {}
        for segment in self.interventions:
            grouped.setdefault(segment.episode_index, []).append(segment)
        return {
            episode: tuple(sorted(rows, key=lambda item: item.intervention_id))
            for episode, rows in grouped.items()
        }

    def validate_episode_alignment(self, episode_index: int, expected_length: int) -> list[str]:
        errors: list[str] = []
        rows = self.frames_by_episode().get(episode_index, ())
        if len(rows) != expected_length:
            errors.append(
                f"feedback_frame_count_mismatch:{episode_index}:{len(rows)}!={expected_length}"
            )
        actual = [frame.frame_index for frame in rows]
        if actual != list(range(expected_length)):
            errors.append(f"feedback_frame_index_not_contiguous:{episode_index}")
        for frame in rows:
            try:
                frame.validate()
            except Exception as exc:
                errors.append(
                    f"feedback_frame_invalid:{episode_index}:{frame.frame_index}:"
                    f"{type(exc).__name__}:{exc}"
                )
        return errors

    def training_eligible_episodes(self) -> list[int]:
        grouped = self.frames_by_episode()
        return sorted(
            episode
            for episode, frames in grouped.items()
            if frames and any(frame.training_eligible for frame in frames)
        )
