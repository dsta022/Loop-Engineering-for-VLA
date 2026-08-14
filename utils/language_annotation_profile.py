"""Validated contracts for trajectory-level language enrichment plugins."""

from __future__ import annotations

import importlib
import inspect
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from utils.semantic_evaluator import EpisodeMotion, VideoSegment


LANGUAGE_ANNOTATION_SCHEMA_VERSION = 1


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_confidence(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{field_name} must be finite and within [0, 1]")
    return confidence


def _frame_range(value: Mapping[str, Any], prefix: str = "") -> tuple[int, int]:
    start_value = value.get("start_frame", value.get("start"))
    end_value = value.get("end_frame", value.get("end"))
    if start_value is None or end_value is None:
        raise ValueError(f"{prefix}start_frame and end_frame are required")
    return int(start_value), int(end_value)


@dataclass(frozen=True)
class LanguageSubtask:
    start_frame: int
    end_frame: int
    text: str
    stage: str = ""
    confidence: float | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LanguageSubtask":
        start_frame, end_frame = _frame_range(value, "subtask.")
        return cls(
            start_frame=start_frame,
            end_frame=end_frame,
            text=_required_text(value.get("text", value.get("subtask")), "subtask.text"),
            stage=str(value.get("stage", value.get("stage_key", ""))).strip(),
            confidence=_optional_confidence(value.get("confidence"), "subtask.confidence"),
        )

    def validate(self, episode_length: int) -> None:
        if not 0 <= self.start_frame <= self.end_frame < episode_length:
            raise ValueError(
                "subtask frame range must satisfy "
                f"0 <= {self.start_frame} <= {self.end_frame} < {episode_length}"
            )
        _required_text(self.text, "subtask.text")
        _optional_confidence(self.confidence, "subtask.confidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "stage": self.stage,
            "text": self.text,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class LanguageEvent:
    frame_index: int
    event: str
    description: str
    end_frame: int | None = None
    confidence: float | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LanguageEvent":
        frame_value = value.get("frame_index", value.get("frame"))
        if frame_value is None:
            raise ValueError("event.frame_index is required")
        end_value = value.get("end_frame")
        return cls(
            frame_index=int(frame_value),
            end_frame=None if end_value is None else int(end_value),
            event=_required_text(value.get("event"), "event.event"),
            description=_required_text(
                value.get("description", value.get("text")), "event.description"
            ),
            confidence=_optional_confidence(value.get("confidence"), "event.confidence"),
        )

    def validate(self, episode_length: int) -> None:
        if not 0 <= self.frame_index < episode_length:
            raise ValueError(
                f"event frame_index must satisfy 0 <= {self.frame_index} < {episode_length}"
            )
        if self.end_frame is not None and not self.frame_index <= self.end_frame < episode_length:
            raise ValueError("event end_frame must be within the episode and not precede frame_index")
        _required_text(self.event, "event.event")
        _required_text(self.description, "event.description")
        _optional_confidence(self.confidence, "event.confidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "end_frame": self.end_frame,
            "event": self.event,
            "description": self.description,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class TimedLanguageDescription:
    start_frame: int
    end_frame: int
    text: str
    level: str = "state"
    confidence: float | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TimedLanguageDescription":
        start_frame, end_frame = _frame_range(value, "description.")
        return cls(
            start_frame=start_frame,
            end_frame=end_frame,
            text=_required_text(value.get("text", value.get("description")), "description.text"),
            level=str(value.get("level", "state")).strip() or "state",
            confidence=_optional_confidence(value.get("confidence"), "description.confidence"),
        )

    def validate(self, episode_length: int) -> None:
        if not 0 <= self.start_frame <= self.end_frame < episode_length:
            raise ValueError("description frame range must stay within the episode")
        _required_text(self.text, "description.text")
        _required_text(self.level, "description.level")
        _optional_confidence(self.confidence, "description.confidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "level": self.level,
            "text": self.text,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class TrajectoryLanguageAnnotation:
    episode_index: int
    length: int
    task: str
    source: str
    subtasks: tuple[LanguageSubtask, ...]
    events: tuple[LanguageEvent, ...]
    state_descriptions: tuple[TimedLanguageDescription, ...]
    summary: str
    failure_description: str = ""
    task_profile: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    quality_context: dict[str, Any] = field(default_factory=dict)
    schema_version: int = LANGUAGE_ANNOTATION_SCHEMA_VERSION

    @classmethod
    def from_result(
        cls,
        value: Mapping[str, Any],
        *,
        episode_index: int,
        length: int,
        task: str,
        source: str,
        task_profile: str = "",
        quality_context: Mapping[str, Any] | None = None,
    ) -> "TrajectoryLanguageAnnotation":
        if value.get("episode_index") is not None and int(value["episode_index"]) != episode_index:
            raise ValueError("annotator returned a different episode_index")
        raw_subtasks = value.get("subtasks", ())
        raw_events = value.get("events", ())
        raw_descriptions = value.get("state_descriptions", value.get("descriptions", ()))
        for field_name, items in (
            ("subtasks", raw_subtasks),
            ("events", raw_events),
            ("state_descriptions", raw_descriptions),
        ):
            if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
                raise TypeError(f"{field_name} must be a sequence")
        raw_details = value.get("details", {})
        if raw_details is None:
            raw_details = {}
        if not isinstance(raw_details, Mapping):
            raise TypeError("details must be a mapping")
        annotation = cls(
            episode_index=episode_index,
            length=length,
            task=task,
            source=source,
            subtasks=tuple(LanguageSubtask.from_dict(item) for item in raw_subtasks),
            events=tuple(LanguageEvent.from_dict(item) for item in raw_events),
            state_descriptions=tuple(
                TimedLanguageDescription.from_dict(item) for item in raw_descriptions
            ),
            summary=_required_text(
                value.get("summary", value.get("trajectory_summary")), "summary"
            ),
            failure_description=str(value.get("failure_description", "")).strip(),
            task_profile=task_profile,
            details=dict(raw_details),
            quality_context=dict(quality_context or {}),
            schema_version=int(
                value.get("schema_version", LANGUAGE_ANNOTATION_SCHEMA_VERSION)
            ),
        )
        annotation.validate()
        return annotation

    def validate(self) -> None:
        if self.schema_version != LANGUAGE_ANNOTATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version:{self.schema_version}")
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        if self.length <= 0:
            raise ValueError("length must be positive")
        _required_text(self.task, "task")
        _required_text(self.source, "source")
        _required_text(self.summary, "summary")
        if not self.subtasks and not self.events and not self.state_descriptions:
            raise ValueError("annotation must include at least one time-aligned label")

        previous_end = -1
        for subtask in self.subtasks:
            subtask.validate(self.length)
            if subtask.start_frame <= previous_end:
                raise ValueError("subtasks must be ordered and non-overlapping")
            previous_end = subtask.end_frame

        previous_event_frame = -1
        for event in self.events:
            event.validate(self.length)
            if event.frame_index < previous_event_frame:
                raise ValueError("events must be ordered by frame_index")
            previous_event_frame = event.frame_index

        previous_description_start = -1
        for description in self.state_descriptions:
            description.validate(self.length)
            if description.start_frame < previous_description_start:
                raise ValueError("state_descriptions must be ordered by start_frame")
            previous_description_start = description.start_frame

        try:
            json.dumps(self.details, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("details must be JSON serializable") from exc
        try:
            json.dumps(self.quality_context, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("quality_context must be JSON serializable") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_index": self.episode_index,
            "length": self.length,
            "task": self.task,
            "task_profile": self.task_profile,
            "source": self.source,
            "subtasks": [item.to_dict() for item in self.subtasks],
            "events": [item.to_dict() for item in self.events],
            "state_descriptions": [item.to_dict() for item in self.state_descriptions],
            "summary": self.summary,
            "failure_description": self.failure_description,
            "details": self.details,
            "quality_context": self.quality_context,
        }

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


@dataclass(frozen=True)
class LanguageAnnotationEpisode:
    dataset_root: Path
    episode_index: int
    length: int
    tasks: tuple[str, ...]
    videos: tuple[VideoSegment, ...]
    task_profile: Mapping[str, Any] | None = None
    semantic_context: Mapping[str, Any] = field(default_factory=dict)
    motion: EpisodeMotion | None = None


@runtime_checkable
class TrajectoryLanguageAnnotator(Protocol):
    name: str

    def annotate(
        self, episode: LanguageAnnotationEpisode
    ) -> TrajectoryLanguageAnnotation | Mapping[str, Any]:
        """Generate time-aligned language labels for one approved episode."""


def annotator_name(annotator: TrajectoryLanguageAnnotator) -> str:
    name = str(getattr(annotator, "name", "")).strip()
    return name or annotator.__class__.__name__


def load_annotator(
    spec: str, config: Mapping[str, Any] | None = None
) -> TrajectoryLanguageAnnotator:
    module_name, separator, attribute_name = spec.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("language annotator must use module:attribute syntax")
    module = importlib.import_module(module_name)
    target = getattr(module, attribute_name)
    options = dict(config or {})
    if inspect.isclass(target):
        candidate = target(**options)
    elif hasattr(target, "annotate"):
        candidate = target
    elif callable(target):
        candidate = target(options)
    else:
        raise TypeError(f"language annotator target is not usable: {spec}")
    if not callable(getattr(candidate, "annotate", None)):
        raise TypeError(f"language annotator does not define annotate(): {spec}")
    return candidate


def load_annotator_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("language annotator config must contain one JSON object")
    return payload


def normalize_annotation_result(
    value: TrajectoryLanguageAnnotation | Mapping[str, Any],
    episode: LanguageAnnotationEpisode,
    source: str,
    task_profile: str = "",
) -> TrajectoryLanguageAnnotation:
    payload = value.to_dict() if isinstance(value, TrajectoryLanguageAnnotation) else value
    if not isinstance(payload, Mapping):
        raise TypeError("language annotator must return a mapping or TrajectoryLanguageAnnotation")
    if len(episode.tasks) != 1:
        raise ValueError("language annotation requires exactly one task per episode")
    annotation = TrajectoryLanguageAnnotation.from_result(
        payload,
        episode_index=episode.episode_index,
        length=episode.length,
        task=episode.tasks[0],
        source=source,
        task_profile=task_profile,
        quality_context=episode.semantic_context,
    )
    if episode.task_profile is not None:
        raw_stages = episode.task_profile.get("stages", ())
        allowed_stages = {
            str(item.get("name", "")).strip()
            for item in raw_stages
            if isinstance(item, Mapping) and str(item.get("name", "")).strip()
        }
        unknown_stages = sorted(
            {item.stage for item in annotation.subtasks if item.stage}
            .difference(allowed_stages)
        )
        if unknown_stages:
            raise ValueError(f"subtask stages are absent from Task Profile: {unknown_stages}")
    return annotation
