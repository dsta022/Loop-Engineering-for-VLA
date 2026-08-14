"""Structured task semantics used by semantic quality evaluators."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROFILE_SCHEMA_VERSION = 1
TASK_KEY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def normalize_task_text(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings")
    return tuple(str(item).strip() for item in value if str(item).strip())


@dataclass(frozen=True)
class TaskObject:
    name: str
    attributes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskObject":
        return cls(
            name=str(value.get("name", "")).strip(),
            attributes=_string_tuple(value.get("attributes", ()), "object.attributes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "attributes": list(self.attributes)}


@dataclass(frozen=True)
class TaskStage:
    name: str
    description: str
    observable_condition: str
    weight: float = 1.0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskStage":
        return cls(
            name=str(value.get("name", "")).strip(),
            description=str(value.get("description", "")).strip(),
            observable_condition=str(value.get("observable_condition", "")).strip(),
            weight=float(value.get("weight", 1.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "observable_condition": self.observable_condition,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class TaskSemanticProfile:
    task_key: str
    task_name: str
    instruction: str
    stages: tuple[TaskStage, ...]
    success_criteria: tuple[str, ...]
    failure_criteria: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    objects: tuple[TaskObject, ...] = ()
    preferred_video_keys: tuple[str, ...] = ()
    review_threshold: float | None = None
    notes: str = ""
    schema_version: int = PROFILE_SCHEMA_VERSION
    pass_threshold: float | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskSemanticProfile":
        raw_stages = value.get("stages", ())
        raw_objects = value.get("objects", ())
        if isinstance(raw_stages, (str, bytes)) or not isinstance(raw_stages, Sequence):
            raise TypeError("stages must be a sequence")
        if isinstance(raw_objects, (str, bytes)) or not isinstance(raw_objects, Sequence):
            raise TypeError("objects must be a sequence")
        threshold_value = value.get("review_threshold")
        pass_threshold_value = value.get("pass_threshold")
        profile = cls(
            task_key=str(value.get("task_key", "")).strip(),
            task_name=str(value.get("task_name", "")).strip(),
            instruction=str(value.get("instruction", "")).strip(),
            stages=tuple(TaskStage.from_dict(item) for item in raw_stages),
            success_criteria=_string_tuple(value.get("success_criteria", ()), "success_criteria"),
            failure_criteria=_string_tuple(value.get("failure_criteria", ()), "failure_criteria"),
            aliases=_string_tuple(value.get("aliases", ()), "aliases"),
            objects=tuple(TaskObject.from_dict(item) for item in raw_objects),
            preferred_video_keys=_string_tuple(
                value.get("preferred_video_keys", ()), "preferred_video_keys"
            ),
            review_threshold=None if threshold_value is None else float(threshold_value),
            pass_threshold=(
                None if pass_threshold_value is None else float(pass_threshold_value)
            ),
            notes=str(value.get("notes", "")).strip(),
            schema_version=int(value.get("schema_version", PROFILE_SCHEMA_VERSION)),
        )
        profile.validate()
        return profile

    @classmethod
    def load(cls, path: Path) -> "TaskSemanticProfile":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError(f"task profile must contain one JSON object: {path}")
        return cls.from_dict(payload)

    def validate(self) -> None:
        errors: list[str] = []
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            errors.append(f"unsupported schema_version:{self.schema_version}")
        if not TASK_KEY_PATTERN.fullmatch(self.task_key):
            errors.append("task_key must match [a-z0-9][a-z0-9_-]*")
        if not self.task_name:
            errors.append("task_name is required")
        if not self.instruction:
            errors.append("instruction is required")
        if not self.stages:
            errors.append("at least one stage is required")
        if not self.success_criteria:
            errors.append("at least one success criterion is required")
        stage_names: set[str] = set()
        for index, stage in enumerate(self.stages):
            if not stage.name:
                errors.append(f"stage[{index}].name is required")
            elif stage.name in stage_names:
                errors.append(f"duplicate stage name:{stage.name}")
            stage_names.add(stage.name)
            if not stage.description:
                errors.append(f"stage[{index}].description is required")
            if not stage.observable_condition:
                errors.append(f"stage[{index}].observable_condition is required")
            if not math.isfinite(stage.weight) or stage.weight <= 0:
                errors.append(f"stage[{index}].weight must be positive")
        for index, obj in enumerate(self.objects):
            if not obj.name:
                errors.append(f"object[{index}].name is required")
        if self.review_threshold is not None and not 0.0 <= self.review_threshold <= 1.0:
            errors.append("review_threshold must be within [0, 1]")
        if self.pass_threshold is not None and not 0.0 <= self.pass_threshold <= 1.0:
            errors.append("pass_threshold must be within [0, 1]")
        if (
            self.review_threshold is not None
            and self.pass_threshold is not None
            and self.review_threshold > self.pass_threshold
        ):
            errors.append("review_threshold must not exceed pass_threshold")
        if errors:
            raise ValueError("invalid task semantic profile: " + "; ".join(errors))

    def match_terms(self) -> tuple[str, ...]:
        values = (self.task_key, self.task_name, self.instruction, *self.aliases)
        return tuple(dict.fromkeys(normalize_task_text(value) for value in values if value.strip()))

    def effective_review_threshold(self, default: float) -> float:
        return self.review_threshold if self.review_threshold is not None else default

    def effective_pass_threshold(self, default: float) -> float:
        return self.pass_threshold if self.pass_threshold is not None else default

    def stage_at_progress(self, value: float) -> str:
        """Map one progress value to the stage whose weight band contains it."""
        if not self.stages:
            return "final_outcome"
        total_weight = sum(stage.weight for stage in self.stages)
        completed_weight = float(value) * total_weight
        running = 0.0
        for stage in self.stages:
            running += stage.weight
            if completed_weight < running:
                return stage.name
        return self.stages[-1].name

    def stage_rank(self, name: str) -> int | None:
        for index, stage in enumerate(self.stages):
            if stage.name == name:
                return index
        return None

    def infer_failure_stage(self, progress_curve: Sequence[float]) -> str:
        if not self.stages:
            return "final_outcome"
        if not progress_curve:
            return self.stages[-1].name
        return self.stage_at_progress(max(float(value) for value in progress_curve))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_key": self.task_key,
            "task_name": self.task_name,
            "instruction": self.instruction,
            "aliases": list(self.aliases),
            "objects": [item.to_dict() for item in self.objects],
            "stages": [item.to_dict() for item in self.stages],
            "success_criteria": list(self.success_criteria),
            "failure_criteria": list(self.failure_criteria),
            "preferred_video_keys": list(self.preferred_video_keys),
            "review_threshold": self.review_threshold,
            "pass_threshold": self.pass_threshold,
            "notes": self.notes,
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
class TaskProfileRegistry:
    profiles: tuple[TaskSemanticProfile, ...]
    _by_term: dict[str, TaskSemanticProfile] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        by_term: dict[str, TaskSemanticProfile] = {}
        for profile in self.profiles:
            profile.validate()
            for term in profile.match_terms():
                previous = by_term.get(term)
                if previous is not None and previous.task_key != profile.task_key:
                    raise ValueError(
                        f"task profile match term {term!r} is shared by "
                        f"{previous.task_key!r} and {profile.task_key!r}"
                    )
                by_term[term] = profile
        object.__setattr__(self, "_by_term", by_term)

    @classmethod
    def from_directory(cls, directory: Path) -> "TaskProfileRegistry":
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"task profile directory not found: {directory}")
        paths = sorted(directory.glob("*.json"))
        if not paths:
            raise FileNotFoundError(f"no task profile JSON files found in: {directory}")
        return cls(tuple(TaskSemanticProfile.load(path) for path in paths))

    def match(self, task_text: str) -> TaskSemanticProfile | None:
        return self._by_term.get(normalize_task_text(task_text))

    def match_any(self, task_texts: Sequence[str]) -> TaskSemanticProfile | None:
        matches = {profile for text in task_texts if (profile := self.match(text)) is not None}
        if len(matches) > 1:
            keys = sorted(profile.task_key for profile in matches)
            raise ValueError(f"episode tasks match multiple semantic profiles: {keys}")
        return next(iter(matches), None)
