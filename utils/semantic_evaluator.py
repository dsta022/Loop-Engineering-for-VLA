"""Model-agnostic contracts for optional semantic episode evaluation.

The base audit remains usable without any model dependencies. A semantic
backend is loaded only when explicitly requested and receives task text plus
episode-scoped video references.
"""

from __future__ import annotations

import importlib
import inspect
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class VideoSegment:
    """A video file and the time interval belonging to one episode."""

    key: str
    path: Path
    from_timestamp: float
    to_timestamp: float


@dataclass(frozen=True)
class EpisodeMotion:
    """Action and state streams already loaded by the audit.

    Both arrays are frame-major with shape (length, dim). They are exposed so
    evaluators and deterministic consistency checks can compare what the
    instruction says against what the arm actually did, without re-reading
    parquet. Either array may be None when the column is absent or unreadable.
    """

    action: np.ndarray | None = None
    state: np.ndarray | None = None
    fps: float | None = None
    gripper_index: int | None = None

    def gripper_series(self) -> np.ndarray | None:
        if self.action is None or self.gripper_index is None:
            return None
        if not 0 <= self.gripper_index < self.action.shape[1]:
            return None
        return self.action[:, self.gripper_index]


@dataclass(frozen=True)
class SemanticEpisode:
    """Inputs exposed to a semantic evaluator."""

    dataset_root: Path
    episode_index: int
    length: int
    tasks: tuple[str, ...]
    videos: tuple[VideoSegment, ...]
    task_profile: Mapping[str, Any] | None = None
    motion: EpisodeMotion | None = None


@dataclass(frozen=True)
class SemanticResult:
    """Normalized semantic assessment returned by an evaluator.

    Reliability contracts, all optional:

    - `score_samples` / `score_std`: repeated scorings of the same episode
      (multi-temperature or multi-frame sampling). Their dispersion becomes the
      score's uncertainty; a backend that samples but reports no point estimate
      gets `success_probability` derived as the sample mean.
    - `details["counterfactual_margin"]`: score(true instruction) minus the best
      score among deliberately corrupted instructions (swapped object, swapped
      target, reversed direction). A small or negative margin means the episode
      is mislabeled, ambiguous, or the evaluator cannot discriminate — the audit
      reviews it either way.
    """

    progress_curve: tuple[float, ...] = ()
    success_probability: float | None = None
    failure_stage: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    score_samples: tuple[float, ...] = ()
    score_std: float | None = None

    @property
    def final_progress(self) -> float | None:
        return self.progress_curve[-1] if self.progress_curve else None

    @property
    def score(self) -> float:
        """Prefer explicit success probability, otherwise final progress."""
        if self.success_probability is not None:
            return self.success_probability
        if self.final_progress is not None:
            return self.final_progress
        raise ValueError("semantic result has neither success probability nor progress")

    @property
    def uncertainty(self) -> float | None:
        """Score dispersion: explicit std wins, else the std of the samples."""
        if self.score_std is not None:
            return self.score_std
        if len(self.score_samples) >= 2:
            return float(np.std(np.asarray(self.score_samples, dtype=np.float64)))
        return None

    @property
    def counterfactual_margin(self) -> float | None:
        value = self.details.get("counterfactual_margin")
        return None if value is None else float(value)


@runtime_checkable
class SemanticEvaluator(Protocol):
    """Interface implemented by optional video-language scoring backends."""

    name: str

    def evaluate(self, episode: SemanticEpisode) -> SemanticResult | Mapping[str, Any]:
        """Return progress and/or success information for one episode."""


def evaluator_name(evaluator: SemanticEvaluator) -> str:
    name = str(getattr(evaluator, "name", "")).strip()
    return name or evaluator.__class__.__name__


def load_evaluator(spec: str, config: Mapping[str, Any] | None = None) -> SemanticEvaluator:
    """Load an evaluator from module:attribute.

    The attribute may be an evaluator instance, a class initialized with
    keyword arguments, or a factory accepting one configuration dictionary.
    """
    module_name, separator, attribute_name = spec.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("semantic evaluator must use module:attribute syntax")

    module = importlib.import_module(module_name)
    target = getattr(module, attribute_name)
    options = dict(config or {})
    if inspect.isclass(target):
        candidate = target(**options)
    elif hasattr(target, "evaluate"):
        candidate = target
    elif callable(target):
        candidate = target(options)
    else:
        raise TypeError(f"semantic evaluator target is not usable: {spec}")

    if not callable(getattr(candidate, "evaluate", None)):
        raise TypeError(f"semantic evaluator does not define evaluate(): {spec}")
    return candidate


def load_evaluator_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("semantic evaluator config must contain one JSON object")
    return payload


def _float_sequence(value: Any, field_name: str) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a numeric sequence")
    return tuple(float(item) for item in value)


def normalize_result(value: SemanticResult | Mapping[str, Any]) -> SemanticResult:
    """Validate backend output and return a stable result object."""
    if isinstance(value, SemanticResult):
        result = value
    elif isinstance(value, Mapping):
        raw_details = value.get("details", {})
        if raw_details is None:
            raw_details = {}
        if not isinstance(raw_details, Mapping):
            raise TypeError("semantic result details must be a mapping")
        result = SemanticResult(
            progress_curve=_float_sequence(value.get("progress_curve"), "progress_curve"),
            success_probability=(
                None if value.get("success_probability") is None else float(value["success_probability"])
            ),
            failure_stage=(
                None if value.get("failure_stage") in {None, ""} else str(value["failure_stage"])
            ),
            details=dict(raw_details),
            score_samples=_float_sequence(value.get("score_samples"), "score_samples"),
            score_std=(
                None if value.get("score_std") is None else float(value["score_std"])
            ),
        )
    else:
        raise TypeError("semantic evaluator must return SemanticResult or a mapping")

    if result.success_probability is None and result.score_samples:
        result = replace(
            result,
            success_probability=sum(result.score_samples) / len(result.score_samples),
        )

    for item in result.progress_curve:
        if not math.isfinite(item) or not 0.0 <= item <= 1.0:
            raise ValueError("progress_curve values must be finite and within [0, 1]")
    for item in result.score_samples:
        if not math.isfinite(item) or not 0.0 <= item <= 1.0:
            raise ValueError("score_samples values must be finite and within [0, 1]")
    if result.score_std is not None and (
        not math.isfinite(result.score_std) or result.score_std < 0.0
    ):
        raise ValueError("score_std must be finite and non-negative")
    if result.success_probability is not None:
        probability = result.success_probability
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("success_probability must be finite and within [0, 1]")
    if not result.progress_curve and result.success_probability is None:
        raise ValueError("semantic result must provide progress_curve or success_probability")

    margin_value = result.details.get("counterfactual_margin")
    if margin_value is not None:
        margin = float(margin_value)
        if not math.isfinite(margin) or not -1.0 <= margin <= 1.0:
            raise ValueError("counterfactual_margin must be finite and within [-1, 1]")

    json.dumps(result.details, ensure_ascii=False)
    return result


def infer_failure_stage(progress_curve: Sequence[float]) -> str:
    """Infer a coarse failure stage when a backend does not provide one."""
    if not progress_curve:
        return "final_outcome"
    peak = max(float(value) for value in progress_curve)
    final = float(progress_curve[-1])
    if peak - final >= 0.15:
        return "regressed_after_progress"
    if peak < 1.0 / 3.0:
        return "early_progress"
    if peak < 2.0 / 3.0:
        return "middle_progress"
    return "late_progress"
