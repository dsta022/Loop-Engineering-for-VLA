"""Pluggable training, evaluation, deployment, and promotion contracts."""

from __future__ import annotations

import importlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class TrainingRequest:
    iteration_id: str
    base_policy: str
    training_dataset: Path
    output_dir: Path
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingResult:
    checkpoint: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRequest:
    iteration_id: str
    checkpoint: str
    evaluation_dataset: Path | None = None
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    metrics: Mapping[str, float]
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeploymentRequest:
    iteration_id: str
    checkpoint: str
    target: str
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeploymentResult:
    deployment_id: str
    target: str
    details: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class PolicyTrainer(Protocol):
    name: str

    def train(self, request: TrainingRequest) -> TrainingResult | Mapping[str, Any]: ...


@runtime_checkable
class PolicyEvaluator(Protocol):
    name: str

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult | Mapping[str, Any]: ...


@runtime_checkable
class PolicyDeployer(Protocol):
    name: str

    def deploy(self, request: DeploymentRequest) -> DeploymentResult | Mapping[str, Any]: ...


def load_plugin(spec: str, method: str, config: Mapping[str, Any] | None = None) -> Any:
    if ":" not in spec:
        raise ValueError("policy iteration plugin must use module:attribute syntax")
    module_name, attribute_name = spec.split(":", 1)
    target = getattr(importlib.import_module(module_name), attribute_name)
    kwargs = dict(config or {})
    if isinstance(target, type):
        target = target(**kwargs)
    elif callable(target) and not hasattr(target, method):
        target = target(kwargs)
    if not hasattr(target, method) or not callable(getattr(target, method)):
        raise TypeError(f"plugin does not define {method}(): {spec}")
    return target


def normalize_training_result(value: TrainingResult | Mapping[str, Any]) -> TrainingResult:
    if isinstance(value, TrainingResult):
        result = value
    elif isinstance(value, Mapping):
        result = TrainingResult(
            checkpoint=str(value.get("checkpoint") or ""),
            metrics=_numeric_metrics(value.get("metrics") or {}),
            details=dict(value.get("details") or {}),
        )
    else:
        raise TypeError("trainer must return TrainingResult or a mapping")
    if not result.checkpoint:
        raise ValueError("trainer returned an empty checkpoint")
    _validate_json(result.details, "training details")
    return result


def normalize_evaluation_result(value: EvaluationResult | Mapping[str, Any]) -> EvaluationResult:
    if isinstance(value, EvaluationResult):
        result = value
    elif isinstance(value, Mapping):
        result = EvaluationResult(
            metrics=_numeric_metrics(value.get("metrics") or {}),
            details=dict(value.get("details") or {}),
        )
    else:
        raise TypeError("evaluator must return EvaluationResult or a mapping")
    if not result.metrics:
        raise ValueError("evaluator returned no metrics")
    _validate_json(result.details, "evaluation details")
    return result


def normalize_deployment_result(value: DeploymentResult | Mapping[str, Any]) -> DeploymentResult:
    if isinstance(value, DeploymentResult):
        result = value
    elif isinstance(value, Mapping):
        result = DeploymentResult(
            deployment_id=str(value.get("deployment_id") or ""),
            target=str(value.get("target") or ""),
            details=dict(value.get("details") or {}),
        )
    else:
        raise TypeError("deployer must return DeploymentResult or a mapping")
    if not result.deployment_id or not result.target:
        raise ValueError("deployer must return deployment_id and target")
    _validate_json(result.details, "deployment details")
    return result


def _numeric_metrics(value: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError("metrics must be a mapping")
    metrics: dict[str, float] = {}
    for key, raw in value.items():
        number = float(raw)
        if not math.isfinite(number):
            raise ValueError(f"metric {key!r} is not finite")
        metrics[str(key)] = number
    return metrics


def _validate_json(value: Any, label: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON serializable") from exc


@dataclass(frozen=True)
class PromotionPolicy:
    min_success_rate: float = 0.0
    max_intervention_rate: float = 1.0
    max_critical_failures: float = 0.0
    min_success_rate_delta: float | None = None
    baseline_success_rate: float | None = None
    require_human_approval: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_success_rate <= 1.0:
            raise ValueError("min_success_rate must be within [0, 1]")
        if not 0.0 <= self.max_intervention_rate <= 1.0:
            raise ValueError("max_intervention_rate must be within [0, 1]")
        if self.max_critical_failures < 0:
            raise ValueError("max_critical_failures must be non-negative")
        if (
            self.baseline_success_rate is not None
            and not 0.0 <= self.baseline_success_rate <= 1.0
        ):
            raise ValueError("baseline_success_rate must be within [0, 1]")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PromotionPolicy":
        return cls(
            min_success_rate=float(value.get("min_success_rate", 0.0)),
            max_intervention_rate=float(value.get("max_intervention_rate", 1.0)),
            max_critical_failures=float(value.get("max_critical_failures", 0.0)),
            min_success_rate_delta=(
                None
                if value.get("min_success_rate_delta") is None
                else float(value["min_success_rate_delta"])
            ),
            baseline_success_rate=(
                None
                if value.get("baseline_success_rate") is None
                else float(value["baseline_success_rate"])
            ),
            require_human_approval=bool(value.get("require_human_approval", True)),
        )

    def evaluate(
        self, metrics: Mapping[str, float], *, human_approved: bool
    ) -> "PromotionDecision":
        reasons: list[str] = []
        success_rate = metrics.get("success_rate")
        intervention_rate = metrics.get("intervention_rate")
        critical_failures = metrics.get("critical_failures")
        if success_rate is not None and not 0.0 <= success_rate <= 1.0:
            reasons.append("invalid_metric:success_rate")
        if intervention_rate is not None and not 0.0 <= intervention_rate <= 1.0:
            reasons.append("invalid_metric:intervention_rate")
        if critical_failures is not None and critical_failures < 0:
            reasons.append("invalid_metric:critical_failures")
        if success_rate is None:
            reasons.append("missing_metric:success_rate")
        elif success_rate < self.min_success_rate:
            reasons.append(
                f"success_rate_below_minimum:{success_rate}<{self.min_success_rate}"
            )
        if intervention_rate is None:
            reasons.append("missing_metric:intervention_rate")
        elif intervention_rate > self.max_intervention_rate:
            reasons.append(
                f"intervention_rate_above_maximum:{intervention_rate}>{self.max_intervention_rate}"
            )
        if critical_failures is None:
            reasons.append("missing_metric:critical_failures")
        elif critical_failures > self.max_critical_failures:
            reasons.append(
                f"critical_failures_above_maximum:{critical_failures}>{self.max_critical_failures}"
            )
        if self.min_success_rate_delta is not None:
            if self.baseline_success_rate is None or success_rate is None:
                reasons.append("success_rate_delta_cannot_be_evaluated")
            elif success_rate - self.baseline_success_rate < self.min_success_rate_delta:
                reasons.append("success_rate_delta_below_minimum")
        if self.require_human_approval and not human_approved:
            reasons.append("human_approval_required")
        return PromotionDecision(approved=not reasons, reasons=tuple(reasons))


@dataclass(frozen=True)
class PromotionDecision:
    approved: bool
    reasons: tuple[str, ...] = ()


def result_to_dict(value: Any) -> dict[str, Any]:
    payload = asdict(value)
    for key, item in list(payload.items()):
        if isinstance(item, Path):
            payload[key] = str(item)
    return payload
