#!/usr/bin/env python3
"""Run one auditable train/evaluate/promote/deploy policy iteration."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import sys

# Allow `python ./pipeline/policy_iteration.py` to run as a plain script: the repo-root
# packages must be importable without installing the project first.
if __package__ in (None, ""):
    _REPO_ROOT = str(Path(__file__).resolve().parents[1])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from policy_improvement.plugins import (
    DeploymentRequest,
    EvaluationRequest,
    PromotionPolicy,
    TrainingRequest,
    load_plugin,
    normalize_deployment_result,
    normalize_evaluation_result,
    normalize_training_result,
    result_to_dict,
)


@dataclass(frozen=True)
class IterationConfig:
    iteration_id: str
    base_policy: str
    training_dataset: Path
    output_dir: Path
    trainer_spec: str
    evaluator_spec: str
    deployer_spec: str
    deployment_target: str
    evaluation_dataset: Path | None
    approval_file: Path | None
    trainer_init: Mapping[str, Any]
    trainer_request: Mapping[str, Any]
    evaluator_init: Mapping[str, Any]
    evaluator_request: Mapping[str, Any]
    deployer_init: Mapping[str, Any]
    deployer_request: Mapping[str, Any]
    promotion: PromotionPolicy

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], config_dir: Path) -> "IterationConfig":
        plugins = value.get("plugins")
        if not isinstance(plugins, Mapping):
            raise ValueError("iteration config requires a plugins object")

        def plugin_config(name: str) -> Mapping[str, Any]:
            item = plugins.get(name)
            if not isinstance(item, Mapping) or not str(item.get("spec") or ""):
                raise ValueError(f"iteration config requires plugins.{name}.spec")
            return item

        trainer = plugin_config("trainer")
        evaluator = plugin_config("evaluator")
        deployer = plugin_config("deployer")

        def resolve_path(raw: Any, *, optional: bool = False) -> Path | None:
            if raw is None or raw == "":
                if optional:
                    return None
                raise ValueError("required path is missing from iteration config")
            path = Path(str(raw)).expanduser()
            return path if path.is_absolute() else (config_dir / path).resolve()

        iteration_id = str(value.get("iteration_id") or "").strip()
        base_policy = str(value.get("base_policy") or "").strip()
        target = str(value.get("deployment_target") or "").strip()
        if not iteration_id or not base_policy or not target:
            raise ValueError(
                "iteration_id, base_policy, and deployment_target must be non-empty"
            )
        promotion_value = value.get("promotion") or {}
        if not isinstance(promotion_value, Mapping):
            raise ValueError("promotion must be a JSON object")
        return cls(
            iteration_id=iteration_id,
            base_policy=base_policy,
            training_dataset=resolve_path(value.get("training_dataset")),  # type: ignore[arg-type]
            output_dir=resolve_path(value.get("output_dir")),  # type: ignore[arg-type]
            trainer_spec=str(trainer["spec"]),
            evaluator_spec=str(evaluator["spec"]),
            deployer_spec=str(deployer["spec"]),
            deployment_target=target,
            evaluation_dataset=resolve_path(value.get("evaluation_dataset"), optional=True),
            approval_file=resolve_path(value.get("approval_file"), optional=True),
            trainer_init=dict(trainer.get("init") or {}),
            trainer_request=dict(trainer.get("request") or {}),
            evaluator_init=dict(evaluator.get("init") or {}),
            evaluator_request=dict(evaluator.get("request") or {}),
            deployer_init=dict(deployer.get("init") or {}),
            deployer_request=dict(deployer.get("request") or {}),
            promotion=PromotionPolicy.from_dict(promotion_value),
        )


def read_iteration_config(path: Path) -> IterationConfig:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("iteration config must contain one JSON object")
    return IterationConfig.from_dict(value, path.resolve().parent)


def human_approval(path: Path | None, iteration_id: str) -> bool:
    if path is None or not path.exists():
        return False
    text = path.read_text(encoding="utf-8").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(value, Mapping):
        return False
    expected_iteration = str(value.get("iteration_id") or "")
    return bool(value.get("approved", False)) and expected_iteration == iteration_id


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def run_policy_iteration(config: IterationConfig, *, overwrite: bool = False) -> int:
    if not config.training_dataset.is_dir():
        raise FileNotFoundError(f"training dataset not found: {config.training_dataset}")
    if config.evaluation_dataset is not None and not config.evaluation_dataset.exists():
        raise FileNotFoundError(f"evaluation dataset not found: {config.evaluation_dataset}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / "iteration_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"iteration manifest already exists: {manifest_path}")

    started_at = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "iteration_id": config.iteration_id,
        "started_at": started_at,
        "base_policy": config.base_policy,
        "training_dataset": str(config.training_dataset.resolve()),
        "evaluation_dataset": (
            str(config.evaluation_dataset.resolve())
            if config.evaluation_dataset is not None
            else None
        ),
        "deployment_target": config.deployment_target,
        "plugins": {
            "trainer": config.trainer_spec,
            "evaluator": config.evaluator_spec,
            "deployer": config.deployer_spec,
        },
        "status": "running",
    }
    _write_manifest(manifest_path, manifest)

    try:
        trainer = load_plugin(config.trainer_spec, "train", config.trainer_init)
        training_output = config.output_dir.resolve() / "training"
        training_output.mkdir(parents=True, exist_ok=True)
        training = normalize_training_result(
            trainer.train(
                TrainingRequest(
                    iteration_id=config.iteration_id,
                    base_policy=config.base_policy,
                    training_dataset=config.training_dataset.resolve(),
                    output_dir=training_output,
                    config=config.trainer_request,
                )
            )
        )
        manifest["training"] = result_to_dict(training)
        _write_manifest(manifest_path, manifest)

        evaluator = load_plugin(config.evaluator_spec, "evaluate", config.evaluator_init)
        evaluation = normalize_evaluation_result(
            evaluator.evaluate(
                EvaluationRequest(
                    iteration_id=config.iteration_id,
                    checkpoint=training.checkpoint,
                    evaluation_dataset=(
                        config.evaluation_dataset.resolve()
                        if config.evaluation_dataset is not None
                        else None
                    ),
                    config=config.evaluator_request,
                )
            )
        )
        manifest["evaluation"] = result_to_dict(evaluation)
        approved_by_human = human_approval(config.approval_file, config.iteration_id)
        decision = config.promotion.evaluate(
            evaluation.metrics, human_approved=approved_by_human
        )
        manifest["promotion"] = {
            "approved": decision.approved,
            "reasons": list(decision.reasons),
            "human_approved": approved_by_human,
        }
        if not decision.approved:
            manifest["status"] = "not_promoted"
            manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
            _write_manifest(manifest_path, manifest)
            return 2

        deployer = load_plugin(config.deployer_spec, "deploy", config.deployer_init)
        deployment = normalize_deployment_result(
            deployer.deploy(
                DeploymentRequest(
                    iteration_id=config.iteration_id,
                    checkpoint=training.checkpoint,
                    target=config.deployment_target,
                    config=config.deployer_request,
                )
            )
        )
        if deployment.target != config.deployment_target:
            raise ValueError("deployer returned a different deployment target")
        manifest["deployment"] = result_to_dict(deployment)
        manifest["status"] = "deployed"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(manifest_path, manifest)
        return 0
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}:{str(exc)[:1000]}"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(manifest_path, manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one plugin-based train/evaluate/promote/deploy iteration."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = read_iteration_config(args.config)
    raise SystemExit(run_policy_iteration(config, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
