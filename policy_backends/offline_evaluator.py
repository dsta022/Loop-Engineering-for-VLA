"""Evaluator plugin that scores a checkpoint offline on a held-out dataset.

It reuses the audit's built-in VLA checkpoint evaluator: for every episode of
``request.evaluation_dataset`` the checkpoint predicts action chunks at sampled
anchors and is scored against the demonstrated actions (1.0 reproduces the
demonstration, 0.0 is no better than holding still).

Metrics returned:

- ``success_rate``: fraction of scored episodes at or above ``pass_threshold``.
  This is an **offline proxy** computed from demonstrations, not a real-robot
  success rate.
- ``mean_action_consistency``, ``min_action_consistency``,
  ``mean_counterfactual_margin`` (when the backend fills it),
  ``episodes_scored``, ``episodes_failed``.

The plugin deliberately does not invent ``intervention_rate`` or
``critical_failures``. Provide measured values through ``extra_metrics`` when
you have them; otherwise the promotion gate refuses deployment, which is the
safe outcome for an offline-only evaluation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.data_quality_audit import (
    AuditConfig,
    EpisodeRecord,
    episode_motion,
    load_metadata,
    read_episode_data,
    semantic_tasks_for_episode,
    semantic_videos_for_episode,
)
from policy_improvement.plugins import EvaluationRequest
from utils.quality_utils import EpisodeQuality
from utils.semantic_evaluator import SemanticEpisode, normalize_result

EvaluatorFactory = Callable[[Mapping[str, Any]], Any]


def default_evaluator_factory(config: Mapping[str, Any]) -> Any:
    from semantic_backends.vla_checkpoint_evaluator import create_evaluator

    return create_evaluator(config)


class OfflineConsistencyEvaluator:
    name = "offline_consistency_evaluator"

    def __init__(
        self,
        *,
        pass_threshold: float = 0.9,
        evaluator_config: Mapping[str, Any] | None = None,
        extra_metrics: Mapping[str, float] | None = None,
        max_episodes: int | None = None,
        evaluator_factory: EvaluatorFactory | None = None,
    ) -> None:
        if not 0.0 <= pass_threshold <= 1.0:
            raise ValueError("pass_threshold must be within [0, 1]")
        if max_episodes is not None and max_episodes < 1:
            raise ValueError("max_episodes must be at least 1")
        self.pass_threshold = float(pass_threshold)
        self.evaluator_config = dict(evaluator_config or {})
        self.extra_metrics = {
            str(key): float(value) for key, value in dict(extra_metrics or {}).items()
        }
        self.max_episodes = max_episodes
        self.evaluator_factory = evaluator_factory or default_evaluator_factory

    def evaluate(self, request: EvaluationRequest) -> dict[str, Any]:
        if request.evaluation_dataset is None:
            raise ValueError("offline evaluation requires evaluation_dataset in the iteration config")
        root = Path(request.evaluation_dataset).resolve()
        evaluator = self.evaluator_factory({**self.evaluator_config, "policy_path": request.checkpoint})
        config = AuditConfig(
            root=root,
            out_dir=root,
            depth_check="none",
            video_check="metadata",
            video_content_check="none",
            sample_frames=0,
            depth_sample_frames=0,
            duplicate_check="none",
            video_sample_stride=1,
            motion_check="none",
        )
        metadata = load_metadata(config)
        scores: list[float] = []
        margins: list[float] = []
        per_episode: list[dict[str, Any]] = []
        failed = 0
        row_count = metadata.episodes.num_rows
        if self.max_episodes is not None:
            row_count = min(row_count, self.max_episodes)
        for row in range(row_count):
            record = EpisodeRecord.from_columns(row, metadata.episode_cols)
            audit = EpisodeQuality(episode_index=record.episode_index, length=record.length)
            data = read_episode_data(root, record, metadata.depth_keys, audit)
            try:
                if data is None or audit.drop_reasons:
                    raise ValueError(f"episode data unreadable: {audit.drop_reasons}")
                tasks = semantic_tasks_for_episode(metadata, data)
                episode = SemanticEpisode(
                    dataset_root=root,
                    episode_index=record.episode_index,
                    length=record.length,
                    tasks=tasks,
                    videos=semantic_videos_for_episode(config, metadata, record),
                    task_profile=None,
                    motion=episode_motion(config, metadata, data),
                )
                result = normalize_result(evaluator.evaluate(episode))
            except Exception as exc:  # noqa: BLE001 - one bad episode must not abort the run
                failed += 1
                per_episode.append(
                    {
                        "episode_index": record.episode_index,
                        "error": f"{type(exc).__name__}:{str(exc)[:300]}",
                    }
                )
                continue
            scores.append(result.score)
            entry: dict[str, Any] = {"episode_index": record.episode_index, "score": result.score}
            if result.counterfactual_margin is not None:
                margins.append(result.counterfactual_margin)
                entry["counterfactual_margin"] = result.counterfactual_margin
            per_episode.append(entry)
        if not scores:
            raise RuntimeError("no episode could be scored; check the checkpoint and the evaluation dataset")
        array = np.asarray(scores, dtype=np.float64)
        metrics: dict[str, float] = {
            "success_rate": float(np.mean(array >= self.pass_threshold)),
            "mean_action_consistency": float(array.mean()),
            "min_action_consistency": float(array.min()),
            "episodes_scored": float(len(scores)),
            "episodes_failed": float(failed),
        }
        if margins:
            metrics["mean_counterfactual_margin"] = float(np.mean(margins))
        metrics.update(self.extra_metrics)
        return {
            "metrics": metrics,
            "details": {
                "backend": self.name,
                "offline_proxy": True,
                "pass_threshold": self.pass_threshold,
                "evaluation_dataset": str(root),
                "checkpoint": request.checkpoint,
                "per_episode": per_episode,
            },
        }


def create_evaluator(config: Mapping[str, Any] | None = None) -> OfflineConsistencyEvaluator:
    return OfflineConsistencyEvaluator(**dict(config or {}))
