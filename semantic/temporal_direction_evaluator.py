"""Temporal-direction evaluator: catch reversed / time-corrupted episodes.

Motion and per-frame alignment are both (nearly) order-blind: reversing an episode
preserves path length, smoothness, and the set of frames, so neither notices. A
reversed demonstration is physically valid but semantically backwards — the task is
undone rather than done.

Detecting that requires an *anchor for which way is progress*, because a trajectory
on its own is reversal-symmetric. This evaluator uses a text anchor: a **start**
description and a **goal** description define a direction in the vision-language
embedding space, and each frame is projected onto that axis. A forward episode's
projection trends up over time; a reversed one trends down. The score is the rank
correlation between frame order and goal-projection — high for forward, low for
reversed — which is order-sensitive by construction.

The start/goal text is taken from config, or derived from a Task Profile's first and
last stage ``observable_condition``. With no anchor available the evaluator cannot
run and raises, so an ensemble skips it gracefully.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from utils.media_utils import ffmpeg_available
from utils.semantic_evaluator import SemanticEpisode, SemanticResult

from semantic.backends import VisionLanguageBackend
from semantic.frame_sampler import FrameSamplingError, sample_segment_frames

# The dataset-level consensus check lives in utils.semantic_consistency (model-free,
# audit-facing) and is re-exported here so this module remains its documented home.
from utils.semantic_consistency import directional_consensus_outliers  # noqa: F401


DEFAULT_NUM_FRAMES = 16
DEFAULT_FRAME_SIZE = 224


def spearman_with_time(values: np.ndarray) -> float:
    """Rank correlation between position (0..N-1) and ``values``.

    +1 = values rise monotonically with time, -1 = fall monotonically, 0 = no
    monotone trend. Rank-based, so it is robust to the non-linear, noisy shape of
    a real goal-projection curve.
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    if n < 3 or float(values.std()) < 1e-12:  # too short or constant -> no direction
        return 0.0
    time_rank = np.arange(n, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    value_rank = np.empty(n, dtype=np.float64)
    value_rank[order] = np.arange(n, dtype=np.float64)
    tr = time_rank - time_rank.mean()
    vr = value_rank - value_rank.mean()
    denom = np.sqrt((tr * tr).sum() * (vr * vr).sum())
    return float((tr * vr).sum() / denom) if denom > 0 else 0.0


class TemporalDirectionEvaluator:
    """Score whether an episode progresses toward its goal over time."""

    def __init__(
        self,
        backend: VisionLanguageBackend,
        *,
        num_frames: int = DEFAULT_NUM_FRAMES,
        frame_size: int = DEFAULT_FRAME_SIZE,
        start_text: str | None = None,
        goal_text: str | None = None,
        video_key: str | None = None,
        ffmpeg_path: str | None = None,
    ) -> None:
        self.backend = backend
        self.num_frames = max(3, int(num_frames))
        self.frame_size = int(frame_size)
        self.start_text = start_text
        self.goal_text = goal_text
        self.video_key = video_key
        self._ffmpeg_path = ffmpeg_path
        self.name = f"temporal_direction:{getattr(backend, 'name', 'backend')}"

    def _resolve_anchor(
        self, task_profile: Mapping[str, Any] | None, tasks: Sequence[str]
    ) -> tuple[str, str]:
        if self.start_text and self.goal_text:
            return self.start_text, self.goal_text
        if isinstance(task_profile, Mapping):
            stages = task_profile.get("stages")
            if isinstance(stages, Sequence) and len(stages) >= 2:
                first, last = stages[0], stages[-1]
                if isinstance(first, Mapping) and isinstance(last, Mapping):
                    start = str(first.get("observable_condition", "")).strip()
                    goal = str(last.get("observable_condition", "")).strip()
                    if start and goal:
                        return start, goal
        raise RuntimeError(
            "temporal-direction evaluator needs start/goal text (config) or a "
            "task profile with >= 2 stages carrying observable_condition"
        )

    def _resolve_ffmpeg(self) -> str:
        path = self._ffmpeg_path or ffmpeg_available()
        if path is None:
            raise RuntimeError("ffmpeg is required for TemporalDirectionEvaluator")
        return path

    def _select_segment(self, episode: SemanticEpisode):
        if not episode.videos:
            raise ValueError("episode has no video segments to score")
        if self.video_key is not None:
            for segment in episode.videos:
                if segment.key == self.video_key:
                    return segment
            raise ValueError(f"configured video_key {self.video_key!r} not present")
        return episode.videos[0]

    def score_frames(
        self,
        frames: np.ndarray,
        tasks: Sequence[str],
        task_profile: Mapping[str, Any] | None = None,
        *,
        video_key: str = "",
    ) -> SemanticResult:
        """Score how strongly the episode trends toward its goal over time.

        Projects each frame embedding onto the start->goal text axis, then
        rank-correlates that goal-alignment with frame order. The correlation is
        reported in details["time_correlation"] (forward rises -> +, reversed
        falls -> -); the dataset pass (directional_consensus_outliers) decides which
        sign is "forward" and flags the minority. success_probability here assumes
        rising==forward, but the CONSENSUS check is what actually decides reversal,
        because a text anchor cannot know a task's polarity on its own.
        """
        if frames.shape[0] < 3:
            raise RuntimeError(f"need >= 3 frames, got {frames.shape[0]}")
        start_text, goal_text = self._resolve_anchor(task_profile, tasks)

        image_features = self.backend.embed_images(list(frames))
        text_features = self.backend.embed_texts([start_text, goal_text])
        axis = text_features[1] - text_features[0]
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            raise RuntimeError("start and goal text embed to the same point")
        axis = axis / norm

        projection = image_features @ axis  # (N,) goal-alignment per frame
        correlation = spearman_with_time(projection)
        success = float(np.clip((correlation + 1.0) / 2.0, 0.0, 1.0))

        low, high = float(projection.min()), float(projection.max())
        if high - low < 1e-9:
            curve = [0.0] * projection.shape[0]
        else:
            curve = [float((v - low) / (high - low)) for v in projection]

        details = {
            "method": "temporal_direction",
            "video_key": video_key,
            "frames_scored": int(frames.shape[0]),
            "time_correlation": correlation,
            "start_text": start_text,
            "goal_text": goal_text,
        }
        return SemanticResult(
            progress_curve=tuple(curve),
            success_probability=success,
            details=details,
        )

    def evaluate(self, episode: SemanticEpisode) -> SemanticResult:
        """Socket entry point: decode the video segment, then score its direction."""
        segment = self._select_segment(episode)
        ffmpeg_path = self._resolve_ffmpeg()
        try:
            frames = sample_segment_frames(
                Path(segment.path), ffmpeg_path,
                segment.from_timestamp, segment.to_timestamp,
                num_frames=self.num_frames, size=self.frame_size,
                segment_frames=episode.length or None,
            )
        except FrameSamplingError as exc:
            raise RuntimeError(f"frame sampling failed: {exc}") from exc
        return self.score_frames(
            frames, episode.tasks, episode.task_profile, video_key=segment.key
        )


def create_evaluator(config: Mapping[str, Any] | None = None) -> TemporalDirectionEvaluator:
    """Factory for the audit's ``module:attribute`` loader.

    Config keys: ``model_name``, ``pretrained``, ``device``, ``batch_size``
    (backend); ``num_frames``, ``frame_size``, ``start_text``, ``goal_text``,
    ``video_key`` (evaluator).
    """
    options = dict(config or {})
    from semantic.backends import OpenClipBackend

    backend = OpenClipBackend(
        model_name=str(options.get("model_name", "ViT-B-32")),
        pretrained=str(options.get("pretrained", "laion2b_s34b_b79k")),
        device=options.get("device"),
        batch_size=int(options.get("batch_size", 32)),
    )
    return TemporalDirectionEvaluator(
        backend,
        num_frames=int(options.get("num_frames", DEFAULT_NUM_FRAMES)),
        frame_size=int(options.get("frame_size", DEFAULT_FRAME_SIZE)),
        start_text=options.get("start_text"),
        goal_text=options.get("goal_text"),
        video_key=options.get("video_key"),
    )
