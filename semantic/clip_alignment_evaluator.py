"""Alignment-based semantic evaluator producing a progress curve for the audit.

This is a concrete implementation of the audit's ``SemanticEvaluator`` protocol.
It samples frames from an episode's video segment, embeds them and a set of text
prompts with a vision-language backend, and turns per-frame image/text similarity
into a progress curve in ``[0, 1]``. That curve is exactly what the base audit
and the model-free consistency checks (progress-shape, stage-timeline) consume;
without an evaluator those checks never run.

Two curve methods:

  * **stage-anchored** (default when the episode carries a Task Profile with
    stages): each frame is scored against every stage's ``observable_condition``;
    progress is the soft, weight-cumulative position of the stage the frame most
    resembles. This is the signal the stage-timeline check was built for.
  * **instruction-only** (fallback): each frame is scored against the task text
    and min-max normalized across the episode. Coarser, but needs no profile and
    works on any task string, including ones no verb lexicon matches.

The scoring is deterministic given fixed frames and a fixed backend: no sampling,
eval-mode inference, no randomness. The heavy backend is constructed only by
``create_evaluator``; the class accepts an injected backend so tests run without
a model runtime.
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
from semantic.smoothing import smooth


DEFAULT_NUM_FRAMES = 16
DEFAULT_FRAME_SIZE = 224
# CLIP cosine similarities sit in ~0.15-0.35; a temperature of 100 drives the
# per-frame stage softmax nearly one-hot, producing brittle argmax-noise curves
# that over-fire the regression check (32% false positives on a clean 25-episode
# run). 30 keeps graded attention and eliminated those false positives.
DEFAULT_STAGE_TEMPERATURE = 30.0
DEFAULT_SMOOTHING = "median"
DEFAULT_SMOOTHING_WINDOW = 3
DEFAULT_SMOOTHING_ALPHA = 0.5
DEFAULT_PROMPT_TEMPLATE = "a robot arm {condition}"


class ClipAlignmentEvaluator:
    """Score task progress from image/text alignment over sampled frames."""

    def __init__(
        self,
        backend: VisionLanguageBackend,
        *,
        num_frames: int = DEFAULT_NUM_FRAMES,
        frame_size: int = DEFAULT_FRAME_SIZE,
        stage_temperature: float = DEFAULT_STAGE_TEMPERATURE,
        video_key: str | None = None,
        ffmpeg_path: str | None = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        smoothing: str = DEFAULT_SMOOTHING,
        smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
        smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
    ) -> None:
        self.backend = backend
        self.num_frames = max(2, int(num_frames))
        self.frame_size = int(frame_size)
        self.stage_temperature = float(stage_temperature)
        self.video_key = video_key
        self._ffmpeg_path = ffmpeg_path
        self.prompt_template = prompt_template
        self.smoothing = smoothing
        self.smoothing_window = int(smoothing_window)
        self.smoothing_alpha = float(smoothing_alpha)
        self.name = f"clip_alignment:{getattr(backend, 'name', 'backend')}"

    def _smooth(self, similarity: np.ndarray) -> np.ndarray:
        """Denoise the per-frame similarity signal along the frame axis."""
        return smooth(
            similarity,
            self.smoothing,
            window=self.smoothing_window,
            alpha=self.smoothing_alpha,
        )

    # -- video selection ---------------------------------------------------

    def _resolve_ffmpeg(self) -> str:
        path = self._ffmpeg_path or ffmpeg_available()
        if path is None:
            raise RuntimeError("ffmpeg is required for ClipAlignmentEvaluator")
        return path

    def _select_segment(self, episode: SemanticEpisode):
        """Pick which camera's video segment to score.

        Uses the configured ``video_key`` if set, else the first available
        segment. The audit may already have filtered ``episode.videos`` to a
        profile's preferred keys before we get here.
        """
        if not episode.videos:
            raise ValueError("episode has no video segments to score")
        if self.video_key is not None:
            for segment in episode.videos:
                if segment.key == self.video_key:
                    return segment
            raise ValueError(
                f"configured video_key {self.video_key!r} not among episode videos"
            )
        return episode.videos[0]

    # -- prompt construction ----------------------------------------------

    def _instruction_text(self, episode: SemanticEpisode) -> str:
        """Join the episode's task strings into one instruction to align against."""
        return " ; ".join(text for text in episode.tasks if text.strip())

    def _stage_prompts(
        self, profile: Mapping[str, Any]
    ) -> tuple[list[str], list[str], list[float]] | None:
        """Turn a Task Profile's stages into (names, text prompts, weights).

        Each stage's ``observable_condition`` becomes a text prompt (via
        ``prompt_template``) and its ``weight`` sets how much of the [0,1] progress
        range that stage spans. Returns ``None`` — falling back to instruction-only
        scoring — when there are no usable stages (missing condition, bad weight).
        """
        stages = profile.get("stages") if isinstance(profile, Mapping) else None
        if not isinstance(stages, Sequence) or len(stages) == 0:
            return None
        names: list[str] = []
        prompts: list[str] = []
        weights: list[float] = []
        for stage in stages:
            if not isinstance(stage, Mapping):
                return None
            condition = str(stage.get("observable_condition", "")).strip()
            name = str(stage.get("name", "")).strip()
            if not condition:
                return None
            weight = float(stage.get("weight", 1.0))
            if not weight > 0.0:
                return None
            names.append(name or condition)
            prompts.append(self.prompt_template.format(condition=condition))
            weights.append(weight)
        return names, prompts, weights

    # -- curve construction ------------------------------------------------

    def _stage_anchored_curve(
        self, image_features: np.ndarray, prompts: list[str], weights: list[float]
    ) -> tuple[list[float], np.ndarray]:
        """Map frames to progress via which stage each frame most resembles.

        For each frame: cosine-similarity to every stage prompt, a temperature-
        scaled softmax over stages (a soft "which stage is this frame in"), then a
        dot with each stage's *cumulative* weight fraction. So a frame that looks
        like the last stage scores near 1, one that looks like the first scores
        near that stage's small cumulative weight. Smooth, monotone-ish, in [0,1];
        it does NOT force saturation, unlike the instruction-only curve. Returns
        (curve, the smoothed per-frame/per-stage similarity matrix).
        """
        text_features = self.backend.embed_texts(prompts)
        raw_similarity = image_features @ text_features.T  # (N, S)
        similarity = self._smooth(raw_similarity)
        scaled = similarity * self.stage_temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled)
        attention = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)  # (N, S)
        weight_array = np.asarray(weights, dtype=np.float64)
        total = float(weight_array.sum())
        # Cumulative weight fraction at the *end* of each stage, in [0, 1].
        cumulative_end = np.cumsum(weight_array) / total
        curve = attention @ cumulative_end  # (N,)
        return [float(np.clip(value, 0.0, 1.0)) for value in curve], similarity

    def _instruction_curve(
        self, image_features: np.ndarray, instruction: str
    ) -> tuple[list[float], np.ndarray]:
        """Fallback curve when there is no profile: per-frame similarity to the
        instruction, min-max normalized across the episode.

        Coarser than the stage-anchored curve and has a built-in artifact — min-max
        forces some frame to exactly 1.0, which downstream reads as an early
        saturation. Prefer a Task Profile when accuracy matters. Returns
        (curve, the smoothed (N,1) similarity).
        """
        text_features = self.backend.embed_texts([instruction])
        raw_similarity = image_features @ text_features.T  # (N, 1)
        similarity = self._smooth(raw_similarity)
        series = similarity[:, 0]
        low = float(series.min())
        high = float(series.max())
        if high - low < 1e-9:
            curve = [0.0 for _ in series]
        else:
            curve = [float((value - low) / (high - low)) for value in series]
        return curve, similarity

    # -- scoring (frames already decoded) ---------------------------------

    def score_frames(
        self,
        frames: np.ndarray,
        tasks: Sequence[str],
        task_profile: Mapping[str, Any] | None = None,
        *,
        video_key: str = "",
    ) -> SemanticResult:
        """Score an already-decoded ``(N, H, W, 3)`` frame array.

        This is the scoring half of :meth:`evaluate`, exposed so callers can
        supply frames directly — for tests, or to probe how the curve responds
        to manipulated frame sequences (a discrimination study). It runs the
        same embedding, smoothing, and curve construction as the full path.
        """
        if frames.shape[0] < 2:
            raise RuntimeError(f"need >= 2 frames to score, got {frames.shape[0]}")
        image_features = self.backend.embed_images(list(frames))

        details: dict[str, Any] = {
            "backend": getattr(self.backend, "name", "backend"),
            "video_key": video_key,
            "frames_sampled": int(frames.shape[0]),
            "smoothing": self.smoothing,
            "smoothing_window": self.smoothing_window,
        }

        stage_data = (
            self._stage_prompts(task_profile)
            if isinstance(task_profile, Mapping)
            else None
        )
        if stage_data is not None:
            names, prompts, weights = stage_data
            curve, similarity = self._stage_anchored_curve(
                image_features, prompts, weights
            )
            details["method"] = "stage_anchored"
            details["stage_names"] = names
            details["final_stage_similarity"] = [
                float(value) for value in similarity[-1]
            ]
        else:
            instruction = " ; ".join(t for t in tasks if t.strip())
            if not instruction:
                raise RuntimeError("episode has no task text to align against")
            curve, similarity = self._instruction_curve(image_features, instruction)
            details["method"] = "instruction_only"
            details["raw_similarity"] = [float(value) for value in similarity[:, 0]]

        return SemanticResult(progress_curve=tuple(curve), details=details)

    # -- protocol entry point ---------------------------------------------

    def evaluate(self, episode: SemanticEpisode) -> SemanticResult:
        """The socket entry point: decode the episode's video segment, then score.

        Sampling is split from scoring (``score_frames``) so tests and experiments
        can feed frames directly. Raises on decode failure; the audit turns that
        into a review, never a drop.
        """
        segment = self._select_segment(episode)
        ffmpeg_path = self._resolve_ffmpeg()
        try:
            frames = sample_segment_frames(
                Path(segment.path),
                ffmpeg_path,
                segment.from_timestamp,
                segment.to_timestamp,
                num_frames=self.num_frames,
                size=self.frame_size,
                segment_frames=episode.length or None,
            )
        except FrameSamplingError as exc:
            raise RuntimeError(f"frame sampling failed: {exc}") from exc
        if frames.shape[0] < 2:
            raise RuntimeError(
                f"only {frames.shape[0]} frame(s) decoded from segment {segment.key}"
            )
        return self.score_frames(
            frames,
            episode.tasks,
            episode.task_profile,
            video_key=segment.key,
        )


def create_evaluator(config: Mapping[str, Any] | None = None) -> ClipAlignmentEvaluator:
    """Factory used by the audit's ``module:attribute`` evaluator loader.

    Recognized config keys: ``model_name``, ``pretrained``, ``device``,
    ``batch_size`` (passed to the backend); ``num_frames``, ``frame_size``,
    ``stage_temperature``, ``video_key``, ``prompt_template``, ``smoothing``
    (one of none/median/moving_average/ema), ``smoothing_window``,
    ``smoothing_alpha`` (evaluator).
    """
    options = dict(config or {})
    from semantic.backends import OpenClipBackend

    backend = OpenClipBackend(
        model_name=str(options.get("model_name", "ViT-B-32")),
        pretrained=str(options.get("pretrained", "laion2b_s34b_b79k")),
        device=options.get("device"),
        batch_size=int(options.get("batch_size", 32)),
    )
    return ClipAlignmentEvaluator(
        backend,
        num_frames=int(options.get("num_frames", DEFAULT_NUM_FRAMES)),
        frame_size=int(options.get("frame_size", DEFAULT_FRAME_SIZE)),
        stage_temperature=float(
            options.get("stage_temperature", DEFAULT_STAGE_TEMPERATURE)
        ),
        video_key=options.get("video_key"),
        prompt_template=str(options.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)),
        smoothing=str(options.get("smoothing", DEFAULT_SMOOTHING)),
        smoothing_window=int(options.get("smoothing_window", DEFAULT_SMOOTHING_WINDOW)),
        smoothing_alpha=float(options.get("smoothing_alpha", DEFAULT_SMOOTHING_ALPHA)),
    )
