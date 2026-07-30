"""Proprioceptive semantic evaluator: task progress from the arm's own motion.

This is a second concrete ``SemanticEvaluator`` that needs no model and no video.
It reads the action/state streams the audit already loads (exposed on
``SemanticEpisode.motion`` as ``EpisodeMotion``) and turns them into a progress
curve plus physical-plausibility signals.

For a manipulation task the arm trajectory carries progress information a fixed
overhead camera does not: a stalled robot barely moves, a truncated demonstration
covers less of its trajectory, and a temporally corrupted sequence shows
non-physical jumps between adjacent frames. The signals here are deliberately
task-agnostic — they read the *shape* of the motion, not the task text — so they
complement, rather than duplicate, the alignment evaluator and the contributor's
verb/gripper consistency check.

Outputs:
  * ``progress_curve`` — normalized cumulative path length along the state
    trajectory (monotone 0->1 for a real episode); feeds the progress-shape and
    stage-timeline checks.
  * ``success_probability`` — a physical-plausibility score in [0, 1] combining
    "did it move at all" and "is the motion smooth"; low for frozen or
    temporally scrambled episodes.
  * ``details`` — total_path, motion_fraction, max_jerk and related metrics, so
    downstream calibration can pick its own discriminators.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from utils.semantic_evaluator import SemanticEpisode, SemanticResult


# A step below this fraction of the episode's median step counts as "not moving".
DEFAULT_STILL_FRACTION = 0.10
# Jerk (3rd difference) at or above this multiple of the median step magnitude is
# non-physical — the signature of shuffled or spliced frames.
DEFAULT_JERK_MULTIPLE = 8.0
DEFAULT_CURVE_POINTS = 32


def _prefer_stream(motion) -> np.ndarray | None:
    """Use state if present (measured pose), else action (commanded)."""
    if motion is None:
        return None
    for stream in (motion.state, motion.action):
        if stream is None:
            continue
        arr = np.asarray(stream, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 1:
            return arr
    return None


def step_magnitudes(trajectory: np.ndarray) -> np.ndarray:
    """Per-step Euclidean distance between consecutive frames, shape (T-1,)."""
    return np.linalg.norm(np.diff(trajectory, axis=0), axis=1)


def cumulative_progress_curve(trajectory: np.ndarray, points: int) -> list[float]:
    """Normalized cumulative path length, resampled to ``points`` values in [0,1]."""
    steps = step_magnitudes(trajectory)
    total = float(steps.sum())
    if total <= 0.0:
        return [0.0] * points
    cumulative = np.concatenate([[0.0], np.cumsum(steps)]) / total
    # resample onto an even index grid so the curve length is stable
    src = np.linspace(0.0, 1.0, cumulative.shape[0])
    dst = np.linspace(0.0, 1.0, points)
    return [float(np.clip(v, 0.0, 1.0)) for v in np.interp(dst, src, cumulative)]


def plausibility_metrics(
    trajectory: np.ndarray,
    still_fraction: float,
    jerk_multiple: float,
) -> dict[str, float]:
    """Task-agnostic descriptors of the motion's shape."""
    steps = step_magnitudes(trajectory)
    median_step = float(np.median(steps)) if steps.size else 0.0
    total_path = float(steps.sum())
    span = float(np.linalg.norm(trajectory.max(axis=0) - trajectory.min(axis=0)))
    if steps.size and median_step > 0.0:
        motion_fraction = float(np.mean(steps >= still_fraction * median_step))
    else:
        motion_fraction = 0.0

    # Direction consistency: mean cosine between consecutive step vectors. This is
    # scale-invariant, so (unlike jerk normalized by the trajectory's own median
    # step) it is not fooled by a uniform scramble — shuffling randomizes step
    # directions and drives it toward 0, while smooth motion keeps it near 1.
    vectors = np.diff(trajectory, axis=0)
    norms = np.linalg.norm(vectors, axis=1)
    direction_consistency = 1.0
    if vectors.shape[0] >= 2:
        a, b = vectors[:-1], vectors[1:]
        denom = norms[:-1] * norms[1:]
        valid = denom > 1e-12
        if np.any(valid):
            cos = np.einsum("ij,ij->i", a, b)[valid] / denom[valid]
            direction_consistency = float(np.mean(np.clip(cos, -1.0, 1.0)))

    # jerk kept as an informational detail (normalized by median step)
    if trajectory.shape[0] >= 4 and median_step > 0.0:
        jerk = np.linalg.norm(np.diff(trajectory, n=3, axis=0), axis=1) / median_step
        max_jerk = float(jerk.max())
        jerk_violations = float(np.mean(jerk >= jerk_multiple))
    else:
        max_jerk = 0.0
        jerk_violations = 0.0
    return {
        "total_path": total_path,
        "span": span,
        "median_step": median_step,
        "motion_fraction": motion_fraction,
        "direction_consistency": direction_consistency,
        "max_jerk": max_jerk,
        "jerk_violation_ratio": jerk_violations,
    }


class MotionEvaluator:
    """Score task progress and plausibility from proprioception alone."""

    def __init__(
        self,
        *,
        still_fraction: float = DEFAULT_STILL_FRACTION,
        jerk_multiple: float = DEFAULT_JERK_MULTIPLE,
        curve_points: int = DEFAULT_CURVE_POINTS,
        min_total_path: float = 0.0,
    ) -> None:
        self.still_fraction = float(still_fraction)
        self.jerk_multiple = float(jerk_multiple)
        self.curve_points = int(curve_points)
        self.min_total_path = float(min_total_path)
        self.name = "motion_proprioceptive"

    def score_trajectory(self, trajectory: np.ndarray) -> SemanticResult:
        """Score an already-extracted ``(T, D)`` motion trajectory."""
        trajectory = np.asarray(trajectory, dtype=np.float64)
        if trajectory.ndim != 2 or trajectory.shape[0] < 2:
            raise RuntimeError("motion trajectory must be 2-D with >= 2 frames")
        if not np.all(np.isfinite(trajectory)):
            raise RuntimeError("motion trajectory contains non-finite values")

        curve = cumulative_progress_curve(trajectory, self.curve_points)
        m = plausibility_metrics(trajectory, self.still_fraction, self.jerk_multiple)

        # plausibility: penalize (a) a robot that did not move and (b) motion whose
        # step directions are inconsistent (temporally scrambled or spliced). "moved"
        # is deliberately "did it move at all", not "fraction of frames moving" —
        # real teleoperated demos pause and dwell often, so a moving-frame fraction
        # would punish normal episodes. Both terms are in [0, 1]; product is the score.
        moved = 1.0 if m["total_path"] > self.min_total_path else 0.0
        smooth = float(np.clip(m["direction_consistency"], 0.0, 1.0))
        success = float(np.clip(moved * smooth, 0.0, 1.0))

        details: dict[str, Any] = {"method": "motion_cumulative_path", **m,
                                   "moved_score": moved, "smooth_score": smooth}
        return SemanticResult(
            progress_curve=tuple(curve),
            success_probability=success,
            details=details,
        )

    def evaluate(self, episode: SemanticEpisode) -> SemanticResult:
        trajectory = _prefer_stream(episode.motion)
        if trajectory is None:
            raise RuntimeError(
                "MotionEvaluator requires episode.motion with an action or state stream"
            )
        return self.score_trajectory(trajectory)


def create_evaluator(config: Mapping[str, Any] | None = None) -> MotionEvaluator:
    """Factory for the audit's ``module:attribute`` evaluator loader.

    Config keys (all optional): ``still_fraction``, ``jerk_multiple``,
    ``curve_points``, ``min_total_path``.
    """
    options = dict(config or {})
    return MotionEvaluator(
        still_fraction=float(options.get("still_fraction", DEFAULT_STILL_FRACTION)),
        jerk_multiple=float(options.get("jerk_multiple", DEFAULT_JERK_MULTIPLE)),
        curve_points=int(options.get("curve_points", DEFAULT_CURVE_POINTS)),
        min_total_path=float(options.get("min_total_path", 0.0)),
    )
