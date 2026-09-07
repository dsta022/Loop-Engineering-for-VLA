"""Rule-based trajectory language annotator built from arm kinematics.

The annotator needs no model. It segments an approved episode at the same
kinematic change points the audit's boundary-alignment check trusts (gripper
level changes and near-stops of the arm), labels the segments with the ordered
stages of a Task Profile when one is configured, and records every gripper
transition as an event. The result is a valid, time-aligned annotation that
passes the enrichment validators and the boundary-alignment check by
construction. Its wording is templated, so treat it as a baseline or as the
skeleton that a stronger language-model annotator refines, not as ground truth.

Usage::

    python ./pipeline/trajectory_language_enrichment.py ./dataset \\
      --annotator language_backends.kinematic_annotator:create_annotator \\
      --task-profile-dir ./task_profiles
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from utils.language_annotation_profile import LanguageAnnotationEpisode
from utils.semantic_consistency import (
    GRIPPER_LEVEL_MARGIN,
    channel_traces,
    kinematic_change_points,
    select_gripper_index,
)
from utils.semantic_evaluator import EpisodeMotion

DEFAULT_MIN_SEGMENT_FRAMES = 5


def profile_stages(task_profile: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """(name, description) pairs of a Task Profile, in order; empty without one."""
    if not isinstance(task_profile, Mapping):
        return []
    raw = task_profile.get("stages")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    stages: list[tuple[str, str]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        description = str(item.get("description", "")).strip() or name.replace("_", " ")
        stages.append((name, description))
    return stages


def select_boundaries(change_points: Sequence[int], length: int, min_segment_frames: int) -> list[int]:
    """Keep change points that leave every segment at least ``min_segment_frames`` long."""
    kept: list[int] = []
    last = 0
    for point in sorted({int(item) for item in change_points}):
        if point <= 0 or point >= length:
            continue
        if point - last < min_segment_frames or length - point < min_segment_frames:
            continue
        kept.append(point)
        last = point
    return kept


def merge_segments(segments: list[tuple[int, int]], target: int) -> list[tuple[int, int]]:
    """Merge the shortest segment into its shorter neighbour until ``target`` remain."""
    result = list(segments)
    while len(result) > max(target, 1):
        lengths = [end - start + 1 for start, end in result]
        index = int(np.argmin(lengths))
        if index == 0:
            neighbour = 1
        elif index == len(result) - 1:
            neighbour = index - 1
        else:
            neighbour = index - 1 if lengths[index - 1] <= lengths[index + 1] else index + 1
        low, high = sorted((index, neighbour))
        result[low : high + 1] = [(result[low][0], result[high][1])]
    return result


def gripper_transitions(series: np.ndarray) -> list[tuple[int, str]]:
    """Frames where the gripper channel settles on a new level, with the level name."""
    values = np.asarray(series, dtype=np.float64).ravel()
    if values.size < 2 or not np.all(np.isfinite(values)):
        return []
    span = float(values.max() - values.min())
    if span <= 0.0:
        return []
    normalized = (values - float(values.min())) / span
    transitions: list[tuple[int, str]] = []
    state: str | None = None
    for frame, value in enumerate(normalized):
        if value <= 0.5 - GRIPPER_LEVEL_MARGIN:
            level = "low"
        elif value >= 0.5 + GRIPPER_LEVEL_MARGIN:
            level = "high"
        else:
            continue
        if state is not None and level != state:
            transitions.append((frame, level))
        state = level
    return transitions


class KinematicSegmentAnnotator:
    """Segment an episode by arm kinematics and label it with Task Profile stages."""

    name = "kinematic_segment_annotator"

    def __init__(
        self,
        *,
        min_segment_frames: int = DEFAULT_MIN_SEGMENT_FRAMES,
        gripper_index: int | None = None,
        max_segments: int | None = None,
    ) -> None:
        if min_segment_frames < 1:
            raise ValueError("min_segment_frames must be at least 1")
        if gripper_index is not None and gripper_index < 0:
            raise ValueError("gripper_index must be non-negative")
        if max_segments is not None and max_segments < 1:
            raise ValueError("max_segments must be at least 1")
        self.min_segment_frames = int(min_segment_frames)
        self.gripper_index = gripper_index
        self.max_segments = max_segments

    def _resolve_motion(self, motion: EpisodeMotion, episode_index: int) -> EpisodeMotion:
        if self.gripper_index is not None:
            return replace(motion, gripper_index=self.gripper_index)
        if motion.gripper_index is not None or motion.action is None:
            return motion
        detected = select_gripper_index({episode_index: channel_traces(motion.action)})
        return replace(motion, gripper_index=detected) if detected is not None else motion

    def annotate(self, episode: LanguageAnnotationEpisode) -> dict[str, Any]:
        if len(episode.tasks) != 1:
            raise ValueError("kinematic annotator requires exactly one task per episode")
        motion = episode.motion
        if motion is None or motion.action is None:
            raise ValueError("kinematic annotator needs the episode action stream (episode.motion)")
        length = int(episode.length)
        action = np.asarray(motion.action, dtype=np.float64)
        if action.ndim != 2 or action.shape[0] != length:
            raise ValueError(f"action stream shape {action.shape} does not match episode length {length}")
        motion = self._resolve_motion(motion, episode.episode_index)

        change_points = kinematic_change_points(motion)
        boundaries = select_boundaries(change_points, length, self.min_segment_frames)
        starts = [0, *boundaries]
        ends = [*[boundary - 1 for boundary in boundaries], length - 1]
        segments = list(zip(starts, ends))

        stages = profile_stages(episode.task_profile)
        target = len(stages) if stages else None
        if self.max_segments is not None:
            target = self.max_segments if target is None else min(target, self.max_segments)
        if target is not None and len(segments) > target:
            segments = merge_segments(segments, target)

        task = str(episode.tasks[0])
        subtasks: list[dict[str, Any]] = []
        for index, (start, end) in enumerate(segments):
            if stages and index < len(stages):
                stage_name, text = stages[index]
            else:
                stage_name = ""
                text = f"{task}: segment {index + 1} of {len(segments)}"
            subtasks.append(
                {"start_frame": int(start), "end_frame": int(end), "stage": stage_name, "text": text}
            )

        events: list[dict[str, Any]] = []
        gripper = motion.gripper_series()
        if gripper is not None:
            for frame, level in gripper_transitions(gripper):
                events.append(
                    {
                        "frame_index": int(frame),
                        "event": "gripper_level_change",
                        "description": f"The gripper moves to its {level} level.",
                    }
                )

        arm_columns = [
            column
            for column in range(action.shape[1])
            if motion.gripper_index is None or column != motion.gripper_index
        ]
        if arm_columns and length > 1:
            speeds = np.linalg.norm(np.diff(action[:, arm_columns], axis=0), axis=1)
        else:
            speeds = np.zeros(max(length - 1, 1))
        median_speed = float(np.median(speeds)) if speeds.size else 0.0
        descriptions: list[dict[str, Any]] = []
        for start, end in segments:
            window = speeds[start : max(end, start + 1)]
            mean_speed = float(np.mean(window)) if window.size else 0.0
            if median_speed <= 0.0 or mean_speed <= 0.25 * median_speed:
                pace = "The arm is nearly still"
            elif mean_speed >= 1.5 * median_speed:
                pace = "The arm moves quickly"
            else:
                pace = "The arm moves steadily"
            descriptions.append(
                {
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "level": "state",
                    "text": f"{pace} between frames {start} and {end}.",
                }
            )

        summary = (
            f"{task}. {len(segments)} kinematic segment(s) and {len(events)} gripper "
            f"transition(s) over {length} frames."
        )
        details = {
            "backend": self.name,
            "change_points": [int(point) for point in change_points],
            "boundaries": [int(point) for point in boundaries],
            "segments_after_merge": len(segments),
            "gripper_index": motion.gripper_index,
            "stage_source": "task_profile" if stages else "generic",
            "median_arm_speed": median_speed,
        }
        return {
            "subtasks": subtasks,
            "events": events,
            "state_descriptions": descriptions,
            "summary": summary,
            "details": details,
        }


def create_annotator(config: Mapping[str, Any] | None = None) -> KinematicSegmentAnnotator:
    """Factory for ``--annotator language_backends.kinematic_annotator:create_annotator``."""
    options = dict(config or {})
    unknown = set(options) - {"min_segment_frames", "gripper_index", "max_segments"}
    if unknown:
        raise ValueError(f"unknown kinematic annotator config keys: {sorted(unknown)}")
    return KinematicSegmentAnnotator(**options)
