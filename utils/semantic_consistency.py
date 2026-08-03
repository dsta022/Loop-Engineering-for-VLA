"""Deterministic semantic consistency checks built from signals the audit already has.

None of these checks load a model. They read three things that are already
available during a quality audit:

  - the progress curve returned by a semantic evaluator,
  - the ordered stages of a Task Profile,
  - the action/state streams read from the episode parquet.

Every finding is review-only, and every check skips itself explicitly when its
preconditions are absent, so a dataset that does not fit a check is reported as
not-checked rather than silently passed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.semantic_evaluator import EpisodeMotion
from utils.task_semantic_profile import TaskSemanticProfile


PROGRESS_ONSET_LEVEL = 0.05
PROGRESS_SATURATION_LEVEL = 0.95
PROGRESS_REGRESSION_DROP = 0.15
PROGRESS_IDLE_TAIL_RATIO = 0.30
PROGRESS_IDLE_HEAD_RATIO = 0.50
PROGRESS_MIN_POINTS_FOR_IDLE = 4

STAGE_DWELL_MIN_SAMPLES = 5
STAGE_DWELL_MAD_SCALE = 5.0
STAGE_DWELL_MIN_DEVIATION = 0.10

GRIPPER_BIMODAL_RATIO = 0.80
GRIPPER_LEVEL_MARGIN = 0.15
GRIPPER_MIN_SPAN_FRACTION = 0.20
MOTION_VERTICAL_DOMINANCE = 0.5

BOUNDARY_ALIGNMENT_MIN_RATIO = 0.5
BOUNDARY_TOLERANCE_FRAMES = 5
BOUNDARY_SPEED_QUANTILE = 0.25
BOUNDARY_MIN_DIP_RATIO = 0.05


# ---------------------------------------------------------------- progress curve


@dataclass(frozen=True)
class ProgressCurveMetrics:
    """Temporal shape of one progress curve, beyond its peak and final value."""

    points: int
    peak: float
    final: float
    mean: float
    onset_ratio: float | None
    saturation_ratio: float | None
    idle_head_ratio: float
    idle_tail_ratio: float
    regression_count: int
    max_drop: float
    total_drop: float

    @property
    def monotonic(self) -> bool:
        return self.regression_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": self.points,
            "peak": self.peak,
            "final": self.final,
            "mean": self.mean,
            "onset_ratio": self.onset_ratio,
            "saturation_ratio": self.saturation_ratio,
            "idle_head_ratio": self.idle_head_ratio,
            "idle_tail_ratio": self.idle_tail_ratio,
            "regression_count": self.regression_count,
            "max_drop": self.max_drop,
            "total_drop": self.total_drop,
            "monotonic": self.monotonic,
        }


def analyze_progress_curve(curve: Sequence[float]) -> ProgressCurveMetrics | None:
    """Summarize a progress curve; None when there is nothing to summarize."""
    values = [float(value) for value in curve]
    if not values:
        return None
    points = len(values)
    last_index = max(points - 1, 1)

    onset_index = next(
        (index for index, value in enumerate(values) if value > PROGRESS_ONSET_LEVEL), None
    )
    saturation_index = next(
        (index for index, value in enumerate(values) if value >= PROGRESS_SATURATION_LEVEL),
        None,
    )
    onset_ratio = None if onset_index is None else onset_index / last_index
    saturation_ratio = None if saturation_index is None else saturation_index / last_index

    drops = [
        previous - current
        for previous, current in zip(values, values[1:])
        if current < previous
    ]
    return ProgressCurveMetrics(
        points=points,
        peak=max(values),
        final=values[-1],
        mean=sum(values) / points,
        onset_ratio=onset_ratio,
        saturation_ratio=saturation_ratio,
        idle_head_ratio=1.0 if onset_ratio is None else onset_ratio,
        idle_tail_ratio=0.0 if saturation_ratio is None else 1.0 - saturation_ratio,
        regression_count=len(drops),
        max_drop=max(drops) if drops else 0.0,
        total_drop=sum(drops) if drops else 0.0,
    )


def progress_curve_findings(
    metrics: ProgressCurveMetrics, *, include_regression: bool = True
) -> tuple[str, ...]:
    """Review reasons implied by the shape of a progress curve.

    ``include_regression`` fires the regression finding against the fixed
    ``PROGRESS_REGRESSION_DROP`` floor. The audit passes ``False`` so the
    regression decision is deferred to a dataset-relative threshold computed over
    the whole population (see ``apply_progress_regression_reviews``), the same way
    action- and state-jump outliers are decided; other callers keep the standalone
    fixed-threshold behavior.
    """
    findings: list[str] = []
    if include_regression and metrics.points >= 2 and metrics.max_drop >= PROGRESS_REGRESSION_DROP:
        findings.append("semantic_progress_regression_review")
    if metrics.points >= PROGRESS_MIN_POINTS_FOR_IDLE:
        if (
            metrics.saturation_ratio is not None
            and metrics.idle_tail_ratio >= PROGRESS_IDLE_TAIL_RATIO
        ):
            findings.append("semantic_progress_idle_tail_review")
        if (
            metrics.onset_ratio is not None
            and metrics.onset_ratio >= PROGRESS_IDLE_HEAD_RATIO
        ):
            findings.append("semantic_progress_idle_head_review")
    return tuple(findings)


# ---------------------------------------------------------------- stage timeline


@dataclass(frozen=True)
class StageSpan:
    stage: str
    start_index: int
    end_index: int

    @property
    def points(self) -> int:
        return self.end_index - self.start_index + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "start_index": self.start_index,
            "end_index": self.end_index,
        }


@dataclass(frozen=True)
class StageTimeline:
    """Which stage each progress point belongs to, plus its ordering defects."""

    source: str
    sequence: tuple[str, ...]
    spans: tuple[StageSpan, ...]
    dwell_ratios: dict[str, float]
    unknown_stages: tuple[str, ...] = ()
    order_violations: tuple[str, ...] = ()
    skipped_stages: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "spans": [span.to_dict() for span in self.spans],
            "dwell_ratios": self.dwell_ratios,
            "unknown_stages": list(self.unknown_stages),
            "order_violations": list(self.order_violations),
            "skipped_stages": list(self.skipped_stages),
        }


def _spans_from_sequence(sequence: Sequence[str]) -> tuple[StageSpan, ...]:
    spans: list[StageSpan] = []
    for index, stage in enumerate(sequence):
        if spans and spans[-1].stage == stage:
            spans[-1] = StageSpan(stage, spans[-1].start_index, index)
        else:
            spans.append(StageSpan(stage, index, index))
    return tuple(spans)


def _explicit_stage_sequence(
    details: Mapping[str, Any], points: int
) -> tuple[str, ...] | None:
    """Read a per-point stage sequence out of evaluator details, if present.

    Two encodings are accepted: one stage name per progress point, or spans of
    the form {"stage": name, "start_index": i, "end_index": j} indexed against
    the progress curve. Anything else is treated as absent.
    """
    raw = details.get("stage_timeline")
    if raw is None or isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return None
    if not raw:
        return None
    if all(isinstance(item, str) for item in raw):
        return tuple(str(item).strip() for item in raw) if len(raw) == points else None

    sequence: list[str] = [""] * points
    for item in raw:
        if not isinstance(item, Mapping):
            return None
        stage = str(item.get("stage", "")).strip()
        start = item.get("start_index")
        end = item.get("end_index")
        if not stage or start is None or end is None:
            return None
        start_index = int(start)
        end_index = int(end)
        if not 0 <= start_index <= end_index < points:
            return None
        for index in range(start_index, end_index + 1):
            sequence[index] = stage
    return tuple(sequence) if all(sequence) else None


def build_stage_timeline(
    profile: TaskSemanticProfile,
    progress_curve: Sequence[float],
    details: Mapping[str, Any] | None = None,
) -> StageTimeline | None:
    """Assign each progress point to a stage and report ordering defects.

    An explicit timeline supplied by the evaluator wins. Otherwise the timeline
    is derived from the progress curve against the cumulative stage weights, in
    which case a backwards step only counts as an ordering violation when the
    accompanying progress drop is larger than curve noise.
    """
    values = [float(value) for value in progress_curve]
    if not values or not profile.stages:
        return None

    explicit = _explicit_stage_sequence(details or {}, len(values))
    if explicit is not None:
        sequence = explicit
        source = "evaluator"
    else:
        sequence = tuple(profile.stage_at_progress(value) for value in values)
        source = "derived_from_progress"

    ranks = [profile.stage_rank(stage) for stage in sequence]
    unknown_stages = tuple(
        dict.fromkeys(stage for stage, rank in zip(sequence, ranks) if rank is None)
    )

    order_violations: list[str] = []
    for index in range(1, len(sequence)):
        previous_rank = ranks[index - 1]
        current_rank = ranks[index]
        if previous_rank is None or current_rank is None or current_rank >= previous_rank:
            continue
        if source == "derived_from_progress":
            drop = values[index - 1] - values[index]
            if drop < PROGRESS_REGRESSION_DROP:
                continue
        order_violations.append(
            f"{sequence[index - 1]}->{sequence[index]}@{index}"
        )

    skipped_stages: tuple[str, ...] = ()
    if len(values) >= 2 * len(profile.stages):
        present = {rank for rank in ranks if rank is not None}
        if present:
            skipped_stages = tuple(
                profile.stages[rank].name
                for rank in range(min(present), max(present) + 1)
                if rank not in present
            )

    spans = _spans_from_sequence(sequence)
    dwell_counts: dict[str, int] = {}
    for stage in sequence:
        dwell_counts[stage] = dwell_counts.get(stage, 0) + 1
    dwell_ratios = {
        stage: count / len(sequence) for stage, count in sorted(dwell_counts.items())
    }
    return StageTimeline(
        source=source,
        sequence=sequence,
        spans=spans,
        dwell_ratios=dwell_ratios,
        unknown_stages=unknown_stages,
        order_violations=tuple(order_violations),
        skipped_stages=skipped_stages,
    )


def stage_timeline_findings(timeline: StageTimeline) -> tuple[str, ...]:
    findings: list[str] = []
    if timeline.unknown_stages:
        findings.append("semantic_stage_unknown_review")
    if timeline.order_violations:
        findings.append("semantic_stage_order_violation_review")
    if timeline.skipped_stages:
        findings.append("semantic_stage_skipped_review")
    return tuple(findings)


def dwell_outliers(
    samples: Mapping[tuple[str, str], list[tuple[int, float]]]
) -> dict[int, list[str]]:
    """Flag per-stage dwell ratios that are far from their task/stage median.

    Keys are (task_profile_key, stage_name); values are (episode_index, ratio).
    A deviation must clear both a robust MAD band and an absolute floor so that
    a tight distribution does not turn ordinary jitter into findings.
    """
    flagged: dict[int, list[str]] = {}
    for (task_key, stage), entries in samples.items():
        ratios = np.asarray([ratio for _, ratio in entries], dtype=np.float64)
        if ratios.size < STAGE_DWELL_MIN_SAMPLES:
            continue
        median = float(np.median(ratios))
        mad = float(np.median(np.abs(ratios - median)))
        band = max(STAGE_DWELL_MAD_SCALE * mad, STAGE_DWELL_MIN_DEVIATION)
        for episode_index, ratio in entries:
            deviation = float(ratio) - median
            if abs(deviation) <= band:
                continue
            direction = "long" if deviation > 0 else "short"
            flagged.setdefault(episode_index, []).append(
                f"{task_key}:{stage}:{direction}:{ratio:.3f}!={median:.3f}"
            )
    return flagged


# ---------------------------------------------------------------- motion signature


@dataclass(frozen=True)
class MotionSignature:
    """What a verb implies about gripper actuation, in direction-free terms.

    Gripper transitions are counted between the two levels the channel actually
    visits, so no assumption is made about whether a high value means open or
    closed. `vertical_direction` is only checked when the caller declares which
    action dimension is vertical.
    """

    name: str
    family: str
    terms: tuple[str, ...]
    min_transitions: int = 0
    max_transitions: int | None = None
    min_round_trips: int = 0
    vertical_direction: int = 0


MOTION_LEXICON: tuple[MotionSignature, ...] = (
    MotionSignature(
        name="grasp",
        family="grasp",
        terms=("pick", "pick up", "grasp", "grab", "take", "抓", "抓取", "拿", "拿起", "夹", "夹起", "捡"),
        min_transitions=1,
    ),
    MotionSignature(
        name="lift",
        family="grasp",
        terms=("lift", "raise", "提起", "举起", "抬起"),
        min_transitions=1,
        vertical_direction=1,
    ),
    MotionSignature(
        name="release",
        family="release",
        terms=("place", "put", "put down", "drop", "release", "放", "放下", "放置", "放入", "松开"),
        min_transitions=1,
    ),
    MotionSignature(
        name="stack",
        family="release",
        terms=("stack", "堆", "堆叠", "叠放"),
        min_transitions=2,
        min_round_trips=1,
    ),
    MotionSignature(
        name="insert",
        family="release",
        terms=("insert", "plug", "插", "插入"),
        min_transitions=1,
    ),
    MotionSignature(
        name="push",
        family="static",
        terms=("push", "shove", "推", "推动"),
        max_transitions=0,
    ),
    MotionSignature(
        name="press",
        family="static",
        terms=("press", "按", "按下", "按压"),
        max_transitions=0,
        vertical_direction=-1,
    ),
    MotionSignature(
        name="wipe",
        family="static",
        terms=("wipe", "sweep", "擦", "擦拭", "扫"),
        max_transitions=0,
    ),
    MotionSignature(
        name="pour",
        family="static",
        terms=("pour", "倒", "倾倒"),
        max_transitions=0,
    ),
)


@dataclass(frozen=True)
class ResolvedSignature:
    names: tuple[str, ...]
    min_transitions: int
    max_transitions: int | None
    min_round_trips: int
    vertical_direction: int

    @property
    def conflicted(self) -> bool:
        return self.max_transitions is not None and self.min_transitions > self.max_transitions

    def to_dict(self) -> dict[str, Any]:
        return {
            "verbs": list(self.names),
            "min_transitions": self.min_transitions,
            "max_transitions": self.max_transitions,
            "min_round_trips": self.min_round_trips,
            "vertical_direction": self.vertical_direction,
        }


_VERB_INFLECTION = r"(?:s|es|ed|ing)?"


def _term_matches(term: str, text: str) -> bool:
    """Match a lexicon term, allowing regular inflection on the leading verb.

    Instructions are usually imperative ("pick up the block"), so this only has
    to cover the regular forms; irregular stems simply fail to match and the
    episode is reported as not-checked rather than mis-checked.
    """
    if not term.isascii():
        return term in text
    words = term.split()
    pattern = re.escape(words[0]) + _VERB_INFLECTION
    if len(words) > 1:
        pattern += r"\s+" + r"\s+".join(re.escape(word) for word in words[1:])
    return re.search(rf"\b{pattern}\b", text) is not None


def resolve_motion_signature(texts: Sequence[str]) -> ResolvedSignature | None:
    """Compose the signatures of every verb found in the task text.

    Requirements combine conservatively: the strictest minimum and the strictest
    maximum win. A grasp verb together with a release verb implies a full
    open/close round trip even when neither verb demands one on its own.
    """
    haystack = " ".join(" ".join(str(text).casefold().split()) for text in texts if text)
    if not haystack:
        return None
    matched = tuple(
        signature
        for signature in MOTION_LEXICON
        if any(_term_matches(term, haystack) for term in signature.terms)
    )
    if not matched:
        return None

    min_transitions = max(signature.min_transitions for signature in matched)
    maximums = [
        signature.max_transitions
        for signature in matched
        if signature.max_transitions is not None
    ]
    max_transitions = min(maximums) if maximums else None
    min_round_trips = max(signature.min_round_trips for signature in matched)
    families = {signature.family for signature in matched}
    if {"grasp", "release"}.issubset(families):
        min_transitions = max(min_transitions, 2)
        min_round_trips = max(min_round_trips, 1)

    directions = {
        signature.vertical_direction
        for signature in matched
        if signature.vertical_direction
    }
    vertical_direction = directions.pop() if len(directions) == 1 else 0
    return ResolvedSignature(
        names=tuple(signature.name for signature in matched),
        min_transitions=min_transitions,
        max_transitions=max_transitions,
        min_round_trips=min_round_trips,
        vertical_direction=vertical_direction,
    )


@dataclass(frozen=True)
class GripperTrace:
    """Level-change statistics for one candidate gripper channel."""

    index: int
    minimum: float
    maximum: float
    span: float
    extreme_ratio: float
    transitions: int
    round_trips: int
    dominant_level_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "span": self.span,
            "extreme_ratio": self.extreme_ratio,
            "transitions": self.transitions,
            "round_trips": self.round_trips,
            "dominant_level_ratio": self.dominant_level_ratio,
        }


def gripper_trace(values: np.ndarray, index: int) -> GripperTrace | None:
    """Count level changes in one channel using its own min/max as the scale."""
    series = np.asarray(values, dtype=np.float64).ravel()
    if series.size < 2 or not np.all(np.isfinite(series)):
        return None
    minimum = float(series.min())
    maximum = float(series.max())
    span = maximum - minimum
    if span <= 0.0:
        return GripperTrace(
            index=index,
            minimum=minimum,
            maximum=maximum,
            span=0.0,
            extreme_ratio=1.0,
            transitions=0,
            round_trips=0,
            dominant_level_ratio=1.0,
        )

    normalized = (series - minimum) / span
    low_level = 0.5 - GRIPPER_LEVEL_MARGIN
    high_level = 0.5 + GRIPPER_LEVEL_MARGIN
    low_count = int(np.count_nonzero(normalized <= low_level))
    high_count = int(np.count_nonzero(normalized >= high_level))
    extreme_ratio = (low_count + high_count) / series.size

    levels: list[str] = []
    for value in normalized:
        if value <= low_level:
            level = "low"
        elif value >= high_level:
            level = "high"
        else:
            continue
        if not levels or levels[-1] != level:
            levels.append(level)
    transitions = max(len(levels) - 1, 0)
    return GripperTrace(
        index=index,
        minimum=minimum,
        maximum=maximum,
        span=span,
        extreme_ratio=extreme_ratio,
        transitions=transitions,
        round_trips=transitions // 2,
        dominant_level_ratio=max(low_count, high_count) / series.size,
    )


def channel_traces(action: np.ndarray | None) -> tuple[GripperTrace, ...]:
    """Trace every action dimension so the gripper channel can be chosen later."""
    if action is None or action.ndim != 2 or action.shape[0] < 2 or action.shape[1] == 0:
        return ()
    traces = [gripper_trace(action[:, index], index) for index in range(action.shape[1])]
    return tuple(trace for trace in traces if trace is not None)


def select_gripper_index(
    traces_by_episode: Mapping[int, Sequence[GripperTrace]]
) -> int | None:
    """Pick the action dimension that behaves like a gripper across the dataset.

    A gripper channel spends nearly all of its time at one of two levels. The
    winner must clear that bar on the median episode and must actually move in
    at least one episode, otherwise no channel is claimed.
    """
    ratios: dict[int, list[float]] = {}
    spans: dict[int, list[float]] = {}
    for traces in traces_by_episode.values():
        for trace in traces:
            ratios.setdefault(trace.index, []).append(trace.extreme_ratio)
            spans.setdefault(trace.index, []).append(trace.span)
    if not ratios:
        return None
    scored = [
        (float(np.median(values)), index)
        for index, values in ratios.items()
        if max(spans[index]) > 0.0
    ]
    if not scored:
        return None
    best_ratio, best_index = max(scored, key=lambda item: (item[0], -item[1]))
    return best_index if best_ratio >= GRIPPER_BIMODAL_RATIO else None


def resolve_actuation(traces: Sequence[GripperTrace], index: int, median_span: float) -> GripperTrace | None:
    """Return the trace for `index`, zeroing transitions on an inactive gripper.

    An episode whose gripper barely moves is renormalized noise, not a sequence
    of grasps, so its self-normalized transition count is discarded.
    """
    trace = next((item for item in traces if item.index == index), None)
    if trace is None:
        return None
    if median_span > 0.0 and trace.span < GRIPPER_MIN_SPAN_FRACTION * median_span:
        return GripperTrace(
            index=trace.index,
            minimum=trace.minimum,
            maximum=trace.maximum,
            span=trace.span,
            extreme_ratio=trace.extreme_ratio,
            transitions=0,
            round_trips=0,
            dominant_level_ratio=1.0,
        )
    return trace


def gripper_pattern_findings(
    signature: ResolvedSignature, trace: GripperTrace
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Compare observed gripper actuation against what the verb requires."""
    mismatches: list[str] = []
    if trace.transitions < signature.min_transitions:
        mismatches.append(
            f"transitions={trace.transitions}<min={signature.min_transitions}"
        )
    if signature.max_transitions is not None and trace.transitions > signature.max_transitions:
        mismatches.append(
            f"transitions={trace.transitions}>max={signature.max_transitions}"
        )
    if trace.round_trips < signature.min_round_trips:
        mismatches.append(
            f"round_trips={trace.round_trips}<min={signature.min_round_trips}"
        )
    if not mismatches:
        return (), {}
    return ("semantic_motion_gripper_pattern_review",), {"mismatches": mismatches}


def vertical_excursions(values: np.ndarray) -> tuple[float, float]:
    """Largest rise and largest fall relative to the first frame."""
    series = np.asarray(values, dtype=np.float64).ravel()
    if series.size < 2 or not np.all(np.isfinite(series)):
        return 0.0, 0.0
    start = float(series[0])
    return max(float(series.max()) - start, 0.0), max(start - float(series.min()), 0.0)


def vertical_direction_findings(
    signature: ResolvedSignature,
    excursions: tuple[float, float] | None,
    vertical_index: int,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Check that a directional verb moved the declared vertical axis that way.

    The comparison is between the rise and the fall of the same channel, so it
    needs no unit or calibration for that axis.
    """
    if signature.vertical_direction == 0 or excursions is None:
        return (), {}
    rise, fall = excursions
    expected, opposite = (rise, fall) if signature.vertical_direction > 0 else (fall, rise)
    metrics = {
        "vertical_index": vertical_index,
        "expected_direction": signature.vertical_direction,
        "rise": rise,
        "fall": fall,
    }
    if opposite > 0.0 and expected < MOTION_VERTICAL_DOMINANCE * opposite:
        return ("semantic_motion_vertical_direction_review",), metrics
    return (), metrics


# ---------------------------------------------------------------- boundary alignment


def kinematic_change_points(motion: EpisodeMotion) -> tuple[int, ...]:
    """Frames where the arm plausibly starts or ends a sub-motion.

    Two sources are merged: gripper level changes, and local minima of the
    non-gripper action speed that fall in the slowest quartile of the episode.
    A minimum must be a genuine dip, so a constant-speed arm contributes nothing
    rather than marking every frame as a change point.
    """
    action = motion.action
    if action is None or action.ndim != 2 or action.shape[0] < 3:
        return ()

    change_points: set[int] = set()
    gripper = motion.gripper_series()
    if gripper is not None:
        series = np.asarray(gripper, dtype=np.float64)
        span = float(series.max() - series.min())
        if span > 0.0:
            normalized = (series - float(series.min())) / span
            state: str | None = None
            for frame, value in enumerate(normalized):
                if value <= 0.5 - GRIPPER_LEVEL_MARGIN:
                    level = "low"
                elif value >= 0.5 + GRIPPER_LEVEL_MARGIN:
                    level = "high"
                else:
                    continue
                if state is not None and level != state:
                    change_points.add(frame)
                state = level

    arm_columns = [
        index
        for index in range(action.shape[1])
        if motion.gripper_index is None or index != motion.gripper_index
    ]
    if arm_columns:
        arm = action[:, arm_columns]
        speed = np.linalg.norm(np.diff(arm, axis=0), axis=1)
        if speed.size >= 3 and np.all(np.isfinite(speed)):
            scale = float(speed.max() - speed.min())
            if scale > max(1e-9, 1e-6 * abs(float(np.median(speed)))):
                slow = float(np.quantile(speed, BOUNDARY_SPEED_QUANTILE))
                for frame in range(1, speed.size - 1):
                    if speed[frame] > slow:
                        continue
                    if speed[frame] > speed[frame - 1] or speed[frame] > speed[frame + 1]:
                        continue
                    dip = max(speed[frame - 1], speed[frame + 1]) - speed[frame]
                    if dip >= BOUNDARY_MIN_DIP_RATIO * scale:
                        change_points.add(frame + 1)
    return tuple(sorted(change_points))


@dataclass(frozen=True)
class BoundaryAlignment:
    boundaries: int
    aligned: int
    max_distance: int
    change_points: int
    tolerance_frames: int

    @property
    def aligned_ratio(self) -> float:
        return self.aligned / self.boundaries if self.boundaries else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundaries": self.boundaries,
            "aligned": self.aligned,
            "aligned_ratio": self.aligned_ratio,
            "max_distance": self.max_distance,
            "change_points": self.change_points,
            "tolerance_frames": self.tolerance_frames,
        }


def subtask_boundary_alignment(
    boundaries: Sequence[int],
    change_points: Sequence[int],
    tolerance_frames: int = BOUNDARY_TOLERANCE_FRAMES,
) -> BoundaryAlignment | None:
    """Score how well annotated segment boundaries land on real motion changes.

    Boundaries invented by a language model tend to fall in the middle of a
    smooth motion; boundaries that describe what happened tend to sit next to a
    gripper event or a near-stop.
    """
    if not boundaries or not change_points:
        return None
    points = np.asarray(sorted(change_points), dtype=np.int64)
    aligned = 0
    max_distance = 0
    for boundary in boundaries:
        distance = int(np.min(np.abs(points - int(boundary))))
        max_distance = max(max_distance, distance)
        if distance <= tolerance_frames:
            aligned += 1
    return BoundaryAlignment(
        boundaries=len(boundaries),
        aligned=aligned,
        max_distance=max_distance,
        change_points=len(points),
        tolerance_frames=tolerance_frames,
    )


def boundary_alignment_findings(alignment: BoundaryAlignment) -> tuple[str, ...]:
    if alignment.aligned_ratio < BOUNDARY_ALIGNMENT_MIN_RATIO:
        return ("language_subtask_boundary_unaligned_review",)
    return ()
