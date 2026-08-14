"""Semantic-audit evaluator backed by a trained VLA policy checkpoint.

A VLA checkpoint (e.g. a lerobot ``pi05``/``smolvla``/``act`` policy under
``checkpoint for audit/pretrained_model``) predicts actions from images, robot
state, and the task instruction. It has no yes/no interface, so this backend
scores an episode by **instruction-conditioned action consistency**:

1. Sample anchor frames across the episode. At each anchor, feed the camera
   frames, the joint state, and the dataset task text to the policy and let it
   predict a short action chunk.
2. Compare the predicted chunk against the actions the demonstrator actually
   executed. The per-anchor skill is ``1 - mae(policy) / mae(hold-baseline)``
   clipped to [0, 1], where the baseline repeats the action at the anchor.
   1.0 means the checkpoint reproduces the demonstration; 0.0 means it does no
   better than freezing the arm. Anchors where the demonstration itself barely
   moves are uninformative and skipped.
3. The mean skill becomes ``success_probability``. Interleaved anchor subsets
   become ``score_samples``, so the audit's dispersion contract flags episodes
   whose agreement swings across time.
4. The same anchors are re-scored under deliberately corrupted instructions
   (reversed verb, swapped direction/color/object). ``counterfactual_margin``
   is the true score minus the best corrupted score: an episode whose actions
   are predicted equally well under a wrong instruction is not grounded in its
   language label and is reviewed by the audit.

No ``progress_curve`` is returned — action agreement is not task progress — so
the audit's progress-shape and stage-timeline checks correctly report
themselves as not-checked instead of misfiring.

Usage::

    python ./pipeline/data_quality_audit.py ./my_dataset \
      --semantic-evaluator semantic_backends.vla_checkpoint_evaluator:create_evaluator \
      --semantic-config ./configs/vla_checkpoint_evaluator.example.json

The factory needs an environment with the policy runtime installed
(``pip install lerobot`` plus the policy's extras); everything importable in
this module stays dependency-free so the contract and math remain testable
without torch.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from utils.dataset_io import read_json
from utils.media_utils import ffmpeg_available, format_seconds
from utils.quality_utils import evenly_spaced_indices

IMAGE_PREFIX = "observation.images."
STATE_KEY = "observation.state"
ACTION_KEY = "action"

DEFAULT_ANCHORS = 6
DEFAULT_EVAL_HORIZON = 10
DEFAULT_SAMPLE_SPLITS = 3
DEFAULT_MAX_COUNTERFACTUALS = 3
STATIC_EPSILON = 1e-6


# ------------------------------------------------------------- counterfactuals

_VERB_SWAPS: tuple[tuple[str, str], ...] = (
    ("pick up", "put down"),
    ("open", "close"),
    ("push", "pull"),
    ("insert", "remove"),
    ("lift", "lower"),
    ("stack", "unstack"),
    ("拿起", "放下"),
    ("抓起", "放下"),
    ("拾起", "放下"),
    ("抬起", "放下"),
    ("打开", "关闭"),
    ("插入", "拔出"),
    ("推", "拉"),
)

_DIRECTION_SWAPS: tuple[tuple[str, str], ...] = (
    ("left", "right"),
    ("up", "down"),
    ("front", "back"),
    ("左", "右"),
    ("上", "下"),
    ("前", "后"),
)

_COLOR_SWAPS: tuple[tuple[str, str], ...] = (
    ("red", "blue"),
    ("green", "yellow"),
    ("black", "white"),
    ("红色", "蓝色"),
    ("绿色", "黄色"),
    ("黑色", "白色"),
)

_GENERIC_NOUNS: tuple[str, ...] = (
    "cup",
    "bottle",
    "block",
    "bowl",
    "plate",
    "drawer",
    "box",
    "towel",
    "杯子",
    "瓶子",
    "方块",
    "碗",
    "盘子",
    "抽屉",
    "箱子",
    "毛巾",
)


def _swap_term(text: str, source: str, target: str) -> str | None:
    """Replace ``source`` with ``target``; None when the swap changes nothing."""
    if source.isascii():
        pattern = re.compile(rf"\b{re.escape(source)}\b", re.IGNORECASE)
        if not pattern.search(text):
            return None
        corrupted = pattern.sub(target, text)
    else:
        if source not in text:
            return None
        corrupted = text.replace(source, target)
    return corrupted if corrupted != text else None


def _pair_corruptions(text: str, pairs: Sequence[tuple[str, str]]) -> list[str]:
    variants: list[str] = []
    for left, right in pairs:
        corrupted = _swap_term(text, left, right) or _swap_term(text, right, left)
        if corrupted is not None:
            variants.append(corrupted)
    return variants


def _noun_corruption(text: str, extra_nouns: Sequence[str]) -> str | None:
    """Replace the first noun found with a different noun absent from the text.

    Replacements from the same script as the matched noun are preferred so a
    Chinese instruction receives a Chinese distractor object.
    """
    nouns = tuple(dict.fromkeys((*extra_nouns, *_GENERIC_NOUNS)))
    lowered = text.casefold()
    present = [noun for noun in nouns if noun and noun.casefold() in lowered]
    if not present:
        return None
    matched = present[0]
    absent = [noun for noun in nouns if noun and noun.casefold() not in lowered]
    same_script = [noun for noun in absent if noun.isascii() == matched.isascii()]
    pool = same_script or absent
    if not pool:
        return None
    return _swap_term(text, matched, pool[0])


def corrupt_instructions(
    text: str, object_names: Sequence[str] = (), limit: int = DEFAULT_MAX_COUNTERFACTUALS
) -> tuple[str, ...]:
    """Deterministically corrupted variants of one instruction.

    Verb reversals come first because they are the strongest semantic flip,
    then direction and color swaps, then an object substitution. Variants are
    deduplicated and capped at ``limit``.
    """
    text = " ".join(str(text).split())
    if not text or limit <= 0:
        return ()
    candidates: list[str] = []
    candidates.extend(_pair_corruptions(text, _VERB_SWAPS))
    candidates.extend(_pair_corruptions(text, _DIRECTION_SWAPS))
    candidates.extend(_pair_corruptions(text, _COLOR_SWAPS))
    noun_variant = _noun_corruption(text, object_names)
    if noun_variant is not None:
        candidates.append(noun_variant)
    unique = tuple(dict.fromkeys(item for item in candidates if item != text))
    return unique[:limit]


# --------------------------------------------------------------- scoring math


def chunk_skill(
    predicted: np.ndarray,
    executed: np.ndarray,
    anchor: int,
    horizon: int,
    epsilon: float = STATIC_EPSILON,
) -> float | None:
    """Agreement of one predicted chunk with the executed actions, in [0, 1].

    The hold baseline repeats the executed action at the anchor, so the score
    is scale-free: 1.0 reproduces the demonstrated motion, 0.0 is no better
    than freezing. None marks an uninformative anchor where the demonstration
    itself barely moves over the horizon.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    executed = np.asarray(executed, dtype=np.float64)
    steps = min(horizon, predicted.shape[0], executed.shape[0] - anchor)
    if steps <= 0:
        return None
    target = executed[anchor : anchor + steps]
    hold = np.repeat(executed[anchor : anchor + 1], steps, axis=0)
    mae_hold = float(np.mean(np.abs(hold - target)))
    if mae_hold <= epsilon:
        return None
    mae_policy = float(np.mean(np.abs(predicted[:steps] - target)))
    return float(np.clip(1.0 - mae_policy / mae_hold, 0.0, 1.0))


def informative_anchors(
    executed: np.ndarray, anchors: Sequence[int], horizon: int, epsilon: float = STATIC_EPSILON
) -> tuple[int, ...]:
    """Anchors whose demonstration moves enough to make the skill well-defined.

    Informativeness depends only on the executed actions, so the same anchor
    set is valid for the true instruction and for every corrupted one.
    """
    executed = np.asarray(executed, dtype=np.float64)
    kept: list[int] = []
    for anchor in anchors:
        steps = min(horizon, executed.shape[0] - anchor)
        if steps <= 0:
            continue
        target = executed[anchor : anchor + steps]
        hold = np.repeat(executed[anchor : anchor + 1], steps, axis=0)
        if float(np.mean(np.abs(hold - target))) > epsilon:
            kept.append(int(anchor))
    return tuple(kept)


def interleaved_means(values: Sequence[float], splits: int) -> tuple[float, ...]:
    """Means of interleaved subsets, used as the audit's score samples."""
    items = [float(value) for value in values]
    if splits <= 1 or len(items) < 2:
        return ()
    groups = [items[offset::splits] for offset in range(splits)]
    means = tuple(float(np.mean(group)) for group in groups if group)
    return means if len(means) >= 2 else ()


def rename_camera_features(
    features: Mapping[str, Any], camera_map: Mapping[str, str]
) -> dict[str, Any]:
    """Rename ``observation.images.<cam>`` features per ``camera_map``.

    ``camera_map`` maps dataset camera names to the names the policy was
    trained with (e.g. ``{"front": "head"}``); other features pass through.
    """
    if not camera_map:
        return dict(features)
    renamed: dict[str, Any] = {}
    for key, spec in features.items():
        if key.startswith(IMAGE_PREFIX):
            camera = key[len(IMAGE_PREFIX) :]
            renamed[IMAGE_PREFIX + camera_map.get(camera, camera)] = spec
        else:
            renamed[key] = spec
    return renamed


def _feature_names(features: Mapping[str, Any], key: str, dim: int, prefix: str) -> list[str]:
    spec = features.get(key)
    names = spec.get("names") if isinstance(spec, Mapping) else None
    if isinstance(names, list) and len(names) == dim and all(isinstance(n, str) for n in names):
        return list(names)
    return [f"{prefix}_{index}" for index in range(dim)]


# ------------------------------------------------------------------ contracts


@runtime_checkable
class ActionChunkPredictor(Protocol):
    """Minimal interface the evaluator needs from a policy runtime."""

    def predict_chunk(
        self,
        observation: Mapping[str, Any],
        task: str,
        horizon: int,
        *,
        features: Mapping[str, Any],
        action_names: Sequence[str],
    ) -> np.ndarray:
        """Predict ``[horizon, action_dim]`` actions for one observation."""


@runtime_checkable
class FrameReader(Protocol):
    def read_frame(self, path: Path, seconds: float, height: int, width: int) -> np.ndarray:
        """Return one ``(height, width, 3)`` uint8 RGB frame at ``seconds``."""


class FfmpegFrameReader:
    """Decode single frames with the same ffmpeg the rest of the audit uses."""

    def __init__(self, ffmpeg_path: str | None = None) -> None:
        self.ffmpeg_path = ffmpeg_path or ffmpeg_available()
        if self.ffmpeg_path is None:
            raise FileNotFoundError("ffmpeg is required to sample frames for the VLA evaluator")

    def read_frame(self, path: Path, seconds: float, height: int, width: int) -> np.ndarray:
        cmd = [
            self.ffmpeg_path,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            format_seconds(max(0.0, seconds)),
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        frame_bytes = height * width * 3
        if result.returncode != 0 or len(result.stdout) < frame_bytes:
            message = result.stderr.decode("utf-8", errors="replace").strip()[:300]
            raise RuntimeError(f"cannot decode frame at {seconds:.3f}s from {path}: {message}")
        frame = np.frombuffer(result.stdout[:frame_bytes], dtype=np.uint8)
        return frame.reshape((height, width, 3))


class _DatasetMetadataShim:
    """Duck-typed stand-in for ``LeRobotDatasetMetadata``.

    ``make_policy`` only reads ``.features`` (lerobot dataset feature dicts —
    exactly the ``features`` block of the audited dataset's ``info.json``) and
    an optional ``.stats``. Normalization statistics for a pretrained pi0/pi05
    checkpoint live in its saved pre/post processor safetensors, so no dataset
    stats are required here.
    """

    def __init__(self, features: Mapping[str, Any], stats: Mapping[str, Any] | None = None):
        normalized: dict[str, Any] = {}
        for key, spec in features.items():
            spec = dict(spec) if isinstance(spec, Mapping) else spec
            if isinstance(spec, dict) and spec.get("dtype") in {"image", "video"} and not spec.get("names"):
                # Toolkit datasets store frames as (H, W, C); lerobot needs the
                # axis names to convert the feature shape to channel-first.
                spec["names"] = ["height", "width", "channels"]
            normalized[key] = spec
        self.features = normalized
        self.stats = stats


class LeRobotChunkPredictor:
    """Default adapter running a lerobot policy checkpoint (lazy imports).

    Mirrors ``record.recorder.policy_runner.LeRobotPolicyRunner``: chunking
    policies (pi0/pi05/smolvla/act) compute a chunk on the first
    ``predict_action`` call and pop the following steps from their internal
    queue, so calling it ``horizon`` times against one observation yields the
    chunk predicted at that anchor.

    Building the policy is deferred to the first ``predict_chunk`` call because
    ``make_policy`` needs dataset metadata to resolve feature shapes, and the
    audited dataset's features only become known per episode.
    """

    def __init__(self, policy_path: str, *, device: str | None = None) -> None:
        try:
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import make_policy, make_pre_post_processors
            from lerobot.utils.control_utils import predict_action
            from lerobot.utils.device_utils import get_safe_torch_device
        except ImportError as exc:
            raise ImportError(
                "the VLA checkpoint evaluator runs a lerobot policy and needs the policy "
                "runtime installed (`pip install lerobot` plus the policy's extras, e.g. "
                "pi05 weights need transformers). Run this semantic audit stage in your "
                "training environment; the structural audit does not need it."
            ) from exc

        cfg = PreTrainedConfig.from_pretrained(policy_path)
        cfg.pretrained_path = policy_path
        if device:
            cfg.device = device
        self._cfg = cfg
        self._make_policy = make_policy
        self._policy: Any = None
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=policy_path
        )
        self._predict_action = predict_action
        self._device = get_safe_torch_device(cfg.device)

    def _ensure_policy(self, features: Mapping[str, Any]) -> None:
        if self._policy is not None:
            return
        import contextlib
        import io

        shim = _DatasetMetadataShim(features)
        # Some policy implementations print unicode progress marks while
        # loading and swallow *any* exception — including a Windows console
        # UnicodeEncodeError — by silently returning an unloaded model.
        # Capturing stdout removes the encoding hazard and lets the failure
        # message be detected instead of ignored.
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            try:
                policy = self._make_policy(self._cfg, ds_meta=shim)
            except TypeError:
                policy = self._make_policy(self._cfg)
        log = buffer.getvalue()
        failure_phrases = (
            "without loading pretrained weights",
            "Could not load state dict",
        )
        if any(phrase in log for phrase in failure_phrases):
            condensed = " | ".join(line for line in log.splitlines() if line.strip())[:1000]
            raise RuntimeError(
                f"policy checkpoint weights failed to load from "
                f"{self._cfg.pretrained_path}: {condensed}"
            )
        self._policy = policy

    def predict_chunk(
        self,
        observation: Mapping[str, Any],
        task: str,
        horizon: int,
        *,
        features: Mapping[str, Any],
        action_names: Sequence[str],
    ) -> np.ndarray:
        from lerobot.datasets.feature_utils import build_dataset_frame
        from lerobot.policies.utils import make_robot_action
        from lerobot.utils.constants import OBS_STR

        self._ensure_policy(features)
        frame = build_dataset_frame(dict(features), dict(observation), prefix=OBS_STR)
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        rows: list[list[float]] = []
        for _ in range(horizon):
            values = self._predict_action(
                observation=frame,
                policy=self._policy,
                device=self._device,
                preprocessor=self._preprocessor,
                postprocessor=self._postprocessor,
                use_amp=getattr(self._cfg, "use_amp", False),
                task=task,
            )
            robot_action = make_robot_action(values, dict(features))
            rows.append([float(robot_action[name]) for name in action_names])
        return np.asarray(rows, dtype=np.float64)


# ------------------------------------------------------------------ evaluator


class VLACheckpointEvaluator:
    """Score episodes by how well a trained VLA reproduces their actions."""

    def __init__(
        self,
        predictor: ActionChunkPredictor,
        frame_reader: FrameReader,
        *,
        anchors: int = DEFAULT_ANCHORS,
        eval_horizon: int = DEFAULT_EVAL_HORIZON,
        sample_splits: int = DEFAULT_SAMPLE_SPLITS,
        max_counterfactuals: int = DEFAULT_MAX_COUNTERFACTUALS,
        camera_map: Mapping[str, str] | None = None,
        static_epsilon: float = STATIC_EPSILON,
        name: str = "vla_action_consistency",
    ) -> None:
        if anchors < 1:
            raise ValueError("anchors must be at least 1")
        if eval_horizon < 1:
            raise ValueError("eval_horizon must be at least 1")
        if sample_splits < 0 or max_counterfactuals < 0:
            raise ValueError("sample_splits and max_counterfactuals must be non-negative")
        self.predictor = predictor
        self.frame_reader = frame_reader
        self.anchors = int(anchors)
        self.eval_horizon = int(eval_horizon)
        self.sample_splits = int(sample_splits)
        self.max_counterfactuals = int(max_counterfactuals)
        self.camera_map = dict(camera_map or {})
        self.static_epsilon = float(static_epsilon)
        self.name = name
        self._dataset_cache: dict[Path, tuple[dict[str, Any], float]] = {}

    def _dataset_features(self, dataset_root: Path) -> tuple[dict[str, Any], float]:
        root = Path(dataset_root)
        cached = self._dataset_cache.get(root)
        if cached is None:
            info = read_json(root / "meta" / "info.json")
            features = info.get("features")
            if not isinstance(features, dict):
                raise ValueError(f"dataset info.json has no features block: {root}")
            fps = float(info.get("fps") or 30.0)
            cached = (features, fps)
            self._dataset_cache[root] = cached
        return cached

    def _anchor_observations(
        self,
        episode: Any,
        anchors: Sequence[int],
        features: Mapping[str, Any],
        fps: float,
        state: np.ndarray | None,
        state_names: Sequence[str],
    ) -> dict[int, dict[str, Any]]:
        observations: dict[int, dict[str, Any]] = {}
        for anchor in anchors:
            observation: dict[str, Any] = {}
            for video in episode.videos:
                spec = features.get(video.key)
                shape = spec.get("shape") if isinstance(spec, Mapping) else None
                if not isinstance(shape, list) or len(shape) < 2:
                    raise ValueError(f"video feature {video.key} has no usable shape")
                height, width = int(shape[0]), int(shape[1])
                seconds = float(video.from_timestamp) + anchor / fps
                frame = self.frame_reader.read_frame(video.path, seconds, height, width)
                camera = video.key[len(IMAGE_PREFIX) :] if video.key.startswith(IMAGE_PREFIX) else video.key
                observation[self.camera_map.get(camera, camera)] = frame
            if state is not None and anchor < state.shape[0]:
                observation.update(
                    {name: float(value) for name, value in zip(state_names, state[anchor])}
                )
            observations[anchor] = observation
        return observations

    def _score_instruction(
        self,
        instruction: str,
        anchors: Sequence[int],
        observations: Mapping[int, Mapping[str, Any]],
        executed: np.ndarray,
        horizon: int,
        features: Mapping[str, Any],
        action_names: Sequence[str],
    ) -> tuple[list[float], dict[str, float]]:
        skills: list[float] = []
        per_anchor: dict[str, float] = {}
        for anchor in anchors:
            predicted = self.predictor.predict_chunk(
                observations[anchor],
                instruction,
                horizon,
                features=features,
                action_names=action_names,
            )
            skill = chunk_skill(predicted, executed, anchor, horizon, self.static_epsilon)
            if skill is not None:
                skills.append(skill)
                per_anchor[str(anchor)] = round(skill, 6)
        return skills, per_anchor

    def evaluate(self, episode: Any) -> dict[str, Any]:
        if not episode.tasks:
            raise ValueError("episode has no task text")
        task = str(episode.tasks[0])
        motion = episode.motion
        if motion is None or motion.action is None:
            raise ValueError("episode action stream is unavailable; consistency scoring needs it")
        executed = np.asarray(motion.action, dtype=np.float64)
        if executed.ndim != 2 or executed.shape[0] < 2:
            raise ValueError("episode action stream is too short to score")
        if not episode.videos:
            raise ValueError("episode has no video segments for the policy to observe")

        features, dataset_fps = self._dataset_features(episode.dataset_root)
        fps = float(motion.fps or dataset_fps)
        total = executed.shape[0]
        horizon = min(self.eval_horizon, total - 1)
        usable = total - horizon
        if usable <= 0:
            raise ValueError("episode is shorter than the evaluation horizon")
        candidates = evenly_spaced_indices(usable, self.anchors)
        anchors = informative_anchors(executed, candidates, horizon, self.static_epsilon)
        if not anchors:
            raise ValueError(
                "no informative anchors: the demonstration is nearly static over every "
                "sampled horizon, so action-consistency is undefined"
            )

        state = (
            np.asarray(motion.state, dtype=np.float64)
            if motion.state is not None and np.asarray(motion.state).ndim == 2
            else None
        )
        state_names = _feature_names(
            features, STATE_KEY, state.shape[1] if state is not None else 0, "state"
        )
        action_names = _feature_names(features, ACTION_KEY, executed.shape[1], "action")
        effective_features = rename_camera_features(features, self.camera_map)
        observations = self._anchor_observations(
            episode, anchors, features, fps, state, state_names
        )

        true_skills, per_anchor = self._score_instruction(
            task, anchors, observations, executed, horizon, effective_features, action_names
        )
        if not true_skills:
            raise ValueError("no anchor produced a usable prediction")
        score = float(np.mean(true_skills))
        samples = interleaved_means(true_skills, self.sample_splits)

        details: dict[str, Any] = {
            "backend": "vla_action_consistency",
            "horizon": int(horizon),
            "anchors": [int(anchor) for anchor in anchors],
            "per_anchor_skill": per_anchor,
            "cameras": [video.key for video in episode.videos],
        }

        object_names = _profile_object_names(episode.task_profile)
        corrupted = corrupt_instructions(task, object_names, self.max_counterfactuals)
        if corrupted:
            corrupted_scores: dict[str, float] = {}
            for text in corrupted:
                skills, _ = self._score_instruction(
                    text, anchors, observations, executed, horizon, effective_features, action_names
                )
                corrupted_scores[text] = round(float(np.mean(skills)), 6) if skills else 0.0
            details["counterfactual_scores"] = corrupted_scores
            margin = float(np.clip(score - max(corrupted_scores.values()), -1.0, 1.0))
            details["counterfactual_margin"] = margin

        return {
            "success_probability": score,
            "score_samples": list(samples),
            "details": details,
        }


def _profile_object_names(task_profile: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(task_profile, Mapping):
        return ()
    objects = task_profile.get("objects")
    if not isinstance(objects, Sequence):
        return ()
    names: list[str] = []
    for item in objects:
        if isinstance(item, Mapping):
            name = str(item.get("name", "")).strip()
            if name:
                names.append(name)
    return tuple(names)


# -------------------------------------------------------------------- factory


def create_evaluator(config: Mapping[str, Any] | None = None) -> VLACheckpointEvaluator:
    """Build the evaluator from ``--semantic-config`` JSON.

    Required: ``policy_path`` — a lerobot pretrained-policy directory such as
    ``checkpoint for audit/pretrained_model``. Optional: ``device``,
    ``anchors``, ``eval_horizon``, ``sample_splits``, ``max_counterfactuals``,
    ``camera_map`` ({dataset_camera: policy_camera}), ``static_epsilon``.
    """
    options = dict(config or {})
    policy_path = str(options.pop("policy_path", "") or "").strip()
    if not policy_path:
        raise ValueError("vla checkpoint evaluator requires 'policy_path' in --semantic-config")
    path = Path(policy_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"policy checkpoint not found: {path}")
    device = options.pop("device", None)
    unknown = set(options) - {
        "anchors",
        "eval_horizon",
        "sample_splits",
        "max_counterfactuals",
        "camera_map",
        "static_epsilon",
    }
    if unknown:
        raise ValueError(f"unknown vla evaluator config keys: {sorted(unknown)}")
    predictor = LeRobotChunkPredictor(str(path), device=device)
    return VLACheckpointEvaluator(
        predictor,
        FfmpegFrameReader(),
        name=f"vla_action_consistency:{path.name}",
        **options,
    )
