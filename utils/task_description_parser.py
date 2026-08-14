"""Pluggable natural-language parsers for task semantic profiles."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from utils.task_semantic_profile import TaskObject, TaskSemanticProfile, TaskStage


@runtime_checkable
class TaskDescriptionParser(Protocol):
    name: str

    def parse(
        self,
        description: str,
        task_name: str = "",
        task_key: str = "",
    ) -> TaskSemanticProfile | Mapping[str, Any]:
        """Parse one detailed description into a validated profile draft."""


def load_parser(spec: str, config: Mapping[str, Any] | None = None) -> TaskDescriptionParser:
    module_name, separator, attribute_name = spec.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("task parser must use module:attribute syntax")
    module = importlib.import_module(module_name)
    target = getattr(module, attribute_name)
    options = dict(config or {})
    if inspect.isclass(target):
        candidate = target(**options)
    elif hasattr(target, "parse"):
        candidate = target
    elif callable(target):
        candidate = target(options)
    else:
        raise TypeError(f"task parser target is not usable: {spec}")
    if not callable(getattr(candidate, "parse", None)):
        raise TypeError(f"task parser does not define parse(): {spec}")
    return candidate


def load_parser_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("task parser config must contain one JSON object")
    return payload


def normalize_parser_result(
    value: TaskSemanticProfile | Mapping[str, Any],
) -> TaskSemanticProfile:
    if isinstance(value, TaskSemanticProfile):
        value.validate()
        return value
    if isinstance(value, Mapping):
        return TaskSemanticProfile.from_dict(value)
    raise TypeError("task parser must return TaskSemanticProfile or a mapping")


_STAGE_RULES: tuple[tuple[str, str], ...] = (
    ("approach", r"接近|靠近|找到|定位|approach|locate|find"),
    ("grasp", r"抓住|抓取|夹住|握住|grasp|grip|pick"),
    ("lift", r"抬起|拿起|提起|离开桌面|lift|raise"),
    ("transport", r"移动到|搬运|移至|移到|transport|move .* to|carry"),
    ("place", r"放入|放进|放置|插入|place|put|insert|lower .* into"),
    ("release", r"松开|释放|张开夹爪|release|open the gripper"),
    ("open", r"打开|拉开|open|pull .* open"),
    ("close", r"关闭|合上|close|shut"),
    ("push", r"推动|按下|推入|push|press"),
    ("pull", r"拉动|拉出|pull"),
    ("rotate", r"旋转|拧动|rotate|turn|twist"),
    ("clean", r"清理|擦拭|整理|clean|wipe|tidy"),
)

_OBJECT_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cup", ("杯子", "水杯", "cup")),
    ("storage_bin", ("收纳箱", "箱子", "bin", "box")),
    ("drawer", ("抽屉", "drawer")),
    ("handle", ("把手", "handle")),
    ("bottle", ("瓶子", "bottle")),
    ("towel", ("毛巾", "towel")),
    ("plate", ("盘子", "plate")),
    ("bowl", ("碗", "bowl")),
    ("block", ("方块", "积木", "block")),
    ("button", ("按钮", "button")),
    ("door", ("门", "door")),
    ("table", ("桌面", "桌子", "table")),
)

_ATTRIBUTE_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("red", ("红色", "red")),
    ("blue", ("蓝色", "blue")),
    ("green", ("绿色", "green")),
    ("yellow", ("黄色", "yellow")),
    ("left", ("左侧", "左边", "left")),
    ("right", ("右侧", "右边", "right")),
    ("transparent", ("透明", "transparent")),
)

_SUCCESS_MARKERS = re.compile(
    r"最终|最后|成功|完成后|必须|保持|应当|应该|at the end|finally|must|should remain|success",
    re.IGNORECASE,
)
_FAILURE_MARKERS = re.compile(
    r"失败|不能|不得|避免|禁止|掉落|外部|outside|fail|must not|avoid|drop",
    re.IGNORECASE,
)


def _contains_cjk(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", value))


def _slugify(value: str, fallback_source: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if slug:
        return slug[:80]
    digest = hashlib.sha256(fallback_source.encode("utf-8")).hexdigest()[:12]
    return f"task_{digest}"


def _split_clauses(description: str) -> list[str]:
    primary = re.split(r"[\n。！？!?；;，,]+", description)
    clauses: list[str] = []
    for item in primary:
        for part in re.split(r"(?:然后|随后|接着|并且|并|and then|then)", item, flags=re.IGNORECASE):
            cleaned = part.strip(" .")
            if cleaned:
                clauses.append(cleaned)
    return clauses


def _stage_name(clause: str) -> str | None:
    for name, pattern in _STAGE_RULES:
        if re.search(pattern, clause, re.IGNORECASE):
            return name
    return None


def _extract_objects(description: str) -> tuple[TaskObject, ...]:
    lowered = description.casefold()
    objects: list[TaskObject] = []
    for object_name, terms in _OBJECT_TERMS:
        positions = [lowered.find(term.casefold()) for term in terms]
        positions = [position for position in positions if position >= 0]
        if not positions:
            continue
        position = min(positions)
        window = lowered[max(0, position - 12) : position + 12]
        attributes = tuple(
            attribute
            for attribute, attribute_terms in _ATTRIBUTE_TERMS
            if any(term.casefold() in window for term in attribute_terms)
        )
        objects.append(TaskObject(name=object_name, attributes=attributes))
    return tuple(objects)


class HeuristicTaskDescriptionParser:
    """Dependency-free parser that creates a conservative editable draft.

    It extracts explicit action clauses and observable terminal requirements.
    The result is intended for review in the task semantic studio, not as a
    substitute for a language model or human confirmation.
    """

    name = "heuristic_task_description_parser"

    def __init__(
        self,
        review_threshold: float | None = None,
        pass_threshold: float | None = None,
    ) -> None:
        if review_threshold is not None and not 0.0 <= review_threshold <= 1.0:
            raise ValueError("review_threshold must be within [0, 1]")
        if pass_threshold is not None and not 0.0 <= pass_threshold <= 1.0:
            raise ValueError("pass_threshold must be within [0, 1]")
        if (
            review_threshold is not None
            and pass_threshold is not None
            and review_threshold > pass_threshold
        ):
            raise ValueError("review_threshold must not exceed pass_threshold")
        self.review_threshold = review_threshold
        self.pass_threshold = pass_threshold

    def parse(
        self,
        description: str,
        task_name: str = "",
        task_key: str = "",
    ) -> TaskSemanticProfile:
        description = " ".join(description.strip().split())
        if not description:
            raise ValueError("task description is empty")
        clauses = _split_clauses(description)
        chinese = _contains_cjk(description)

        stages: list[TaskStage] = []
        stage_counts: dict[str, int] = {}
        for clause in clauses:
            name = _stage_name(clause)
            if name is None or (_SUCCESS_MARKERS.search(clause) and name not in {"release", "place"}):
                continue
            stage_counts[name] = stage_counts.get(name, 0) + 1
            unique_name = name if stage_counts[name] == 1 else f"{name}_{stage_counts[name]}"
            observable = (
                f"视频中可观察到：{clause}"
                if chinese
                else f"Video evidence confirms: {clause}"
            )
            stages.append(
                TaskStage(
                    name=unique_name,
                    description=clause,
                    observable_condition=observable,
                )
            )

        if not stages:
            stages.append(
                TaskStage(
                    name="complete_task",
                    description=description,
                    observable_condition=(
                        f"视频中可观察到任务描述成立：{description}"
                        if chinese
                        else f"Video evidence confirms the task description: {description}"
                    ),
                )
            )

        success = tuple(clause for clause in clauses if _SUCCESS_MARKERS.search(clause))
        if not success:
            success = (stages[-1].observable_condition,)
        failure = tuple(clause for clause in clauses if _FAILURE_MARKERS.search(clause))
        if not failure:
            failure = (
                ("最终成功条件未满足",)
                if chinese
                else ("The final success criteria are not satisfied",)
            )

        resolved_name = task_name.strip() or clauses[0][:80]
        resolved_key = task_key.strip() or _slugify(resolved_name, description)
        aliases = tuple(dict.fromkeys(value for value in (resolved_name, clauses[0]) if value))
        profile = TaskSemanticProfile(
            task_key=resolved_key,
            task_name=resolved_name,
            instruction=description,
            aliases=aliases,
            objects=_extract_objects(description),
            stages=tuple(stages),
            success_criteria=success,
            failure_criteria=failure,
            review_threshold=self.review_threshold,
            pass_threshold=self.pass_threshold,
            notes=(
                "由本地启发式解析器生成的草稿；运行语义质检前请人工确认。"
                if chinese
                else "Draft generated by the local heuristic parser; review before auditing."
            ),
        )
        profile.validate()
        return profile
