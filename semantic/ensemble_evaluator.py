"""Ensemble semantic evaluator: combine several evaluators behind one interface.

The discrimination study showed no single signal catches every failure — motion
nails frozen and temporally scrambled episodes, alignment is needed for
semantic-but-physically-valid errors. An ensemble runs several child evaluators
and combines their scores so the audit sees the *union* of what they detect,
while still presenting one ``SemanticResult`` to the existing socket.

Two properties make this safe to drop into the conservative audit:

  * **Graceful degradation.** A child that cannot run on an episode (no video for
    an alignment child, no action stream for a motion child) is skipped, its error
    recorded in ``details``, and the remaining children still decide. The audit
    therefore keeps working when only some signals are available.
  * **Conservative combination.** The default ``combine="min"`` takes the most
    pessimistic child score, so any evaluator that sees a problem pulls the episode
    toward review — matching the audit's "when unsure, review" doctrine.

Children are built with the same ``module:attribute`` loader the audit uses, so an
ensemble is configured entirely from JSON and composes any current or future
evaluator without code changes here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from utils.semantic_evaluator import (
    SemanticEpisode,
    SemanticResult,
    evaluator_name,
    load_evaluator,
    normalize_result,
)


COMBINE_MODES = ("min", "mean", "weighted")


@dataclass
class EnsembleChild:
    name: str
    evaluator: Any
    weight: float = 1.0


class EnsembleEvaluator:
    """Run several child evaluators and combine them into one result."""

    def __init__(
        self,
        children: Sequence[EnsembleChild],
        *,
        combine: str = "min",
        curve_from: str | None = None,
        require: int = 1,
    ) -> None:
        if combine not in COMBINE_MODES:
            raise ValueError(f"combine must be one of {COMBINE_MODES}")
        if not children:
            raise ValueError("ensemble needs at least one child evaluator")
        self.children = list(children)
        self.combine = combine
        self.curve_from = curve_from
        self.require = max(1, int(require))
        self.name = "ensemble(" + ",".join(c.name for c in self.children) + ")"

    def _combine_scores(self, scores: list[float], weights: list[float]) -> float:
        """Reduce the children's scores to one. ``min`` is conservative (any child
        that sees a problem wins → maximizes failure recall, inherits every child's
        false positives); ``mean``/``weighted`` average instead."""
        if self.combine == "min":
            return float(min(scores))
        if self.combine == "mean":
            return float(sum(scores) / len(scores))
        total = sum(weights)
        if total <= 0.0:
            return float(sum(scores) / len(scores))
        return float(sum(s * w for s, w in zip(scores, weights)) / total)

    def evaluate(self, episode: SemanticEpisode) -> SemanticResult:
        """Run every child, skip the ones that can't (recording why), combine the
        rest. Raises only if fewer than ``require`` children ran — that graceful
        degradation is the point: the audit keeps working with whatever signals are
        available (motion when there's no video, alignment when there's no motion).
        """
        child_reports: list[dict[str, Any]] = []
        ok: list[tuple[EnsembleChild, SemanticResult]] = []
        for child in self.children:
            try:
                result = normalize_result(child.evaluator.evaluate(episode))
            except Exception as exc:  # a child that cannot run is skipped, not fatal
                child_reports.append(
                    {"name": child.name, "ok": False,
                     "error": f"{type(exc).__name__}:{str(exc)[:200]}"}
                )
                continue
            ok.append((child, result))
            child_reports.append(
                {"name": child.name, "ok": True, "score": result.score,
                 "method": result.details.get("method"), "weight": child.weight}
            )

        if len(ok) < self.require:
            raise RuntimeError(
                f"ensemble requires >= {self.require} usable children, got {len(ok)} "
                f"({'; '.join(r.get('error', '') for r in child_reports if not r['ok'])})"
            )

        scores = [r.score for _, r in ok]
        weights = [c.weight for c, _ in ok]
        combined = self._combine_scores(scores, weights)

        curve = self._select_curve(ok)
        details: dict[str, Any] = {
            "method": "ensemble",
            "combine": self.combine,
            "n_children_ok": len(ok),
            "children": child_reports,
            "combined_score": combined,
        }
        return SemanticResult(
            progress_curve=curve,
            success_probability=combined,
            details=details,
        )

    def _select_curve(self, ok: list[tuple[EnsembleChild, SemanticResult]]) -> tuple[float, ...]:
        """Pass through one child's progress curve for the shape/stage checks.

        Prefer the configured ``curve_from`` child; otherwise the first usable
        child that produced a non-empty curve. Curves from different signals mean
        different things, so they are not averaged — one meaningful curve beats a
        blended one.
        """
        if self.curve_from is not None:
            for child, result in ok:
                if child.name == self.curve_from and result.progress_curve:
                    return result.progress_curve
        for _, result in ok:
            if result.progress_curve:
                return result.progress_curve
        return ()


def create_evaluator(config: Mapping[str, Any] | None = None) -> EnsembleEvaluator:
    """Factory for the audit's ``module:attribute`` loader.

    Config shape::

        {
          "combine": "min",            # min | mean | weighted
          "curve_from": "motion",       # child name to source the progress curve
          "require": 1,                 # minimum children that must run
          "children": [
            {"name": "motion", "spec": "semantic.motion_evaluator:create_evaluator",
             "config": {...}, "weight": 1.0},
            {"name": "clip", "spec": "semantic.clip_alignment_evaluator:create_evaluator",
             "config": {...}, "weight": 1.0}
          ]
        }
    """
    options = dict(config or {})
    raw_children = options.get("children")
    if not isinstance(raw_children, Sequence) or not raw_children:
        raise ValueError("ensemble config must list at least one child in 'children'")

    children: list[EnsembleChild] = []
    for entry in raw_children:
        if not isinstance(entry, Mapping) or "spec" not in entry:
            raise ValueError("each child needs a 'spec' (module:attribute)")
        evaluator = load_evaluator(str(entry["spec"]), entry.get("config"))
        name = str(entry.get("name") or evaluator_name(evaluator))
        children.append(EnsembleChild(name=name, evaluator=evaluator,
                                      weight=float(entry.get("weight", 1.0))))

    return EnsembleEvaluator(
        children,
        combine=str(options.get("combine", "min")),
        curve_from=options.get("curve_from"),
        require=int(options.get("require", 1)),
    )
