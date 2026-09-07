"""Trainer plugin that runs a configurable command line.

Placeholders in ``command``, ``checkpoint``, and ``metrics_file`` are filled
from the training request: ``{iteration_id}``, ``{base_policy}``,
``{training_dataset}``, ``{output_dir}``, plus every key of the request's
``config`` mapping (for example ``{steps}``). The command must leave a
checkpoint at the formatted ``checkpoint`` path; an optional JSON
``metrics_file`` with numeric values becomes the training metrics.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from policy_improvement.plugins import TrainingRequest


def request_fields(request: TrainingRequest) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "iteration_id": request.iteration_id,
        "base_policy": request.base_policy,
        "training_dataset": str(request.training_dataset),
        "output_dir": str(request.output_dir),
    }
    for key, value in dict(request.config or {}).items():
        fields.setdefault(str(key), value)
    return fields


class CommandTrainer:
    name = "command_trainer"

    def __init__(
        self,
        *,
        command: Sequence[str] | str,
        checkpoint: str,
        cwd: str | None = None,
        metrics_file: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if isinstance(command, str):
            parts = shlex.split(command, posix=os.name != "nt")
        else:
            parts = [str(part) for part in command]
        if not parts:
            raise ValueError("command_trainer requires a non-empty command")
        if not str(checkpoint).strip():
            raise ValueError("command_trainer requires a checkpoint path template")
        self.command = parts
        self.checkpoint = str(checkpoint)
        self.cwd = cwd
        self.metrics_file = metrics_file
        self.env = dict(env or {})

    def train(self, request: TrainingRequest) -> dict[str, Any]:
        fields = request_fields(request)
        argv = [part.format(**fields) for part in self.command]
        completed = subprocess.run(
            argv,
            cwd=self.cwd or None,
            env={**os.environ, **self.env} if self.env else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        tail = completed.stdout[-2000:] if completed.stdout else ""
        if completed.returncode != 0:
            raise RuntimeError(
                f"training command failed with exit code {completed.returncode}: {argv}\n{tail}"
            )
        checkpoint_path = Path(self.checkpoint.format(**fields)).expanduser()
        if not checkpoint_path.exists():
            raise RuntimeError(
                f"training command finished but produced no checkpoint at {checkpoint_path}"
            )
        metrics: dict[str, float] = {}
        if self.metrics_file:
            metrics_path = Path(self.metrics_file.format(**fields)).expanduser()
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("metrics_file must contain one JSON object")
            metrics = {str(key): float(value) for key, value in payload.items()}
        return {
            "checkpoint": str(checkpoint_path.resolve()),
            "metrics": metrics,
            "details": {"command": argv, "returncode": completed.returncode, "log_tail": tail},
        }


def create_trainer(config: Mapping[str, Any] | None = None) -> CommandTrainer:
    return CommandTrainer(**dict(config or {}))
