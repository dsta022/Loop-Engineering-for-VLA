"""Deployer plugin that publishes a checkpoint into a target directory.

The checkpoint is copied to ``<target_root>/<target>/<iteration_id>`` and
``<target_root>/<target>/current.json`` is rewritten atomically to point at it,
so a robot host that watches that file always sees a complete checkpoint.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from policy_improvement.plugins import DeploymentRequest


class FilesystemDeployer:
    name = "filesystem_deployer"

    def __init__(self, *, target_root: str, overwrite: bool = False) -> None:
        if not str(target_root).strip():
            raise ValueError("filesystem_deployer requires target_root")
        self.target_root = Path(target_root).expanduser()
        self.overwrite = bool(overwrite)

    def deploy(self, request: DeploymentRequest) -> dict[str, Any]:
        source = Path(request.checkpoint).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"checkpoint not found: {source}")
        target = request.target.strip()
        if not target or any(token in target for token in ("/", "\\", "..")):
            raise ValueError("deployment target must be a simple directory name")
        target_dir = self.target_root / target
        destination = target_dir / request.iteration_id
        if destination.exists():
            if not self.overwrite:
                raise FileExistsError(f"deployment already exists: {destination}")
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        target_dir.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        pointer = target_dir / "current.json"
        payload = {
            "iteration_id": request.iteration_id,
            "checkpoint": str(destination.resolve()),
            "deployed_at": datetime.now(timezone.utc).isoformat(),
        }
        temporary = pointer.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(pointer)
        return {
            "deployment_id": f"{target}/{request.iteration_id}",
            "target": target,
            "details": {"checkpoint": str(destination.resolve()), "pointer": str(pointer.resolve())},
        }


def create_deployer(config: Mapping[str, Any] | None = None) -> FilesystemDeployer:
    return FilesystemDeployer(**dict(config or {}))
