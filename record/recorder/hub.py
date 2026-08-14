"""Optional Hugging Face Hub upload for a recorded dataset.

Uses ``huggingface_hub`` directly (no lerobot). Import stays lazy so recording
never requires the hub client unless ``push_to_hub`` is set.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def push_dataset(root: Path, repo_id: str, *, private: bool = False, tags: list[str] | None = None) -> None:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "push_to_hub requires 'huggingface_hub' (pip install huggingface_hub)."
        ) from exc

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(root), repo_id=repo_id, repo_type="dataset")
    logger.info("uploaded %s to https://huggingface.co/datasets/%s", root, repo_id)
