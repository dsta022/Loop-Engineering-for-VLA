#!/usr/bin/env python3
"""Push a local real LeRobot dataset folder to Hugging Face Hub.

Token resolution order:
  1. the HF_TOKEN environment variable
     (PowerShell: $env:HF_TOKEN="hf_..."; CMD: set HF_TOKEN=hf_...)
  2. the token cached by `huggingface-cli login`

Tokens are never read from this source file, so nothing secret can be
committed by accident.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sys

# Allow `python ./pipeline/hf_push.py` to run as a plain script: the repo-root
# packages must be importable without installing the project first.
if __package__ in (None, ""):
    _REPO_ROOT = str(Path(__file__).resolve().parents[1])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from utils.hf_upload_utils import (
    count_files_and_bytes,
    human_size,
    import_hf_api,
    require_supported_call,
    upload_with_folder,
    upload_with_large_folder,
    validate_lerobot_dataset_for_upload,
)


# ---------------------------------------------------------------------------
# Defaults. The local dataset contents are uploaded to the repo root:
#
# determ_lerobot_rgb_depth_vla/
#   data/
#   depth_sidecar/
#   meta/
#   videos/
#
# becomes:
#
# your_huggingface_username/your_dataset_name
#   data/
#   depth_sidecar/
#   meta/
#   videos/
# ---------------------------------------------------------------------------
HF_USERNAME = "dadm022"
HF_DATASET_NAME = "determ_lerobot_rgb_depth_vla"

DEFAULT_LOCAL_DATASET_DIR = Path("determ_lerobot_rgb_depth_vla")
DEFAULT_PRIVATE_REPO = False
DEFAULT_REVISION = "main"
DEFAULT_NUM_WORKERS = 8


@dataclass(frozen=True)
class UploadConfig:
    dataset_dir: Path
    repo_id: str
    token: str
    private: bool
    revision: str
    method: str
    num_workers: int
    commit_message: str
    skip_local_check: bool
    strict_no_reparse: bool
    dry_run: bool


def cached_hub_token() -> str:
    """Return the token stored by `huggingface-cli login`, or an empty string."""
    try:
        from huggingface_hub import get_token
    except ImportError:
        return ""
    try:
        return (get_token() or "").strip()
    except Exception:  # noqa: BLE001 - any cache problem simply means "no token"
        return ""


def resolve_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip() or cached_hub_token()
    placeholder_tokens = {"", "hf_your_write_token", "your_token_here"}
    if not token or token in placeholder_tokens:
        raise SystemExit(
            "No Hugging Face token found. Set HF_TOKEN in the shell or run `huggingface-cli login`."
        )
    if not token.startswith("hf_"):
        raise SystemExit("HF token does not look valid. A user access token normally starts with 'hf_'.")
    return token


def resolve_repo_id(repo_id: str | None, dataset_name: str) -> str:
    if repo_id:
        return repo_id
    username = HF_USERNAME.strip()
    if not username or username == "your_huggingface_username":
        raise SystemExit("Fill HF_USERNAME at the top of this script or pass --repo-id USERNAME/DATASET_NAME.")
    if not dataset_name or dataset_name == "your_dataset_name":
        raise SystemExit("Fill HF_DATASET_NAME at the top of this script or pass --dataset-name DATASET_NAME.")
    return f"{username}/{dataset_name}"


def print_upload_preview(config: UploadConfig, info: dict[str, Any]) -> None:
    file_count, total_bytes = count_files_and_bytes(config.dataset_dir)
    print(f"Dataset dir: {config.dataset_dir}")
    print(f"Repo id: {config.repo_id}")
    print("Repo type: dataset")
    print(f"Private: {config.private}")
    print(f"Revision: {config.revision}")
    print(f"Files: {file_count}")
    print(f"Size: {human_size(total_bytes)}")
    if info:
        print(
            "LeRobot metadata: "
            f"episodes={info.get('total_episodes')} "
            f"frames={info.get('total_frames')} "
            f"tasks={info.get('total_tasks')}"
        )


def create_dataset_repo(config: UploadConfig) -> Any:
    HfApi = import_hf_api()
    api = HfApi(token=config.token)
    create_kwargs = {
        "repo_id": config.repo_id,
        "repo_type": "dataset",
        "private": config.private,
        "exist_ok": True,
        "token": config.token,
    }
    api.create_repo(**require_supported_call(api.create_repo, create_kwargs))
    return api


def upload_dataset_contents(config: UploadConfig, api: Any) -> None:
    if config.method in {"auto", "large-folder"} and hasattr(api, "upload_large_folder"):
        print("Uploading with HfApi.upload_large_folder ...")
        upload_with_large_folder(
            api=api,
            repo_id=config.repo_id,
            dataset_dir=config.dataset_dir,
            token=config.token,
            revision=config.revision,
            private=config.private,
            num_workers=config.num_workers,
        )
    elif config.method == "large-folder":
        raise RuntimeError(
            "Installed huggingface_hub does not provide upload_large_folder. "
            "Upgrade huggingface_hub or use --method folder."
        )
    else:
        print("Uploading with HfApi.upload_folder ...")
        upload_with_folder(
            api=api,
            repo_id=config.repo_id,
            dataset_dir=config.dataset_dir,
            token=config.token,
            revision=config.revision,
            commit_message=config.commit_message,
        )


def run_upload(config: UploadConfig) -> None:
    info = {}
    if not config.skip_local_check:
        info = validate_lerobot_dataset_for_upload(config.dataset_dir, strict_no_reparse=config.strict_no_reparse)
    print_upload_preview(config, info)
    if config.dry_run:
        print("Dry run only. Nothing was uploaded.")
        return

    api = create_dataset_repo(config)
    upload_dataset_contents(config, api)
    print("=" * 80)
    print("UPLOAD COMPLETE")
    print(f"https://huggingface.co/datasets/{config.repo_id}")


def push_dataset(
    dataset_dir: Path,
    repo_id: str,
    token: str,
    private: bool,
    revision: str,
    method: str,
    num_workers: int,
    commit_message: str,
    skip_local_check: bool,
    strict_no_reparse: bool,
    dry_run: bool,
) -> None:
    """Push a dataset; kept for compatibility with existing usage."""
    run_upload(
        UploadConfig(
            dataset_dir=dataset_dir.resolve(),
            repo_id=repo_id,
            token=token,
            private=private,
            revision=revision,
            method=method,
            num_workers=num_workers,
            commit_message=commit_message,
            skip_local_check=skip_local_check,
            strict_no_reparse=strict_no_reparse,
            dry_run=dry_run,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Push a local LeRobot dataset folder to Hugging Face Hub.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_LOCAL_DATASET_DIR)
    parser.add_argument("--dataset-name", default=HF_DATASET_NAME)
    parser.add_argument("--repo-id", default=None, help="Full repo id, e.g. username/dataset_name")
    parser.add_argument("--private", action="store_true", default=DEFAULT_PRIVATE_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--method", choices=["auto", "large-folder", "folder"], default="auto")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--commit-message", default="Upload LeRobot RGB-D VLA dataset")
    parser.add_argument("--skip-local-check", action="store_true")
    parser.add_argument("--allow-reparse", action="store_true", help="Allow symlink/junction/reparse paths")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = "hf_dry_run_token" if args.dry_run else resolve_token()
    repo_id = resolve_repo_id(args.repo_id, args.dataset_name)
    push_dataset(
        dataset_dir=args.dataset_dir,
        repo_id=repo_id,
        token=token,
        private=args.private,
        revision=args.revision,
        method=args.method,
        num_workers=args.num_workers,
        commit_message=args.commit_message,
        skip_local_check=args.skip_local_check,
        strict_no_reparse=not args.allow_reparse,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
