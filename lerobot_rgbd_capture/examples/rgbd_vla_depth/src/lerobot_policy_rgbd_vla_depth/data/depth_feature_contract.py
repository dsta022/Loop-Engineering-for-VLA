"""Project-local contract for RGB-D feature naming and dataset storage."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any

from lerobot.datasets.utils import DEFAULT_IMAGE_PATH

DEPTH_SIDECAR_DIR = "depth_sidecar"
LEGACY_DEPTH_KEY_PATTERN = re.compile(r"^observation\.images\.(?P<camera>.+)_depth$")
CANONICAL_DEPTH_KEY_PATTERN = re.compile(r"^observation\.depth\.(?P<camera>.+)$")


def dataset_depth_key(camera_name: str) -> str:
    return f"observation.images.{camera_name}_depth"


def canonical_depth_key(camera_name: str) -> str:
    return f"observation.depth.{camera_name}"


def depth_camera_name_from_key(key: str) -> str | None:
    legacy = LEGACY_DEPTH_KEY_PATTERN.match(key)
    if legacy:
        return legacy.group("camera")
    canonical = CANONICAL_DEPTH_KEY_PATTERN.match(key)
    if canonical:
        return canonical.group("camera")
    return None


def is_depth_feature_key(key: str) -> bool:
    return depth_camera_name_from_key(key) is not None


def iter_depth_feature_keys(features: dict[str, Any]) -> list[str]:
    return sorted(key for key in features if is_depth_feature_key(key))


def canonicalize_depth_item(item: dict[str, Any]) -> dict[str, Any]:
    canonical_item = dict(item)
    for key in list(item):
        camera_name = depth_camera_name_from_key(key)
        if camera_name is None:
            continue
        canonical_item.setdefault(canonical_depth_key(camera_name), item[key])
    return canonical_item


def force_depth_features_to_images(features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Force depth streams to use PNG-backed image storage instead of video."""
    coerced_features = deepcopy(features)
    for key, spec in coerced_features.items():
        if is_depth_feature_key(key):
            spec["dtype"] = "image"
    return coerced_features


def depth_sidecar_relpath(feature_key: str, episode_index: int, frame_index: int) -> Path:
    """Return the permanent sidecar PNG path for a depth frame."""
    return (
        Path(DEPTH_SIDECAR_DIR)
        / feature_key
        / f"episode-{episode_index:06d}"
        / f"frame-{frame_index:06d}.png"
    )


def build_depth_recording_manifest(features: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Describe where raw depth streams are stored inside the recorded dataset."""
    depth_streams = []
    for key in iter_depth_feature_keys(features):
        camera_name = depth_camera_name_from_key(key)
        if camera_name is None:
            continue
        spec = features[key]
        depth_streams.append(
            {
                "camera": camera_name,
                "dataset_key": key,
                "canonical_key": canonical_depth_key(camera_name),
                "storage_dtype": spec.get("dtype"),
                "shape": list(spec.get("shape", ())),
                "unit": "millimeter",
                "path_template": DEFAULT_IMAGE_PATH.replace("{image_key}", key),
                "sidecar_path_template": str(depth_sidecar_relpath(key, 0, 0)).replace(
                    "episode-000000/frame-000000.png",
                    "episode-{episode_index:06d}/frame-{frame_index:06d}.png",
                ),
            }
        )

    return {
        "version": 1,
        "format": "rgbd_vla_depth_recording",
        "storage": "lerobot_dataset_mixed_visual_with_depth_sidecar",
        "depth_streams": depth_streams,
    }
