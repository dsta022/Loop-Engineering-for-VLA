"""Depth feature naming + uint16 PNG sidecar storage (lerobot-free).

Ported from the previous overlay's ``depth_feature_contract.py`` +
``depth_io.py``; the single lerobot import (``DEFAULT_IMAGE_PATH``) is inlined
so the manifest keeps the same shape the downstream audit/merge already expect.
"""

from __future__ import annotations

import binascii
import re
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

# lerobot's default image path template, inlined (used only in the manifest doc).
DEFAULT_IMAGE_PATH = "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png"

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


def depth_sidecar_relpath(feature_key: str, episode_index: int, frame_index: int) -> Path:
    return (
        Path(DEPTH_SIDECAR_DIR)
        / feature_key
        / f"episode-{episode_index:06d}"
        / f"frame-{frame_index:06d}.png"
    )


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = binascii.crc32(tag)
    crc = binascii.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def encode_depth_png(depth_mm: np.ndarray) -> bytes:
    """Return lossless 16-bit grayscale PNG bytes for a depth frame in millimeters.

    Pure numpy + zlib (no Pillow). Samples are big-endian uint16, matching
    ``utils.media_utils.read_png_array`` which reads 16-bit PNGs as ``>u2``.
    """
    arr = np.asarray(depth_mm)
    if arr.dtype != np.uint16:
        raise ValueError(f"depth PNGs must be uint16 millimeters, got {arr.dtype}")
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"depth PNGs must be 2D or (H, W, 1), got shape {arr.shape}")
    height, width = arr.shape
    big_endian = np.ascontiguousarray(arr.astype(">u2"))
    rows = big_endian.view(np.uint8).reshape(height, width * 2)
    # Prepend a filter-type byte (0 = None) to each scanline.
    raw = np.concatenate([np.zeros((height, 1), dtype=np.uint8), rows], axis=1)
    ihdr = struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)
    idat = zlib.compress(raw.tobytes(), 6)
    return _PNG_SIGNATURE + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


def save_depth_sidecar_frame(
    dataset_root: str | Path,
    feature_key: str,
    episode_index: int,
    frame_index: int,
    depth_mm: np.ndarray,
) -> Path:
    """Persist a depth frame to the permanent sidecar PNG directory."""
    output_path = Path(dataset_root) / depth_sidecar_relpath(feature_key, episode_index, frame_index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encode_depth_png(depth_mm))
    return output_path


def build_depth_recording_manifest(features: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Describe where raw depth streams live inside the recorded dataset.

    Matches the previous overlay's manifest shape so the toolkit's audit/merge
    treat these datasets identically.
    """
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
