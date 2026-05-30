"""Depth PNG helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .depth_feature_contract import depth_sidecar_relpath


def save_depth_png(path: str | Path, depth_mm: np.ndarray) -> Path:
    """Save a uint16 depth image in millimeters as a lossless PNG."""
    output_path = Path(path)
    if depth_mm.dtype != np.uint16:
        raise ValueError(f"Depth PNGs must be saved as uint16 millimeters, got {depth_mm.dtype}.")
    if depth_mm.ndim == 3 and depth_mm.shape[-1] == 1:
        depth_mm = depth_mm[..., 0]
    if depth_mm.ndim != 2:
        raise ValueError(f"Depth PNGs must be 2D or (H, W, 1), got shape {depth_mm.shape}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(depth_mm).save(output_path)
    return output_path


def load_depth_png(path: str | Path) -> np.ndarray:
    """Load a uint16 depth PNG without changing its units."""
    image = Image.open(path)
    array = np.array(image, dtype=np.uint16)
    if array.ndim != 2:
        raise ValueError(f"Expected a single-channel depth PNG, got shape {array.shape}.")
    return array


def save_depth_sidecar_frame(
    dataset_root: str | Path,
    feature_key: str,
    episode_index: int,
    frame_index: int,
    depth_mm: np.ndarray,
) -> Path:
    """Persist a depth frame to the project-local sidecar PNG directory."""
    output_path = Path(dataset_root) / depth_sidecar_relpath(feature_key, episode_index, frame_index)
    return save_depth_png(output_path, depth_mm)
