"""Tests for lossless depth PNG helpers."""

import numpy as np

from lerobot_policy_rgbd_vla_depth.data.depth_io import (
    load_depth_png,
    save_depth_png,
    save_depth_sidecar_frame,
)


def test_depth_png_roundtrip(tmp_path):
    depth = (np.arange(36, dtype=np.uint16).reshape(6, 6) * 10).astype(np.uint16)
    path = save_depth_png(tmp_path / "depth.png", depth)
    loaded = load_depth_png(path)
    assert loaded.dtype == np.uint16
    assert np.array_equal(depth, loaded)


def test_depth_sidecar_roundtrip(tmp_path):
    depth = np.arange(16, dtype=np.uint16).reshape(4, 4)
    path = save_depth_sidecar_frame(
        dataset_root=tmp_path,
        feature_key="observation.images.front_depth",
        episode_index=2,
        frame_index=7,
        depth_mm=depth,
    )
    assert path.exists()
    assert "depth_sidecar/observation.images.front_depth/episode-000002/frame-000007.png" in str(path)
    loaded = load_depth_png(path)
    assert np.array_equal(depth, loaded)
