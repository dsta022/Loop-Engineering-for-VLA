"""Tests for the project-local depth feature contract."""

from lerobot_policy_rgbd_vla_depth.data.depth_feature_contract import (
    build_depth_recording_manifest,
    canonical_depth_key,
    canonicalize_depth_item,
    dataset_depth_key,
    depth_sidecar_relpath,
    depth_camera_name_from_key,
    force_depth_features_to_images,
    is_depth_feature_key,
)


def test_depth_key_helpers():
    legacy_key = dataset_depth_key("front")
    canonical_key = canonical_depth_key("front")
    assert legacy_key == "observation.images.front_depth"
    assert canonical_key == "observation.depth.front"
    assert depth_camera_name_from_key(legacy_key) == "front"
    assert depth_camera_name_from_key(canonical_key) == "front"
    assert is_depth_feature_key(legacy_key)


def test_canonicalize_depth_item_adds_alias():
    item = {"observation.images.front_depth": "depth_tensor"}
    aliased = canonicalize_depth_item(item)
    assert aliased["observation.depth.front"] == "depth_tensor"


def test_force_depth_features_to_images_and_manifest():
    features = {
        "observation.images.front": {"dtype": "video", "shape": (480, 640, 3)},
        "observation.images.front_depth": {"dtype": "video", "shape": (480, 640, 1)},
    }
    coerced = force_depth_features_to_images(features)
    assert coerced["observation.images.front"]["dtype"] == "video"
    assert coerced["observation.images.front_depth"]["dtype"] == "image"

    manifest = build_depth_recording_manifest(coerced)
    assert len(manifest["depth_streams"]) == 1
    assert manifest["depth_streams"][0]["camera"] == "front"
    assert manifest["depth_streams"][0]["dataset_key"] == "observation.images.front_depth"
    assert "images/observation.images.front_depth/" in manifest["depth_streams"][0]["path_template"]
    assert manifest["depth_streams"][0]["sidecar_path_template"].startswith(
        "depth_sidecar/observation.images.front_depth/"
    )


def test_depth_sidecar_relpath():
    path = depth_sidecar_relpath("observation.images.front_depth", episode_index=3, frame_index=12)
    assert str(path) == "depth_sidecar/observation.images.front_depth/episode-000003/frame-000012.png"
