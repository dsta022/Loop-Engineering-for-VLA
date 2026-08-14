#!/usr/bin/env python
"""Shared camera-profile resolver for every recording entry point.

Camera hardware facts (backend type, serial number, resolution, depth on/off)
live once in ``camera_profiles.yaml``. A recording run selects a profile with
``--camera_profile=<name>``; this module then loads the record YAML given by
``--config_path``, replaces ``robot.cameras`` with the selected profile, writes
the merged configuration to a temporary YAML file, and rewrites ``sys.argv`` to
point at it. The stock LeRobot/draccus entry points never see the extra flags.

Deployed layout (after copying the overlays into a LeRobot checkout):

    <lerobot_root>/camera_profiles.yaml
    <lerobot_root>/camera_profiles.py

The resolver is dependency-free apart from PyYAML and never imports LeRobot,
so profile selection and validation stay testable outside a robot runtime.
"""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PROFILES_FILENAME = "camera_profiles.yaml"
PROFILE_FLAG = "--camera_profile"
PROFILES_PATH_FLAG = "--camera_profiles_path"
CONFIG_PATH_FLAG = "--config_path"
DEPTH_CAPABLE_TYPES = {"orbbec", "intelrealsense"}


class CameraProfileError(ValueError):
    """Raised when the camera profile file or selection is invalid."""


def _extract_flag(argv: list[str], flag: str) -> tuple[list[str], str | None]:
    """Remove ``flag=value`` or ``flag value`` from argv; return remaining argv and the value."""
    remaining: list[str] = []
    value: str | None = None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == flag:
            if index + 1 >= len(argv):
                raise CameraProfileError(f"{flag} requires a value")
            value = argv[index + 1]
            index += 2
            continue
        if arg.startswith(f"{flag}="):
            value = arg.split("=", 1)[1]
            index += 1
            continue
        remaining.append(arg)
        index += 1
    if value is not None and not value.strip():
        raise CameraProfileError(f"{flag} requires a non-empty value")
    return remaining, value


def _find_flag_value(argv: list[str], flag: str) -> tuple[str | None, int | None, bool]:
    """Locate ``flag`` in argv; return (value, index of the argument holding the value, inline?)."""
    for index, arg in enumerate(argv):
        if arg == flag:
            if index + 1 >= len(argv):
                raise CameraProfileError(f"{flag} requires a value")
            return argv[index + 1], index + 1, False
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1], index, True
    return None, None, False


def load_camera_profiles(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise CameraProfileError(
            f"camera profiles file not found: {path}. Copy record/camera_profiles.yaml "
            "to the LeRobot repository root or pass --camera_profiles_path explicitly."
        )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("profiles"), dict):
        raise CameraProfileError(f"{path} must contain a top-level 'profiles' mapping")
    if not payload["profiles"]:
        raise CameraProfileError(f"{path} defines no profiles")
    for name, cameras in payload["profiles"].items():
        if not isinstance(cameras, dict) or not cameras:
            raise CameraProfileError(f"profile '{name}' must be a non-empty mapping of camera names")
        for camera_name, camera in cameras.items():
            if not isinstance(camera, dict) or not str(camera.get("type", "")).strip():
                raise CameraProfileError(
                    f"profile '{name}' camera '{camera_name}' must be a mapping with a 'type' field"
                )
    default_profile = payload.get("default_profile")
    if default_profile is not None and default_profile not in payload["profiles"]:
        raise CameraProfileError(
            f"default_profile '{default_profile}' does not exist in {sorted(payload['profiles'])}"
        )
    return payload


def select_profile(payload: dict[str, Any], requested: str) -> tuple[str, dict[str, Any]]:
    name = requested
    if requested == "default":
        default_profile = payload.get("default_profile")
        if not default_profile:
            raise CameraProfileError("--camera_profile=default requires 'default_profile' in the profiles file")
        name = str(default_profile)
    profiles = payload["profiles"]
    if name not in profiles:
        raise CameraProfileError(f"unknown camera profile '{name}'; available: {sorted(profiles)}")
    return name, profiles[name]


def validate_profile_cameras(profile_name: str, cameras: dict[str, Any], allow_depth: bool) -> None:
    depth_cameras = []
    for camera_name, camera in cameras.items():
        camera_type = str(camera.get("type", "")).strip()
        use_depth = bool(camera.get("use_depth", False))
        if use_depth and camera_type not in DEPTH_CAPABLE_TYPES:
            raise CameraProfileError(
                f"profile '{profile_name}' camera '{camera_name}' sets use_depth=true but type "
                f"'{camera_type}' has no depth stream (depth-capable: {sorted(DEPTH_CAPABLE_TYPES)})"
            )
        if use_depth:
            depth_cameras.append(camera_name)
    if depth_cameras and not allow_depth:
        raise CameraProfileError(
            f"profile '{profile_name}' enables depth on {depth_cameras}, but this recorder is RGB-only. "
            "Use the rgbd_record recording path for depth capture, or select an RGB-only profile."
        )


def describe_cameras(cameras: dict[str, Any]) -> str:
    parts = []
    for camera_name, camera in cameras.items():
        detail = str(camera.get("type", "?"))
        width, height = camera.get("width"), camera.get("height")
        if width and height:
            detail += f" {width}x{height}"
        if camera.get("use_depth"):
            detail += "+depth"
        parts.append(f"{camera_name}={detail}")
    return ", ".join(parts)


def merge_cameras_into_record_config(config: Any, cameras: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict) or not isinstance(config.get("robot"), dict):
        raise CameraProfileError("record config must contain a 'robot' mapping to receive the camera profile")
    merged = copy.deepcopy(config)
    merged["robot"]["cameras"] = copy.deepcopy(cameras)
    return merged


def rewrite_argv_with_camera_profile(
    argv: list[str],
    *,
    allow_depth: bool,
    search_root: Path | None = None,
) -> list[str]:
    """Apply ``--camera_profile`` to argv; return argv unchanged when no profile is requested."""
    argv, requested = _extract_flag(argv, PROFILE_FLAG)
    argv, profiles_path_value = _extract_flag(argv, PROFILES_PATH_FLAG)
    if requested is None:
        if profiles_path_value is not None:
            raise CameraProfileError(f"{PROFILES_PATH_FLAG} requires {PROFILE_FLAG}")
        return argv

    profiles_path = (
        Path(profiles_path_value)
        if profiles_path_value is not None
        else Path(search_root or os.getcwd()) / DEFAULT_PROFILES_FILENAME
    )
    payload = load_camera_profiles(profiles_path)
    profile_name, cameras = select_profile(payload, requested)
    validate_profile_cameras(profile_name, cameras, allow_depth)

    config_value, config_index, inline = _find_flag_value(argv, CONFIG_PATH_FLAG)
    if config_value is None or config_index is None:
        raise CameraProfileError(f"{PROFILE_FLAG} requires {CONFIG_PATH_FLAG} to locate the record config")
    config_path = Path(config_value)
    if not config_path.exists():
        raise CameraProfileError(f"record config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    merged = merge_cameras_into_record_config(config, cameras)

    fd, merged_path = tempfile.mkstemp(prefix=f"record_{profile_name}_", suffix=".yaml")
    os.close(fd)
    Path(merged_path).write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")

    rewritten = list(argv)
    rewritten[config_index] = f"{CONFIG_PATH_FLAG}={merged_path}" if inline else merged_path
    print(
        f"[camera_profiles] profile '{profile_name}' from {profiles_path}: {describe_cameras(cameras)}"
    )
    print(f"[camera_profiles] merged record config: {merged_path}")
    return rewritten
