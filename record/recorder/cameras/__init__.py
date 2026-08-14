"""Framework-free camera backends (opencv / realsense / orbbec / mock)."""

from __future__ import annotations

from ..config import BaseCameraConfig
from .base import Camera, ColorMode, DeviceNotConnectedError


def make_camera(name: str, config: BaseCameraConfig) -> Camera:
    """Instantiate a single camera backend from its config (lazy SDK imports)."""
    if config.type == "opencv":
        from .opencv_camera import OpenCVCamera

        return OpenCVCamera(config)
    if config.type == "intelrealsense":
        from .realsense_camera import RealSenseCamera

        return RealSenseCamera(config)
    if config.type == "orbbec":
        from .orbbec_camera import OrbbecCamera

        return OrbbecCamera(config)
    if config.type == "mock":
        from .mock_camera import MockCamera

        return MockCamera(config)
    raise ValueError(f"camera '{name}' has unsupported type '{config.type}'")


def make_cameras(configs: dict[str, BaseCameraConfig]) -> dict[str, Camera]:
    return {name: make_camera(name, cfg) for name, cfg in configs.items()}


__all__ = ["Camera", "ColorMode", "DeviceNotConnectedError", "make_camera", "make_cameras"]
