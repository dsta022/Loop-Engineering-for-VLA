"""Intel RealSense RGB-D backend (D405 / D435 / D435i) via pyrealsense2.

Depth shares the color stream resolution/fps and is aligned to color, then
converted to uint16 millimeters to match the dataset depth contract.
"""

from __future__ import annotations

import numpy as np

from ..config import RealSenseCameraConfig
from .base import Camera, ColorMode, DeviceNotConnectedError, check_connected, get_cv2_rotation


class RealSenseCamera(Camera):
    def __init__(self, config: RealSenseCameraConfig):
        super().__init__(config)
        self.serial_number_or_name = config.serial_number_or_name
        self._rotation = get_cv2_rotation(config.rotation)
        self._pipeline = None
        self._align = None
        self._depth_scale_mm = 1.0

    def __str__(self) -> str:
        return f"RealSenseCamera({self.serial_number_or_name})"

    @property
    def is_connected(self) -> bool:
        return self._pipeline is not None

    def connect(self) -> None:
        import pyrealsense2 as rs

        if self.is_connected:
            return
        pipeline = rs.pipeline()
        config = rs.config()
        serial = str(self.serial_number_or_name)
        if serial and not serial.isdigit():
            # Treat non-numeric names as serials too; RealSense matches by serial string.
            config.enable_device(serial)
        elif serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, int(self.width), int(self.height), rs.format.bgr8, int(self.fps))
        if self.use_depth:
            config.enable_stream(
                rs.stream.depth, int(self.width), int(self.height), rs.format.z16, int(self.fps)
            )
        try:
            profile = pipeline.start(config)
        except Exception as exc:  # noqa: BLE001
            raise DeviceNotConnectedError(f"{self} failed to start: {exc}") from exc

        if self.use_depth:
            self._align = rs.align(rs.stream.color)
            depth_sensor = profile.get_device().first_depth_sensor()
            # meters-per-unit -> millimeters-per-unit
            self._depth_scale_mm = float(depth_sensor.get_depth_scale()) * 1000.0
        self._pipeline = pipeline

    @check_connected
    def read(self) -> tuple[np.ndarray, np.ndarray | None]:
        import cv2

        frames = self._pipeline.wait_for_frames()
        if self.use_depth and self._align is not None:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError(f"{self} received no color frame.")
        color_bgr = np.asanyarray(color_frame.get_data())
        color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB) if self.color_mode == ColorMode.RGB else color_bgr

        depth = None
        if self.use_depth:
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                raise RuntimeError(f"{self} received no depth frame.")
            raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            depth = np.clip(raw * self._depth_scale_mm, 0, 65535).astype(np.uint16)

        if self._rotation is not None:
            color = cv2.rotate(color, self._rotation)
            if depth is not None:
                depth = cv2.rotate(depth, self._rotation)

        return self._validate_color(color), self._as_depth_mm(depth)

    def disconnect(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = None
                self._align = None
