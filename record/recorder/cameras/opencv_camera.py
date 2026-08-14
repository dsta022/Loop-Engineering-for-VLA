"""Plain UVC / OpenCV RGB camera backend (no depth)."""

from __future__ import annotations

import numpy as np

from ..config import OpenCVCameraConfig
from .base import Camera, ColorMode, DeviceNotConnectedError, check_connected, get_cv2_rotation


def _coerce_index(value):
    if isinstance(value, int):
        return value
    text = str(value)
    if text.isdigit():
        return int(text)
    return text


class OpenCVCamera(Camera):
    def __init__(self, config: OpenCVCameraConfig):
        super().__init__(config)
        self.index_or_path = _coerce_index(config.index_or_path)
        self._rotation = get_cv2_rotation(config.rotation)
        self._cap = None

    def __str__(self) -> str:
        return f"OpenCVCamera({self.index_or_path})"

    @property
    def is_connected(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def connect(self) -> None:
        import cv2

        if self.is_connected:
            return
        cap = cv2.VideoCapture(self.index_or_path)
        if not cap.isOpened():
            cap.release()
            raise DeviceNotConnectedError(
                f"{self} could not be opened. Check the camera index/path and that it is free."
            )
        if self.width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        if self.height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
        if self.fps is not None:
            cap.set(cv2.CAP_PROP_FPS, float(self.fps))
        self._cap = cap

    @check_connected
    def read(self) -> tuple[np.ndarray, None]:
        import cv2

        ok, frame_bgr = self._cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"{self} failed to read a frame.")
        if self.color_mode == ColorMode.RGB:
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        else:
            frame = frame_bgr
        if self._rotation is not None:
            frame = cv2.rotate(frame, self._rotation)
        return self._validate_color(frame), None

    def disconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
