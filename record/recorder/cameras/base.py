"""Shared camera primitives (no lerobot dependency).

All backends expose one uniform contract::

    cam.connect()
    color, depth = cam.read()   # color: (H, W, 3) uint8 RGB; depth: (H, W) uint16 mm or None
    cam.disconnect()

This replaces lerobot's ``Camera`` base plus its ``read_latest``/``read_depth``
split; the recorder only needs an atomic latest (color, depth) pair per frame.
"""

from __future__ import annotations

import enum
import functools
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..config import BaseCameraConfig


class ColorMode(str, enum.Enum):
    RGB = "rgb"
    BGR = "bgr"


class DeviceNotConnectedError(RuntimeError):
    """Raised when a camera operation is attempted before ``connect()``."""


def check_connected(method):
    @functools.wraps(method)
    def wrapper(self: "Camera", *args, **kwargs):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected; call connect() first.")
        return method(self, *args, **kwargs)

    return wrapper


def get_cv2_rotation(rotation: int) -> int | None:
    """Map a degrees value (0/90/180/270) to a cv2 rotation flag, or None."""
    import cv2

    return {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }.get(int(rotation) % 360 if rotation else 0)


class Camera(ABC):
    """Base class for RGB(-D) camera backends."""

    def __init__(self, config: BaseCameraConfig):
        self.config = config
        self.width: int | None = config.width
        self.height: int | None = config.height
        self.fps: int | None = config.fps
        self.use_depth: bool = bool(getattr(config, "use_depth", False))
        self.color_mode = ColorMode(getattr(config, "color_mode", "rgb"))

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.config.type})"

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def read(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Return the latest ``(rgb, depth)`` frame.

        ``rgb`` is ``(H, W, 3)`` uint8; ``depth`` is ``(H, W)`` uint16 millimeters
        when ``use_depth`` else ``None``.
        """

    @abstractmethod
    def disconnect(self) -> None: ...

    # -- shared helpers ---------------------------------------------------- #
    def _validate_color(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"{self} expected (H, W, 3) color frame, got {image.shape}")
        if self.height is not None and self.width is not None:
            if image.shape[0] != self.height or image.shape[1] != self.width:
                raise RuntimeError(
                    f"{self} color frame {image.shape[1]}x{image.shape[0]} "
                    f"!= configured {self.width}x{self.height}"
                )
        return image.astype(np.uint8, copy=False)

    @staticmethod
    def _as_depth_mm(depth: Any) -> np.ndarray | None:
        if depth is None:
            return None
        arr = np.asarray(depth)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr.astype(np.uint16, copy=False)
