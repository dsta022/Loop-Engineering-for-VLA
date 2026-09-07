"""Optional live camera preview for recording sessions.

``display_data: true`` in a record config opens one OpenCV window per camera
and refreshes it every control step. Depth streams are colorized and shown
next to their RGB frame. The preview degrades to a no-op, with one warning,
when OpenCV is missing or was built without GUI support
(``opencv-python-headless``); recording itself is never affected.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np

logger = logging.getLogger(__name__)

DEPTH_PREVIEW_MIN_MM = 300
DEPTH_PREVIEW_MAX_MM = 4000


def depth_to_gray(
    depth: np.ndarray, vmin: int = DEPTH_PREVIEW_MIN_MM, vmax: int = DEPTH_PREVIEW_MAX_MM
) -> np.ndarray:
    """Map uint16 millimetre depth to uint8; invalid (zero) pixels become 0."""
    arr = np.asarray(depth)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    span = max(int(vmax) - int(vmin), 1)
    scaled = (np.clip(arr.astype(np.float32), vmin, vmax) - vmin) / span * 255.0
    gray = scaled.astype(np.uint8)
    gray[arr == 0] = 0
    return gray


class FramePreview:
    """Show the latest observation of every camera in OpenCV windows."""

    def __init__(self, camera_names: list[str], *, enabled: bool, title: str = "recorder") -> None:
        self.camera_names = list(camera_names)
        self.title = title
        self.enabled = bool(enabled) and bool(self.camera_names)
        self._cv2: Any = None
        if self.enabled:
            try:
                import cv2

                self._cv2 = cv2
            except ImportError:
                logger.warning("display_data requested but OpenCV is not installed; preview disabled")
                self.enabled = False

    def show(self, observation: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        cv2 = self._cv2
        try:
            for name in self.camera_names:
                frame = observation.get(name)
                if frame is None:
                    continue
                panel = cv2.cvtColor(np.ascontiguousarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)
                depth = observation.get(f"{name}_depth")
                if depth is not None:
                    colored = cv2.applyColorMap(depth_to_gray(depth), cv2.COLORMAP_TURBO)
                    if colored.shape[:2] != panel.shape[:2]:
                        colored = cv2.resize(colored, (panel.shape[1], panel.shape[0]))
                    panel = np.concatenate([panel, colored], axis=1)
                cv2.imshow(f"{self.title}: {name}", panel)
            cv2.waitKey(1)
        except Exception as exc:  # noqa: BLE001 - headless OpenCV raises cv2.error here
            logger.warning("live preview disabled: %s", str(exc).splitlines()[0][:200])
            self.enabled = False
            self.close()

    def close(self) -> None:
        if self._cv2 is None:
            return
        try:
            self._cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass
