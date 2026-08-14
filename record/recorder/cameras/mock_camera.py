"""Synthetic camera for offline testing (no hardware, no SDK).

Produces deterministic but non-constant, non-frozen frames so the toolkit's
quality audit (which flags black/white/constant/frozen video and all-invalid
depth) treats mock datasets as valid content.
"""

from __future__ import annotations

import numpy as np

from ..config import MockCameraConfig
from .base import Camera


class MockCamera(Camera):
    def __init__(self, config: MockCameraConfig):
        super().__init__(config)
        self._connected = False
        self._frame = 0
        self._rng = np.random.default_rng(config.seed)
        h, w = int(config.height), int(config.width)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        self._gradient = (xx / max(w - 1, 1) * 180.0 + yy / max(h - 1, 1) * 60.0)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def read(self) -> tuple[np.ndarray, np.ndarray | None]:
        h, w = int(self.height), int(self.width)
        phase = self._frame
        base = (self._gradient + phase * 3.0) % 256.0
        rgb = np.empty((h, w, 3), dtype=np.uint8)
        rgb[..., 0] = base.astype(np.uint8)
        rgb[..., 1] = ((base + 85.0) % 256.0).astype(np.uint8)
        rgb[..., 2] = ((base + 170.0) % 256.0).astype(np.uint8)
        # Moving bright box guarantees per-frame motion so freeze checks pass.
        box = 24
        cy = (phase * 5) % max(h - box, 1)
        cx = (phase * 7) % max(w - box, 1)
        rgb[cy : cy + box, cx : cx + box, :] = 255
        # Light noise avoids exact duplicates across otherwise-identical setups.
        rgb = np.clip(rgb.astype(np.int16) + self._rng.integers(-4, 5, rgb.shape), 0, 255).astype(np.uint8)

        depth = None
        if self.use_depth:
            ramp = ((self._gradient * 4.0 + phase * 10.0) % 4000.0) + 300.0
            depth = ramp.astype(np.uint16)

        self._frame += 1
        return rgb, depth

    def disconnect(self) -> None:
        self._connected = False
