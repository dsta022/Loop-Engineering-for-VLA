"""Synthetic motor bus for offline testing (no serial hardware).

Mirrors the :class:`FeetechBus` interface used by the robot/teleop layers and
produces smooth synthetic joint motion so recorded state/action pass the audit
(finite values, small jumps, non-zero motion).
"""

from __future__ import annotations

import math

from .feetech_bus import Motor


class MockBus:
    def __init__(self, motors: dict[str, Motor], *, phase: float = 0.0, **_ignored):
        self.motors = motors
        self.calibration: dict = {}
        self._connected = False
        self._t = 0
        self._phase = phase

    def __str__(self) -> str:
        return "MockBus"

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    # No-op configuration hooks (parity with FeetechBus).
    def disable_torque(self) -> None: ...
    def enable_torque(self) -> None: ...
    def configure_so_follower(self) -> None: ...
    def write_calibration(self, calibration) -> None:
        self.calibration = calibration

    def _positions(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for i, motor in enumerate(self.motors):
            angle = self._t * 0.1 + self._phase + i * 0.5
            values[motor] = 30.0 * math.sin(angle)
        return values

    def read_present_position(self) -> dict[str, float]:
        values = self._positions()
        self._t += 1
        return values

    def write_goal_position(self, goal: dict[str, float]) -> None:  # noqa: ARG002
        return None

    def set_half_turn_homings(self) -> dict[str, int]:
        return {motor: 0 for motor in self.motors}

    def record_ranges_of_motion(self, motors, stop_event):  # noqa: ARG002
        return {m: 0 for m in motors}, {m: 4095 for m in motors}
