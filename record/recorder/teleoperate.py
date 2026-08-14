"""Teleoperation preview + calibration (no recording), lerobot-free."""

from __future__ import annotations

import logging
import time

from .config import RecordConfig
from .motors.calibration import calibration_path
from .session import build_robot, build_teleop

logger = logging.getLogger(__name__)


def run_teleoperate(cfg: RecordConfig, *, calibrate: bool = False, duration_s: float | None = None) -> None:
    """Stream leader -> follower. With ``calibrate`` run interactive calibration first."""
    robot = build_robot(cfg)
    teleop = build_teleop(cfg)
    if teleop is None:
        raise ValueError("teleoperation requires a 'teleop' block in the config")

    if calibrate:
        robot.bus.connect()
        robot.run_calibration(calibration_path(robot.robot_type, robot.id, kind="robots"))
        robot.bus.disconnect()
        teleop.bus.connect()
        teleop.run_calibration(calibration_path(teleop.teleop_type, teleop.id, kind="teleoperators"))
        teleop.bus.disconnect()
        print("[recorder] calibration complete.")
        return

    fps = cfg.dataset.fps
    period = 1.0 / fps
    robot.connect()
    teleop.connect()
    print("[recorder] teleoperation preview. Press Ctrl+C to stop.")
    start = time.perf_counter()
    try:
        while duration_s is None or (time.perf_counter() - start) < duration_s:
            loop_start = time.perf_counter()
            action = teleop.get_action()
            robot.send_action(action)
            dt = time.perf_counter() - loop_start
            if period - dt > 0:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n[recorder] stopping teleoperation.")
    finally:
        robot.disconnect()
        teleop.disconnect()
