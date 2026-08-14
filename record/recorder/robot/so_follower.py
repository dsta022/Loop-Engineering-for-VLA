"""SO-100/SO-101 follower arm: Feetech bus + cameras, lerobot-free.

Behaviour mirrors the previous lerobot ``SOFollower``: six sts3215 joints
(``shoulder_pan``..``gripper``), degrees or [-100,100] body normalization with a
[0,100] gripper, optional per-camera depth normalized to ``(H, W, 1)`` uint16.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from ..cameras import make_cameras
from ..config import RobotConfig
from ..motors import MotorCalibration, calibration_path, load_calibration, save_calibration
from ..motors.feetech_bus import FeetechBus, Motor, MotorNormMode
from ..motors.mock_bus import MockBus

logger = logging.getLogger(__name__)

SO_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FULL_TURN_MOTOR = "wrist_roll"


def build_so_motors(use_degrees: bool) -> dict[str, Motor]:
    body_mode = MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    motors: dict[str, Motor] = {}
    for index, name in enumerate(SO_JOINTS, start=1):
        mode = MotorNormMode.RANGE_0_100 if name == "gripper" else body_mode
        motors[name] = Motor(id=index, model="sts3215", norm_mode=mode)
    return motors


def ensure_safe_goal_position(
    goal: dict[str, float], present: dict[str, float], max_relative_target: float
) -> dict[str, float]:
    safe: dict[str, float] = {}
    for motor, target in goal.items():
        current = present[motor]
        delta = target - current
        if abs(delta) > max_relative_target:
            delta = max(-max_relative_target, min(max_relative_target, delta))
            logger.warning("clipping '%s' relative move to %.3f", motor, max_relative_target)
        safe[motor] = current + delta
    return safe


class SOFollower:
    def __init__(self, config: RobotConfig):
        self.config = config
        self.robot_type = config.type
        self.id = config.id or config.type
        self.motors = build_so_motors(config.use_degrees)
        if config.mock:
            self.bus = MockBus(self.motors, phase=0.0)
        else:
            self.bus = FeetechBus(config.port, self.motors)
        self.cameras = make_cameras(config.cameras)

    # -- feature spec ------------------------------------------------------ #
    @property
    def motor_names(self) -> list[str]:
        return list(self.motors)

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motors}

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = dict(self.action_features)
        for name, cam in self.config.cameras.items():
            features[name] = (cam.height, cam.width, 3)
            if getattr(cam, "use_depth", False):
                features[f"{name}_depth"] = (cam.output_depth_height, cam.output_depth_width, 1)
        return features

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    # -- lifecycle --------------------------------------------------------- #
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.config.mock:
            self._load_or_run_calibration(calibrate)
            self.bus.configure_so_follower()
        for cam in self.cameras.values():
            cam.connect()
        logger.info("%s connected.", self.robot_type)

    def _load_or_run_calibration(self, calibrate: bool) -> None:
        path = calibration_path(self.robot_type, self.id, kind="robots")
        if path.exists():
            calibration = load_calibration(path)
            self.bus.write_calibration(calibration)
            logger.info("loaded calibration from %s", path)
            return
        if not calibrate:
            raise FileNotFoundError(
                f"no calibration for '{self.id}' at {path}. Run the teleoperate entrypoint "
                "with --calibrate, or copy an existing lerobot calibration file to that path."
            )
        self.run_calibration(path)

    def run_calibration(self, path) -> None:
        """Interactive homing + range recording (ENTER-driven), lerobot-compatible."""
        print(f"\nCalibrating {self.robot_type} ({self.id}).")
        self.bus.disable_torque()
        for motor in self.motors:
            self.bus.write("Operating_Mode", motor, 0)
        input("Move the arm to the MIDDLE of its range of motion, then press ENTER...")
        homing = self.bus.set_half_turn_homings()

        unknown = [m for m in self.motors if m != FULL_TURN_MOTOR]
        print(
            f"Move all joints except '{FULL_TURN_MOTOR}' through their FULL range.\n"
            "Recording... press ENTER to stop."
        )
        stop = threading.Event()
        result: dict = {}

        def _record():
            result["mins"], result["maxes"] = self.bus.record_ranges_of_motion(unknown, stop)

        worker = threading.Thread(target=_record, daemon=True)
        worker.start()
        input()
        stop.set()
        worker.join()
        mins, maxes = result["mins"], result["maxes"]
        mins[FULL_TURN_MOTOR], maxes[FULL_TURN_MOTOR] = 0, 4095

        calibration = {
            motor: MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing[motor],
                range_min=mins[motor],
                range_max=maxes[motor],
            )
            for motor, m in self.motors.items()
        }
        self.bus.write_calibration(calibration)
        save_calibration(path, calibration)
        print(f"Calibration saved to {path}")

    # -- IO ---------------------------------------------------------------- #
    def get_observation(self) -> dict:
        obs: dict = {}
        positions = self.bus.read_present_position()
        for motor, value in positions.items():
            obs[f"{motor}.pos"] = value
        for name, cam in self.cameras.items():
            color, depth = cam.read()
            obs[name] = color
            if getattr(cam, "use_depth", False) and depth is not None:
                if depth.ndim == 2:
                    depth = depth[:, :, None]
                obs[f"{name}_depth"] = depth
        return obs

    def send_action(self, action: dict) -> dict:
        goal = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        if self.config.max_relative_target is not None and not self.config.mock:
            present = self.bus.read_present_position()
            goal = ensure_safe_goal_position(goal, present, self.config.max_relative_target)
        self.bus.write_goal_position(goal)
        return {f"{motor}.pos": val for motor, val in goal.items()}

    def disconnect(self) -> None:
        if self.bus.is_connected:
            if not self.config.mock:
                self.bus.disable_torque() if self.config.disable_torque_on_disconnect else None
            self.bus.disconnect()
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()
        logger.info("%s disconnected.", self.robot_type)
