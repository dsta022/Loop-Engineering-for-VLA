"""SO-100/SO-101 leader arm (passive Feetech arm read for teleoperation).

The leader is back-drivable: torque stays off and we only read normalized joint
positions. Its normalization mode must match the follower so leader positions map
straight onto follower goal positions.
"""

from __future__ import annotations

import logging
import threading

from ..config import TeleopConfig
from ..motors import MotorCalibration, calibration_path, load_calibration, save_calibration
from ..motors.feetech_bus import FeetechBus
from ..motors.mock_bus import MockBus
from ..robot.so_follower import FULL_TURN_MOTOR, build_so_motors

logger = logging.getLogger(__name__)


class SOLeader:
    def __init__(self, config: TeleopConfig, use_degrees: bool = False):
        self.config = config
        self.teleop_type = config.type
        self.id = config.id or config.type
        self.motors = build_so_motors(use_degrees)
        if config.mock:
            # Phase offset so the leader trajectory differs from the follower.
            self.bus = MockBus(self.motors, phase=0.35)
        else:
            self.bus = FeetechBus(config.port, self.motors)

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.config.mock:
            self._load_or_run_calibration(calibrate)
        logger.info("%s connected.", self.teleop_type)

    def _load_or_run_calibration(self, calibrate: bool) -> None:
        path = calibration_path(self.teleop_type, self.id, kind="teleoperators")
        if path.exists():
            self.bus.write_calibration(load_calibration(path))
            logger.info("loaded leader calibration from %s", path)
            return
        if not calibrate:
            raise FileNotFoundError(
                f"no leader calibration for '{self.id}' at {path}. Run teleoperate --calibrate "
                "or copy an existing lerobot calibration file to that path."
            )
        self.run_calibration(path)

    def run_calibration(self, path) -> None:
        print(f"\nCalibrating leader {self.teleop_type} ({self.id}).")
        self.bus.disable_torque()
        input("Move the leader arm to the MIDDLE of its range, then press ENTER...")
        homing = self.bus.set_half_turn_homings()
        unknown = [m for m in self.motors if m != FULL_TURN_MOTOR]
        print(f"Move all joints except '{FULL_TURN_MOTOR}' through their FULL range. Press ENTER to stop.")
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
                id=m.id, drive_mode=0, homing_offset=homing[motor],
                range_min=mins[motor], range_max=maxes[motor],
            )
            for motor, m in self.motors.items()
        }
        self.bus.write_calibration(calibration)
        save_calibration(path, calibration)
        print(f"Leader calibration saved to {path}")

    def get_action(self) -> dict:
        return {f"{motor}.pos": value for motor, value in self.bus.read_present_position().items()}

    def disconnect(self) -> None:
        if self.bus.is_connected:
            self.bus.disconnect()
        logger.info("%s disconnected.", self.teleop_type)
