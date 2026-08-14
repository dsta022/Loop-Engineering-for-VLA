"""Feetech STS3215 motor bus (scservo_sdk), ported lerobot-free from lerobot 0.6.

Reproduces lerobot's calibration + normalization exactly so recorded joint
values and sent goal positions match what lerobot produces on the same arm:

* homing offset is written to the motor ``Homing_Offset`` register at connect,
* ``Present_Position`` is read raw, sign-magnitude decoded (bit 15), then mapped
  to the motor's norm range,
* goal positions are un-normalized then sign-magnitude encoded (bit 15).

SO-100/SO-101 arms all use ``drive_mode=0``, so the drive-mode sign flip is a
no-op here, but it is implemented faithfully.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass

from .calibration import MotorCalibration
from .tables import (
    DEFAULT_BAUDRATE,
    MODEL_RESOLUTION,
    PROTOCOL_VERSION,
    STS_CONTROL_TABLE,
    STS_ENCODING_TABLE,
    decode_sign_magnitude,
    encode_sign_magnitude,
)

logger = logging.getLogger(__name__)


class MotorNormMode(str, enum.Enum):
    RANGE_0_100 = "range_0_100"
    RANGE_M100_100 = "range_m100_100"
    DEGREES = "degrees"


@dataclass
class Motor:
    id: int
    model: str
    norm_mode: MotorNormMode


class FeetechBus:
    apply_drive_mode = True

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        *,
        baudrate: int = DEFAULT_BAUDRATE,
        protocol_version: int = PROTOCOL_VERSION,
    ):
        self.port = port
        self.motors = motors
        self.calibration = calibration or {}
        self.baudrate = baudrate
        self.protocol_version = protocol_version
        self._scs = None
        self._port_handler = None
        self._packet_handler = None

    def __str__(self) -> str:
        return f"FeetechBus({self.port})"

    # -- connection -------------------------------------------------------- #
    @property
    def is_connected(self) -> bool:
        return self._port_handler is not None

    def connect(self) -> None:
        import scservo_sdk as scs

        if self.is_connected:
            return
        self._scs = scs
        self._port_handler = scs.PortHandler(self.port)
        self._packet_handler = scs.PacketHandler(self.protocol_version)
        if not self._port_handler.openPort():
            self._port_handler = None
            raise ConnectionError(f"{self} failed to open serial port {self.port}.")
        if not self._port_handler.setBaudRate(self.baudrate):
            self.disconnect()
            raise ConnectionError(f"{self} failed to set baudrate {self.baudrate}.")
        for motor, m in self.motors.items():
            _, comm, err = self._packet_handler.ping(self._port_handler, m.id)
            if comm != scs.COMM_SUCCESS:
                self.disconnect()
                raise ConnectionError(
                    f"{self} could not ping motor '{motor}' (id {m.id}). "
                    f"Check wiring, port, and baudrate."
                )

    def disconnect(self) -> None:
        if self._port_handler is not None:
            try:
                self._port_handler.closePort()
            finally:
                self._port_handler = None
                self._packet_handler = None

    # -- low-level single register access ---------------------------------- #
    def _addr_len(self, reg: str) -> tuple[int, int]:
        if reg not in STS_CONTROL_TABLE:
            raise KeyError(f"unknown register {reg!r}")
        return STS_CONTROL_TABLE[reg]

    def write(self, reg: str, motor: str, value: int) -> None:
        scs = self._scs
        addr, length = self._addr_len(reg)
        motor_id = self.motors[motor].id
        raw = int(value)
        if reg in STS_ENCODING_TABLE:
            raw = encode_sign_magnitude(raw, STS_ENCODING_TABLE[reg])
        if length == 1:
            comm, err = self._packet_handler.write1ByteTxRx(self._port_handler, motor_id, addr, raw)
        elif length == 2:
            comm, err = self._packet_handler.write2ByteTxRx(self._port_handler, motor_id, addr, raw)
        else:
            raise ValueError(f"unsupported register length {length} for {reg}")
        if comm != scs.COMM_SUCCESS:
            raise IOError(f"{self} write {reg}={value} to '{motor}' failed (comm={comm}, err={err}).")

    def read(self, reg: str, motor: str) -> int:
        scs = self._scs
        addr, length = self._addr_len(reg)
        motor_id = self.motors[motor].id
        if length == 1:
            value, comm, err = self._packet_handler.read1ByteTxRx(self._port_handler, motor_id, addr)
        elif length == 2:
            value, comm, err = self._packet_handler.read2ByteTxRx(self._port_handler, motor_id, addr)
        else:
            raise ValueError(f"unsupported register length {length} for {reg}")
        if comm != scs.COMM_SUCCESS:
            raise IOError(f"{self} read {reg} from '{motor}' failed (comm={comm}, err={err}).")
        if reg in STS_ENCODING_TABLE:
            value = decode_sign_magnitude(int(value), STS_ENCODING_TABLE[reg])
        return int(value)

    # -- torque / configuration ------------------------------------------- #
    def disable_torque(self) -> None:
        for motor in self.motors:
            self.write("Torque_Enable", motor, 0)
            self.write("Lock", motor, 0)

    def enable_torque(self) -> None:
        for motor in self.motors:
            self.write("Torque_Enable", motor, 1)
            self.write("Lock", motor, 1)

    def write_calibration(self, calibration: dict[str, MotorCalibration]) -> None:
        for motor, cal in calibration.items():
            self.write("Homing_Offset", motor, cal.homing_offset)
            self.write("Min_Position_Limit", motor, cal.range_min)
            self.write("Max_Position_Limit", motor, cal.range_max)
        self.calibration = calibration

    def configure_so_follower(self) -> None:
        """Position mode + gentle PID; matches lerobot SOFollower.configure()."""
        for motor in self.motors:
            self.write("Operating_Mode", motor, 0)  # position control
            self.write("P_Coefficient", motor, 16)
            self.write("I_Coefficient", motor, 0)
            self.write("D_Coefficient", motor, 32)
            if motor == "gripper":
                self.write("Max_Torque_Limit", motor, 500)
                self.write("Protection_Current", motor, 250)
                self.write("Overload_Torque", motor, 25)

    # -- sync read/write with calibration ---------------------------------- #
    def _sync_read_raw(self, reg: str) -> dict[str, int]:
        scs = self._scs
        addr, length = self._addr_len(reg)
        group = scs.GroupSyncRead(self._port_handler, self._packet_handler, addr, length)
        for m in self.motors.values():
            group.addParam(m.id)
        comm = group.txRxPacket()
        if comm != scs.COMM_SUCCESS:
            raise IOError(f"{self} sync read {reg} failed (comm={comm}).")
        values: dict[str, int] = {}
        for motor, m in self.motors.items():
            raw = int(group.getData(m.id, addr, length))
            if reg in STS_ENCODING_TABLE:
                raw = decode_sign_magnitude(raw, STS_ENCODING_TABLE[reg])
            values[motor] = raw
        return values

    def _sync_write_raw(self, reg: str, values: dict[str, int]) -> None:
        scs = self._scs
        addr, length = self._addr_len(reg)
        group = scs.GroupSyncWrite(self._port_handler, self._packet_handler, addr, length)
        for motor, value in values.items():
            raw = int(value)
            if reg in STS_ENCODING_TABLE:
                raw = encode_sign_magnitude(raw, STS_ENCODING_TABLE[reg])
            data = [scs.SCS_LOBYTE(raw), scs.SCS_HIBYTE(raw)] if length == 2 else [raw]
            group.addParam(self.motors[motor].id, data)
        comm = group.txPacket()
        if comm != scs.COMM_SUCCESS:
            raise IOError(f"{self} sync write {reg} failed (comm={comm}).")

    def _normalize(self, motor: str, raw: int) -> float:
        cal = self.calibration[motor]
        min_, max_ = cal.range_min, cal.range_max
        if max_ == min_:
            raise ValueError(f"invalid calibration for '{motor}': min == max")
        drive = self.apply_drive_mode and cal.drive_mode
        mode = self.motors[motor].norm_mode
        bounded = min(max_, max(min_, raw))
        if mode is MotorNormMode.RANGE_M100_100:
            norm = ((bounded - min_) / (max_ - min_)) * 200 - 100
            return -norm if drive else norm
        if mode is MotorNormMode.RANGE_0_100:
            norm = ((bounded - min_) / (max_ - min_)) * 100
            return 100 - norm if drive else norm
        # DEGREES
        mid = (min_ + max_) / 2
        max_res = MODEL_RESOLUTION[self.motors[motor].model] - 1
        return (raw - mid) * 360 / max_res

    def _unnormalize(self, motor: str, value: float) -> int:
        cal = self.calibration[motor]
        min_, max_ = cal.range_min, cal.range_max
        if max_ == min_:
            raise ValueError(f"invalid calibration for '{motor}': min == max")
        drive = self.apply_drive_mode and cal.drive_mode
        mode = self.motors[motor].norm_mode
        if mode is MotorNormMode.RANGE_M100_100:
            value = -value if drive else value
            bounded = min(100.0, max(-100.0, value))
            return int(((bounded + 100) / 200) * (max_ - min_) + min_)
        if mode is MotorNormMode.RANGE_0_100:
            value = 100 - value if drive else value
            bounded = min(100.0, max(0.0, value))
            return int((bounded / 100) * (max_ - min_) + min_)
        # DEGREES
        mid = (min_ + max_) / 2
        max_res = MODEL_RESOLUTION[self.motors[motor].model] - 1
        return int((value * max_res / 360) + mid)

    def read_present_position(self) -> dict[str, float]:
        raw = self._sync_read_raw("Present_Position")
        return {motor: self._normalize(motor, value) for motor, value in raw.items()}

    def write_goal_position(self, goal: dict[str, float]) -> None:
        raw = {motor: self._unnormalize(motor, value) for motor, value in goal.items()}
        self._sync_write_raw("Goal_Position", raw)

    # -- interactive calibration ------------------------------------------ #
    def set_half_turn_homings(self) -> dict[str, int]:
        """Center each motor's range: homing_offset = present - resolution/2."""
        homing: dict[str, int] = {}
        for motor in self.motors:
            self.write("Homing_Offset", motor, 0)
        raw = self._sync_read_raw("Present_Position")
        for motor, present in raw.items():
            max_res = MODEL_RESOLUTION[self.motors[motor].model]
            offset = present - max_res // 2
            homing[motor] = offset
            self.write("Homing_Offset", motor, offset)
        return homing

    def record_ranges_of_motion(self, motors: list[str], stop_event) -> tuple[dict[str, int], dict[str, int]]:
        """Stream positions until ``stop_event`` is set; return per-motor min/max."""
        mins = {motor: 4095 for motor in motors}
        maxes = {motor: 0 for motor in motors}
        while not stop_event.is_set():
            raw = self._sync_read_raw("Present_Position")
            for motor in motors:
                mins[motor] = min(mins[motor], raw[motor])
                maxes[motor] = max(maxes[motor], raw[motor])
            time.sleep(0.01)
        return mins, maxes
