"""STS3215 control table + resolution + sign-magnitude encoding table.

Copied from lerobot 0.6 (``motors/feetech/tables.py``) — the addresses and the
sign-bit encoding table are hardware facts and must match exactly for the
motors to behave and for recorded joint values to line up with lerobot's.
"""

from __future__ import annotations

# Feetech STS/SMS series control table: register name -> (address, length_bytes).
STS_CONTROL_TABLE: dict[str, tuple[int, int]] = {
    "Model_Number": (3, 2),
    "ID": (5, 1),
    "Baud_Rate": (6, 1),
    "Return_Delay_Time": (7, 1),
    "Response_Status_Level": (8, 1),
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
    "Max_Temperature_Limit": (13, 1),
    "Max_Voltage_Limit": (14, 1),
    "Min_Voltage_Limit": (15, 1),
    "Max_Torque_Limit": (16, 2),
    "Phase": (18, 1),
    "Unloading_Condition": (19, 1),
    "LED_Alarm_Condition": (20, 1),
    "P_Coefficient": (21, 1),
    "D_Coefficient": (22, 1),
    "I_Coefficient": (23, 1),
    "Minimum_Startup_Force": (24, 2),
    "CW_Dead_Zone": (26, 1),
    "CCW_Dead_Zone": (27, 1),
    "Protection_Current": (28, 2),
    "Angular_Resolution": (30, 1),
    "Homing_Offset": (31, 2),
    "Operating_Mode": (33, 1),
    "Protective_Torque": (34, 1),
    "Protection_Time": (35, 1),
    "Overload_Torque": (36, 1),
    "Velocity_closed_loop_P_proportional_coefficient": (37, 1),
    "Over_Current_Protection_Time": (38, 1),
    "Velocity_closed_loop_I_integral_coefficient": (39, 1),
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Goal_Time": (44, 2),
    "Goal_Velocity": (46, 2),
    "Torque_Limit": (48, 2),
    "Lock": (55, 1),
    "Present_Position": (56, 2),
    "Present_Velocity": (58, 2),
    "Present_Load": (60, 2),
    "Present_Voltage": (62, 1),
    "Present_Temperature": (63, 1),
    "Status": (65, 1),
    "Moving": (66, 1),
    "Present_Current": (69, 2),
    "Maximum_Acceleration": (85, 1),
}

# Sign bit position for sign-magnitude registers (STS series).
STS_ENCODING_TABLE: dict[str, int] = {
    "Homing_Offset": 11,
    "Goal_Position": 15,
    "Present_Position": 15,
    "Goal_Velocity": 15,
    "Present_Velocity": 15,
}

MODEL_RESOLUTION: dict[str, int] = {
    "sts3215": 4096,
    "sts3250": 4096,
}

MODEL_NUMBER: dict[str, int] = {
    "sts3215": 777,
    "sts3250": 2825,
}

# Feetech SO-100/SO-101 arms ship at 1 Mbps.
DEFAULT_BAUDRATE = 1_000_000
PROTOCOL_VERSION = 0


def encode_sign_magnitude(value: int, sign_bit_index: int) -> int:
    """Encode a signed int as sign-magnitude with the sign at ``sign_bit_index``."""
    max_magnitude = (1 << sign_bit_index) - 1
    magnitude = abs(int(value))
    if magnitude > max_magnitude:
        raise ValueError(f"magnitude {magnitude} exceeds {max_magnitude} for sign bit {sign_bit_index}")
    direction_bit = 1 if value < 0 else 0
    return (direction_bit << sign_bit_index) | magnitude


def decode_sign_magnitude(encoded_value: int, sign_bit_index: int) -> int:
    """Decode a sign-magnitude int whose sign lives at ``sign_bit_index``."""
    direction_bit = (encoded_value >> sign_bit_index) & 1
    magnitude = encoded_value & ((1 << sign_bit_index) - 1)
    return -magnitude if direction_bit else magnitude
