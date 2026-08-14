"""Motor calibration dataclass + JSON persistence, compatible with lerobot.

Calibration files use the exact lerobot layout and schema so arms already
calibrated with lerobot keep working without re-calibration:

    <HF_LEROBOT_HOME>/calibration/<robots|teleoperators>/<device_type>/<id>.json

    {"shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 42,
                      "range_min": 900, "range_max": 3200}, ...}
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class MotorCalibration:
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


def hf_lerobot_home() -> Path:
    """Resolve the lerobot cache root the same way lerobot does."""
    explicit = os.environ.get("HF_LEROBOT_HOME")
    if explicit:
        return Path(explicit).expanduser()
    hf_home = os.environ.get("HF_HOME")
    base = Path(hf_home).expanduser() if hf_home else Path.home() / ".cache" / "huggingface"
    return base / "lerobot"


def calibration_path(device_type: str, device_id: str, *, kind: str = "robots") -> Path:
    """Return the on-disk calibration file path for a robot or teleoperator id."""
    return hf_lerobot_home() / "calibration" / kind / device_type / f"{device_id}.json"


def load_calibration(path: Path) -> dict[str, MotorCalibration]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    calibration: dict[str, MotorCalibration] = {}
    for motor, values in payload.items():
        calibration[motor] = MotorCalibration(
            id=int(values["id"]),
            drive_mode=int(values.get("drive_mode", 0)),
            homing_offset=int(values["homing_offset"]),
            range_min=int(values["range_min"]),
            range_max=int(values["range_max"]),
        )
    return calibration


def save_calibration(path: Path, calibration: dict[str, MotorCalibration]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {motor: asdict(cal) for motor, cal in calibration.items()}
    path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
