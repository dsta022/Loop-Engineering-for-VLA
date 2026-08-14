#!/usr/bin/env python
"""RGB teleoperation preview / calibration (framework-free; no lerobot required).

    python ./record/rgb_record/rgb_teleoperate.py \
      --config_path=./record/rgb_record/configs/head_wrist_rgb_teleoperate.yaml [--calibrate]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recorder.cli import run_teleoperate_cli  # noqa: E402

if __name__ == "__main__":
    run_teleoperate_cli(sys.argv, allow_depth=False)
