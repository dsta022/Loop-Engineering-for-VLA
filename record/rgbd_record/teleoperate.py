#!/usr/bin/env python
"""RGB-D teleoperation preview / calibration (framework-free).

    python ./record/rgbd_record/teleoperate.py \
      --config_path=./record/rgbd_record/configs/record_rgbd.yaml [--calibrate]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recorder.cli import run_teleoperate_cli  # noqa: E402

if __name__ == "__main__":
    run_teleoperate_cli(sys.argv, allow_depth=True)
