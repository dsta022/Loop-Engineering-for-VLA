#!/usr/bin/env python
"""RGB-D recording entrypoint (framework-free; no lerobot required).

    python ./record/rgbd_record/record.py \
      --config_path=./record/rgbd_record/configs/record_rgbd.yaml \
      [--camera_profile=realsense_d435_front_d405_wrist] [--mock]

RGB streams are stored as mp4 video; depth streams are stored as lossless uint16
PNG sidecars under depth_sidecar/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # record/  (for `recorder`)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (for utils/, policy_improvement/)

from recorder.cli import run_record_cli  # noqa: E402

if __name__ == "__main__":
    run_record_cli(sys.argv, allow_depth=True)
