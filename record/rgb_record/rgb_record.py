#!/usr/bin/env python
"""RGB-only recording entrypoint (framework-free; no lerobot required).

    python ./record/rgb_record/rgb_record.py \
      --config_path=./record/rgb_record/configs/head_wrist_rgb_record.yaml \
      [--camera_profile=rgb_head_wrist] [--mock] [--dataset.num_episodes=5]

Depth-enabled profiles/cameras are rejected here; use ../rgbd_record for depth.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # record/  (for `recorder`)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (for utils/, policy_improvement/)

from recorder.cli import run_record_cli  # noqa: E402

if __name__ == "__main__":
    run_record_cli(sys.argv, allow_depth=False)
