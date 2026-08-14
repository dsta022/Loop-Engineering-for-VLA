#!/usr/bin/env python
"""Human-guided policy feedback collection (framework-free recorder).

    python ./record/rgbd_record/feedback_record.py \
      --config_path=./record/rgbd_record/configs/record_rgbd_feedback.yaml \
      --policy.path=./outputs/train/run/checkpoints/last/pretrained_model

Teleoperated capture, intervention labelling, and the feedback sidecar are all
framework-free. Only the in-loop policy inference uses a policy runtime, isolated
behind recorder.policy_runner.PolicyRunner (default: a lazy lerobot adapter).
Use --mock for an offline dry-run with a stub policy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # record/  (for `recorder`)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (for utils/, policy_improvement/)

from recorder.cli import run_feedback_cli  # noqa: E402

if __name__ == "__main__":
    run_feedback_cli(sys.argv, allow_depth=True)
