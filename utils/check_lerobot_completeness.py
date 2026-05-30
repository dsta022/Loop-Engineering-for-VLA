#!/usr/bin/env python3
"""Completeness-only entry point for LeRobot RGB and RGB-D datasets.

This tool runs the deterministic structural checks from the main audit without
semantic/content review checks. It is useful after merge/copy operations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lerobot_data_quality_audit import audit_dataset  # noqa: E402
from utils.quality_utils import DEFAULT_REPORT_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Check LeRobot dataset structural completeness.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--depth-check", choices=["none", "header", "full"], default="header")
    parser.add_argument("--video-check", choices=["metadata", "ffprobe", "decode"], default="ffprobe")
    args = parser.parse_args()

    out_dir = args.out_dir or args.dataset.parent / DEFAULT_REPORT_DIR / f"{args.dataset.name}_completeness"
    raise SystemExit(
        audit_dataset(
            root=args.dataset,
            out_dir=out_dir,
            depth_check=args.depth_check,
            video_check=args.video_check,
            video_content_check="none",
            sample_frames=0,
            depth_sample_frames=0,
            duplicate_check="none",
            video_sample_stride=300,
        )
    )


if __name__ == "__main__":
    main()
