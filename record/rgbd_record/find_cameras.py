#!/usr/bin/env python
"""List connected cameras (framework-free replacement for lerobot-find-cameras).

    python ./record/rgbd_record/find_cameras.py opencv
    python ./record/rgbd_record/find_cameras.py realsense
    python ./record/rgbd_record/find_cameras.py orbbec
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # record/  (for `recorder`)

from recorder.find_cameras import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv)
