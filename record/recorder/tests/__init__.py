"""Unit tests for the framework-free recorder.

Run from the repository root with the project's Python (no pytest):

    python -m unittest discover -s record/recorder/tests -t .

The package inserts the repo root and record/ onto sys.path so `recorder`,
`utils`, and `policy_improvement` all import.
"""

from __future__ import annotations

import sys
from pathlib import Path

RECORD_DIR = Path(__file__).resolve().parents[2]      # .../record
TOOLKIT_ROOT = RECORD_DIR.parent                       # repo root
for _path in (str(TOOLKIT_ROOT), str(RECORD_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
