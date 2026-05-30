"""Compatibility wrapper for the project-local RGB-D recording entrypoint."""

import sys
from pathlib import Path
import runpy

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


if __name__ == "__main__":
    runpy.run_path(str(PROJECT_ROOT / "scripts" / "record.py"), run_name="__main__")

