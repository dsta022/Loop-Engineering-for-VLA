"""Shared CLI plumbing for the record/teleoperate entrypoints.

Handles ``--camera_profile`` (via the framework-free ``camera_profiles`` resolver),
a ``--mock`` flag for offline runs, config loading, and the RGB-only depth guard.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

RECORD_DIR = Path(__file__).resolve().parents[1]      # .../record
TOOLKIT_ROOT = RECORD_DIR.parent                       # repo root (for utils/, policy_improvement/)


def ensure_paths() -> None:
    for path in (str(TOOLKIT_ROOT), str(RECORD_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _pop_flag(argv: list[str], flag: str) -> tuple[list[str], bool]:
    present = flag in argv
    return [arg for arg in argv if arg != flag], present


def load_cli_config(argv: list[str], *, allow_depth: bool):
    """Resolve camera profile + config from argv; return (cfg, mock)."""
    ensure_paths()
    from camera_profiles import rewrite_argv_with_camera_profile

    from recorder.config import enforce_rgb_only, load_record_config
    from recorder.session import apply_mock

    argv = rewrite_argv_with_camera_profile(list(argv), allow_depth=allow_depth, search_root=RECORD_DIR)
    argv, mock = _pop_flag(argv, "--mock")
    cfg = load_record_config(argv[1:])
    if not allow_depth:
        enforce_rgb_only(cfg.robot)
    if mock:
        cfg = apply_mock(cfg)
    return cfg, mock


def run_record_cli(argv: list[str], *, allow_depth: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg, _ = load_cli_config(argv, allow_depth=allow_depth)
    from recorder.session import run_record

    root = run_record(cfg)
    print(f"[recorder] dataset written to: {root}")


def run_teleoperate_cli(argv: list[str], *, allow_depth: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    argv, calibrate = _pop_flag(list(argv), "--calibrate")
    cfg, _ = load_cli_config(argv, allow_depth=allow_depth)
    from recorder.teleoperate import run_teleoperate

    run_teleoperate(cfg, calibrate=calibrate)


def run_feedback_cli(argv: list[str], *, allow_depth: bool = True) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg, mock = load_cli_config(argv, allow_depth=allow_depth)
    from recorder.feedback import run_feedback

    root = run_feedback(cfg, mock=mock)
    print(f"[recorder] feedback dataset written to: {root}")
