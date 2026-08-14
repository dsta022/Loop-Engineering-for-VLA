"""Framework-free (lerobot-free) recording library for the VLAnything toolkit.

This package replaces the previous "copy into a lerobot checkout" overlay. It
talks to cameras and Feetech motors directly and writes LeRobot **v3.0** datasets
using this toolkit's own lerobot-free helpers in ``utils/`` — so recording no
longer depends on the lerobot framework.

Only the optional in-loop policy inference used by ``feedback_record`` needs a
policy runtime, and that is isolated behind :class:`recorder.policy_runner.PolicyRunner`.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
