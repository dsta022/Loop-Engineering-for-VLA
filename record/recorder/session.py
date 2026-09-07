"""Wire a :class:`RecordConfig` into robot + teleop + dataset writer and run it."""

from __future__ import annotations

import copy
import logging
from pathlib import Path

from .config import MockCameraConfig, RecordConfig
from .dataset.features import build_features
from .dataset.writer import LeRobotDatasetWriter
from .motors.calibration import hf_lerobot_home
from .record_loop import run_record_session
from .robot.so_follower import SOFollower
from .teleop.so_leader import SOLeader

logger = logging.getLogger(__name__)


def apply_mock(cfg: RecordConfig) -> RecordConfig:
    """Return a copy of ``cfg`` with mock robot/teleop and synthetic cameras."""
    cfg = copy.deepcopy(cfg)
    cfg.robot.mock = True
    if cfg.teleop is not None:
        cfg.teleop.mock = True
    mock_cameras = {}
    for name, cam in cfg.robot.cameras.items():
        mock_cameras[name] = MockCameraConfig(
            type="mock",
            fps=cam.fps or cfg.dataset.fps,
            width=cam.width or 64,
            height=cam.height or 48,
            use_depth=bool(getattr(cam, "use_depth", False)),
        )
    cfg.robot.cameras = mock_cameras
    return cfg


def resolve_root(cfg: RecordConfig) -> Path:
    if cfg.dataset.root:
        return Path(cfg.dataset.root)
    return hf_lerobot_home() / cfg.dataset.repo_id


def build_robot(cfg: RecordConfig) -> SOFollower:
    return SOFollower(cfg.robot)


def build_teleop(cfg: RecordConfig) -> SOLeader | None:
    if cfg.teleop is None:
        return None
    use_degrees = cfg.teleop.use_degrees
    if use_degrees is None:
        use_degrees = cfg.robot.use_degrees
    return SOLeader(cfg.teleop, use_degrees=use_degrees)


def build_writer(cfg: RecordConfig, robot: SOFollower) -> LeRobotDatasetWriter:
    features = build_features(
        robot.motor_names,
        cfg.robot.cameras,
        fps=cfg.dataset.fps,
        use_video=cfg.dataset.video,
        vcodec=cfg.dataset.vcodec,
    )
    writer = LeRobotDatasetWriter(
        resolve_root(cfg),
        cfg.dataset.repo_id,
        cfg.dataset.fps,
        robot.robot_type,
        features,
        robot.motor_names,
        vcodec=cfg.dataset.vcodec,
        encoder_threads=cfg.dataset.encoder_threads,
        resume=cfg.resume,
    )
    writer.create()
    return writer


def run_record(cfg: RecordConfig, *, policy_runner=None) -> Path:
    """Build the stack and record a full session; returns the dataset root."""
    if cfg.dataset.repo_id == "_teleoperate_only":
        raise ValueError(
            "this config has no 'dataset' block (teleoperate-only). Add a dataset block to record."
        )
    robot = build_robot(cfg)
    teleop = build_teleop(cfg)
    writer = build_writer(cfg, robot)
    logger.info("recording %s -> %s", cfg.dataset.repo_id, writer.root)
    run_record_session(cfg, robot, teleop, writer, policy_runner=policy_runner)

    if cfg.dataset.push_to_hub:
        from .hub import push_dataset

        push_dataset(writer.root, cfg.dataset.repo_id, private=cfg.dataset.private, tags=cfg.dataset.tags)
    return writer.root
