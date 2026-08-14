"""Teleoperated (or policy-driven) recording loop, lerobot-free.

Mirrors the control flow of the previous lerobot recorder: per episode, read the
robot observation, obtain an action (from the leader teleoperator or an optional
:class:`~recorder.policy_runner.PolicyRunner`), send it to the follower, and hand
the frame to the dataset writer at a fixed fps. Right/Left/Esc keyboard events end,
re-record, or stop the session.
"""

from __future__ import annotations

import logging
import time

from .config import RecordConfig
from .dataset.writer import LeRobotDatasetWriter
from .teleop.keyboard import init_keyboard_listener

logger = logging.getLogger(__name__)


def _precise_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def run_episode(
    *,
    robot,
    teleop,
    writer: LeRobotDatasetWriter | None,
    events: dict,
    fps: int,
    task: str,
    control_time_s: float,
    policy_runner=None,
    max_frames: int | None = None,
) -> None:
    """Run one episode (record when ``writer`` is provided, else a reset phase)."""
    if policy_runner is not None:
        policy_runner.reset()
    period = 1.0 / fps
    start = time.perf_counter()
    timestamp = 0.0
    frames = 0
    while timestamp < control_time_s:
        loop_start = time.perf_counter()
        if events.get("exit_early"):
            events["exit_early"] = False
            break

        observation = robot.get_observation()
        if policy_runner is not None:
            action = policy_runner.select_action(observation, task=task)
        elif teleop is not None:
            action = teleop.get_action()
        else:
            logger.warning("no teleop or policy provided; skipping action")
            action = {f"{motor}.pos": observation[f"{motor}.pos"] for motor in robot.motor_names}

        sent = robot.send_action(action)
        if writer is not None:
            writer.add_frame(observation, sent, task)
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break

        dt = time.perf_counter() - loop_start
        if dt > period:
            logger.warning("record loop slower than target fps: %.1f < %d Hz", 1.0 / max(dt, 1e-9), fps)
        _precise_sleep(period - dt)
        timestamp = time.perf_counter() - start


def run_record_session(
    cfg: RecordConfig,
    robot,
    teleop,
    writer: LeRobotDatasetWriter,
    *,
    policy_runner=None,
) -> None:
    """Record ``cfg.dataset.num_episodes`` episodes with reset + keyboard control."""
    listener, events = init_keyboard_listener()
    task = cfg.dataset.single_task
    fps = cfg.dataset.fps
    try:
        robot.connect()
        if teleop is not None:
            teleop.connect()

        recorded = 0
        while recorded < cfg.dataset.num_episodes and not events["stop_recording"]:
            logger.info("Recording episode %d", writer.total_episodes)
            run_episode(
                robot=robot, teleop=teleop, writer=writer, events=events,
                fps=fps, task=task, control_time_s=cfg.dataset.episode_time_s,
                policy_runner=policy_runner,
            )

            if events["rerecord_episode"]:
                logger.info("Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                writer.clear_episode_buffer()
                continue

            # Optional reset phase between episodes (no recording).
            more_episodes = recorded < cfg.dataset.num_episodes - 1
            if not events["stop_recording"] and more_episodes and cfg.dataset.reset_time_s > 0:
                logger.info("Reset the environment")
                run_episode(
                    robot=robot, teleop=teleop, writer=None, events=events,
                    fps=fps, task=task, control_time_s=cfg.dataset.reset_time_s,
                    policy_runner=None,
                )

            writer.save_episode()
            recorded += 1
    finally:
        writer.finalize() if writer.total_episodes > 0 else None
        if robot.is_connected:
            robot.disconnect()
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()
        if listener is not None:
            listener.stop()
