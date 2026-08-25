"""Human-guided policy feedback collection (framework-free recorder core).

Reuses the toolkit's lerobot-free ``policy_improvement`` intervention/capture/
sidecar machinery unchanged; only the dataset writer, robot/teleop, and in-loop
policy inference are the framework-free recorder's own. Behaviour mirrors the
previous lerobot ``feedback_record`` (corrections_only / event_buffer / continuous
modes, explicit Recovery/Correction labelling, feedback sidecar outside the dataset).
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from policy_improvement.capture import (
    AsyncTriggerMonitor,
    BoundedCaptureBuffer,
    CapturedPacket,
    EpisodeRotator,
    EventCaptureController,
    TriggerSignal,
    load_trigger_config,
    load_window_trigger,
)
from policy_improvement.contracts import (
    CaptureMode,
    ControlPhase,
    EpisodeFeedback,
    FeedbackFrame,
    FeedbackSidecarWriter,
    InterventionSegment,
    TriggerSource,
)
from policy_improvement.intervention import (
    CapabilityHandoverAdapter,
    InterventionCommand,
    InterventionController,
    InterventionEventQueue,
    InterventionKeyConfig,
    create_intervention_key_listener,
)

from .config import RecordConfig
from .dataset.writer import LeRobotDatasetWriter
from .policy_runner import build_policy_runner
from .session import build_robot, build_teleop, build_writer
from .teleop.keyboard import should_attempt_keyboard_listener

logger = logging.getLogger(__name__)


@dataclass
class AutomaticTriggerConfig:
    plugin: str | None = None
    config_path: Path | None = None
    interval_frames: int = 30
    post_trigger_seconds: float = 5.0


@dataclass
class FeedbackCollectionConfig:
    mode: str = CaptureMode.CORRECTIONS_ONLY.value
    sidecar_dir: Path | None = None
    session_id: str | None = None
    source_policy: str = ""
    overwrite_sidecar: bool = False
    keyboard_enabled: bool = True
    toggle_control_key: str = "i"
    begin_correction_key: str = "c"
    cancel_key: str = "x"
    save_event_key: str = "h"
    stop_key: str = "esc"
    buffer_seconds: float = 15.0
    buffer_memory_mb: float = 2048.0
    require_feedback_alignment: bool = True
    synchronization_seconds: float = 0.5
    automatic_trigger: AutomaticTriggerConfig = field(default_factory=AutomaticTriggerConfig)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "FeedbackCollectionConfig":
        payload = dict(payload or {})
        auto = dict(payload.pop("automatic_trigger", {}) or {})
        known = {f for f in cls.__dataclass_fields__ if f != "automatic_trigger"}
        kwargs = {k: v for k, v in payload.items() if k in known}
        if "sidecar_dir" in kwargs and kwargs["sidecar_dir"] is not None:
            kwargs["sidecar_dir"] = Path(kwargs["sidecar_dir"])
        cfg = cls(**kwargs)
        cfg.automatic_trigger = AutomaticTriggerConfig(
            plugin=auto.get("plugin"),
            config_path=Path(auto["config_path"]) if auto.get("config_path") else None,
            interval_frames=int(auto.get("interval_frames", 30)),
            post_trigger_seconds=float(auto.get("post_trigger_seconds", 5.0)),
        )
        CaptureMode(cfg.mode)
        return cfg


@dataclass(frozen=True)
class DatasetPacket:
    timestamp: float
    observation: Mapping[str, Any]
    executed_action: Any
    policy_action: Any
    phase: ControlPhase
    intervention_id: int
    trigger_source: TriggerSource
    training_eligible: bool
    details: Mapping[str, Any] = field(default_factory=dict)


def clone_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: clone_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone_value(v) for v in value)
    try:
        return copy.deepcopy(value)
    except Exception:  # noqa: BLE001
        return value


def _make_feedback_frame(writer: LeRobotDatasetWriter, packet: DatasetPacket, origin: float) -> FeedbackFrame:
    return FeedbackFrame(
        episode_index=writer.total_episodes,
        frame_index=writer.current_episode_length,
        timestamp=max(0.0, packet.timestamp - origin),
        phase=packet.phase,
        intervention_id=packet.intervention_id,
        trigger_source=packet.trigger_source,
        policy_action=packet.policy_action,
        executed_action=packet.executed_action,
        training_eligible=packet.training_eligible,
        details=packet.details,
    )


def _store_packet(writer, packet, origin, frames, task):
    frame = _make_feedback_frame(writer, packet, origin)
    writer.add_frame(dict(packet.observation), packet.executed_action, task)
    frames.append(frame)


def _segment(episode_index: int, window: Any) -> InterventionSegment:
    return InterventionSegment(
        episode_index=episode_index,
        intervention_id=window.intervention_id,
        recovery_start_frame=window.recovery_start_frame,
        recovery_end_frame=window.recovery_end_frame,
        correction_start_frame=window.correction_start_frame,
        correction_end_frame=window.correction_end_frame,
        trigger_source=window.trigger_source,
        outcome=window.outcome,
    )


def _save_feedback_episode(writer, sidecar, frames, segments, *, task, source_policy, mode, outcome, trigger_details=None):
    if not frames:
        return
    episode_index = writer.total_episodes
    writer.save_episode()
    sidecar.append_episode(
        frames,
        EpisodeFeedback(
            episode_index=episode_index,
            length=len(frames),
            source_policy=source_policy,
            capture_mode=mode,
            task=task,
            outcome=outcome,
            intervention_count=len(segments),
            details=dict(trigger_details or {}),
        ),
        segments,
    )


def run_feedback_loop(*, cfg, fb_cfg, robot, teleop, writer, policy_runner, sidecar) -> None:
    mode = CaptureMode(fb_cfg.mode)
    task = cfg.dataset.single_task
    source_policy = fb_cfg.source_policy or (cfg.policy.path if cfg.policy else "") or ""
    controller = InterventionController()
    handover = CapabilityHandoverAdapter(
        synchronization_seconds=fb_cfg.synchronization_seconds,
        require_feedback_alignment=fb_cfg.require_feedback_alignment,
    )
    events = InterventionEventQueue()
    listener = None
    if fb_cfg.keyboard_enabled and should_attempt_keyboard_listener():
        try:
            listener = create_intervention_key_listener(
                events,
                InterventionKeyConfig(
                    toggle_control=fb_cfg.toggle_control_key,
                    begin_correction=fb_cfg.begin_correction_key,
                    cancel=fb_cfg.cancel_key,
                    save_event=fb_cfg.save_event_key,
                    stop=fb_cfg.stop_key,
                ),
            )
        except RuntimeError as exc:
            logger.warning("keyboard intervention disabled: %s", exc)

    ring = BoundedCaptureBuffer(
        max_seconds=fb_cfg.buffer_seconds, fps=cfg.dataset.fps, max_memory_mb=fb_cfg.buffer_memory_mb
    )
    event_capture = EventCaptureController(ring)
    rotator = EpisodeRotator(cfg.dataset.episode_time_s)
    trigger_monitor = None
    if fb_cfg.automatic_trigger.plugin:
        if mode != CaptureMode.EVENT_BUFFER:
            raise ValueError("automatic triggers are only valid in event_buffer mode")
        trigger_monitor = AsyncTriggerMonitor(
            load_window_trigger(
                fb_cfg.automatic_trigger.plugin,
                load_trigger_config(fb_cfg.automatic_trigger.config_path),
            )
        )

    frames: list[FeedbackFrame] = []
    segments: list[InterventionSegment] = []
    origin: float | None = None
    event_deadline: float | None = None
    last_robot_action: Any = None
    last_policy_action: Any = None
    loop_index = 0
    stop_requested = False

    def flush_event_buffer(signal: TriggerSignal) -> None:
        nonlocal origin, event_deadline
        buffered = event_capture.start(signal)
        if buffered:
            origin = buffered[0].timestamp
            for captured in buffered:
                _store_packet(writer, captured.payload, origin, frames, task)
        elif origin is None:
            origin = time.perf_counter()
        event_deadline = time.perf_counter() + fb_cfg.automatic_trigger.post_trigger_seconds

    def finish_episode(outcome: str) -> None:
        nonlocal frames, segments, origin, event_deadline
        signal = event_capture.stop() if mode == CaptureMode.EVENT_BUFFER else None
        trigger_details = None
        if signal is not None:
            trigger_details = {
                "trigger_source": TriggerSource(signal.source).value,
                "trigger_score": signal.score,
                "trigger_reason": signal.reason,
                "trigger_details": dict(signal.details),
            }
        _save_feedback_episode(
            writer, sidecar, frames, segments, task=task, source_policy=source_policy,
            mode=mode, outcome=outcome, trigger_details=trigger_details,
        )
        frames = []
        segments = []
        origin = None
        event_deadline = None
        controller.reset_episode()
        rotator.reset()

    policy_runner.reset()
    try:
        while writer.total_episodes < cfg.dataset.num_episodes:
            loop_start = time.perf_counter()
            for command in events.drain():
                phase = controller.phase
                if command == InterventionCommand.STOP:
                    stop_requested = True
                elif command == InterventionCommand.TOGGLE_CONTROL:
                    if phase == ControlPhase.AUTONOMOUS:
                        try:
                            controller.request_takeover(TriggerSource.MANUAL)
                            handover.prepare_takeover(robot, teleop, last_robot_action)
                            if mode == CaptureMode.EVENT_BUFFER and not event_capture.recording:
                                flush_event_buffer(TriggerSignal(True, TriggerSource.MANUAL, reason="human_takeover"))
                            controller.confirm_synchronized(writer.current_episode_length)
                            handover.begin_human_control(robot, teleop)
                            logger.info("Human recovery started")
                        except Exception:
                            controller.cancel()
                            raise
                    elif phase == ControlPhase.CORRECTION:
                        window = controller.complete(writer.current_episode_length)
                        handover.end_human_control(robot, teleop)
                        handover.resume_autonomous(robot, teleop)
                        policy_runner.reset()
                        segments.append(_segment(writer.total_episodes, window))
                        logger.info("Human correction completed")
                        if mode in {CaptureMode.CORRECTIONS_ONLY, CaptureMode.EVENT_BUFFER}:
                            finish_episode("success_after_intervention")
                elif command == InterventionCommand.BEGIN_CORRECTION:
                    controller.begin_correction(writer.current_episode_length)
                    logger.info("Correction phase started")
                elif command == InterventionCommand.SAVE_EVENT and mode == CaptureMode.EVENT_BUFFER:
                    if not event_capture.recording:
                        flush_event_buffer(TriggerSignal(True, TriggerSource.MANUAL, reason="manual_event"))
                    elif controller.phase == ControlPhase.AUTONOMOUS:
                        finish_episode("captured_event")
                elif command == InterventionCommand.CANCEL:
                    logger.warning("cancel ignored after human motion begins; complete the correction")

            if stop_requested and controller.phase == ControlPhase.AUTONOMOUS:
                break
            if trigger_monitor is not None:
                signal = trigger_monitor.poll()
                if signal is not None and signal.triggered and not event_capture.recording:
                    flush_event_buffer(signal)

            observation = robot.get_observation()
            phase = controller.phase
            if phase == ControlPhase.AUTONOMOUS:
                policy_action = policy_runner.select_action(observation, task=task)
                decision = controller.select_action(policy_action)
                last_policy_action = clone_value(policy_action)
            else:
                human_action = teleop.get_action()
                decision = controller.select_action(last_policy_action, human_action)

            if decision.hold_position:
                if last_robot_action is not None:
                    robot.send_action(last_robot_action)
                time.sleep(max(1 / cfg.dataset.fps - (time.perf_counter() - loop_start), 0))
                continue

            sent = robot.send_action(decision.executed_action)
            last_robot_action = clone_value(sent)
            timestamp = time.perf_counter()
            packet = DatasetPacket(
                timestamp=timestamp,
                observation=clone_value(observation),
                executed_action=clone_value(decision.executed_action),
                policy_action=clone_value(last_policy_action),
                phase=decision.phase,
                intervention_id=decision.intervention_id,
                trigger_source=controller.trigger_source if decision.intervention_id >= 0 else TriggerSource.NONE,
                training_eligible=decision.training_eligible,
            )

            should_store = mode == CaptureMode.CONTINUOUS or (
                mode == CaptureMode.CORRECTIONS_ONLY
                and decision.phase in {ControlPhase.RECOVERY, ControlPhase.CORRECTION}
            )
            if mode == CaptureMode.EVENT_BUFFER:
                should_store = event_capture.observe(CapturedPacket(timestamp, packet))
            if should_store:
                if origin is None:
                    origin = timestamp
                _store_packet(writer, packet, origin, frames, task)

            loop_index += 1
            if (
                trigger_monitor is not None
                and not event_capture.recording
                and loop_index % fb_cfg.automatic_trigger.interval_frames == 0
            ):
                trigger_monitor.submit(ring.snapshot())
            if (
                mode == CaptureMode.EVENT_BUFFER
                and event_capture.recording
                and event_deadline is not None
                and time.perf_counter() >= event_deadline
                and controller.phase == ControlPhase.AUTONOMOUS
            ):
                finish_episode("automatic_event_capture")
            if mode == CaptureMode.CONTINUOUS and frames and rotator.should_rotate(controller.phase):
                finish_episode("continuous_observation")

            time.sleep(max(1 / cfg.dataset.fps - (time.perf_counter() - loop_start), 0))

        if frames and controller.phase == ControlPhase.AUTONOMOUS:
            finish_episode("session_end")
    finally:
        if trigger_monitor is not None:
            trigger_monitor.close()
        if listener is not None:
            listener.stop()


def _default_sidecar_dir(root: Path, session_id: str) -> Path:
    root = Path(root).resolve()
    return root.parent / "feedback_outputs" / root.name / session_id


def run_feedback(cfg: RecordConfig, *, mock: bool = False) -> Path:
    if cfg.policy is None or cfg.teleop is None:
        raise ValueError("feedback collection requires both a 'policy' and a 'teleop' block")
    fb_cfg = FeedbackCollectionConfig.from_mapping(cfg.feedback)

    robot = build_robot(cfg)
    teleop = build_teleop(cfg)
    writer = build_writer(cfg, robot)
    policy_runner = build_policy_runner(cfg.policy, robot.motor_names, writer.features, mock=mock)

    session_id = fb_cfg.session_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sidecar_dir = fb_cfg.sidecar_dir or _default_sidecar_dir(writer.root, session_id)
    source_policy = fb_cfg.source_policy or (cfg.policy.path or "")
    sidecar = FeedbackSidecarWriter(
        Path(sidecar_dir),
        session_id=session_id,
        source_dataset=Path(writer.root),
        source_policy=source_policy,
        capture_mode=fb_cfg.mode,
        fps=cfg.dataset.fps,
        overwrite=fb_cfg.overwrite_sidecar,
        details={"robot_type": robot.robot_type, "task": cfg.dataset.single_task},
    )

    robot.connect()
    teleop.connect()
    try:
        run_feedback_loop(
            cfg=cfg, fb_cfg=fb_cfg, robot=robot, teleop=teleop,
            writer=writer, policy_runner=policy_runner, sidecar=sidecar,
        )
    finally:
        sidecar.close()
        if writer.total_episodes > 0:
            writer.finalize()
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()
        if cfg.dataset.push_to_hub and writer.total_episodes > 0:
            from .hub import push_dataset

            push_dataset(writer.root, cfg.dataset.repo_id, private=cfg.dataset.private, tags=cfg.dataset.tags)
    return writer.root
