"""Thread-safe human takeover state machine and hardware handover adapters."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from queue import Empty, SimpleQueue
from threading import Lock
from typing import Any, Mapping, Protocol

from .contracts import ControlPhase, TriggerSource


logger = logging.getLogger(__name__)


class InterventionCommand(str, Enum):
    TOGGLE_CONTROL = "toggle_control"
    BEGIN_CORRECTION = "begin_correction"
    CANCEL = "cancel"
    SAVE_EVENT = "save_event"
    STOP = "stop"


@dataclass(frozen=True)
class InterventionWindow:
    intervention_id: int
    trigger_source: TriggerSource
    recovery_start_frame: int
    recovery_end_frame: int
    correction_start_frame: int
    correction_end_frame: int
    outcome: str = "completed"


@dataclass(frozen=True)
class ActionDecision:
    phase: ControlPhase
    intervention_id: int
    executed_action: Any
    training_eligible: bool
    hold_position: bool = False


class InterventionController:
    """Validate control handoffs and keep frame ranges explicit.

    Frame indices are episode-local indices in the dataset being written.  The
    controller never talks to hardware; hardware side effects are delegated to
    a HandoverAdapter so the state machine remains unit-testable.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._phase = ControlPhase.AUTONOMOUS
        self._next_intervention_id = 0
        self._active_id = -1
        self._trigger_source = TriggerSource.NONE
        self._recovery_start: int | None = None
        self._correction_start: int | None = None

    @property
    def phase(self) -> ControlPhase:
        with self._lock:
            return self._phase

    @property
    def active_intervention_id(self) -> int:
        with self._lock:
            return self._active_id

    @property
    def trigger_source(self) -> TriggerSource:
        with self._lock:
            return self._trigger_source

    def request_takeover(self, source: TriggerSource | str = TriggerSource.MANUAL) -> int:
        with self._lock:
            if self._phase != ControlPhase.AUTONOMOUS:
                raise RuntimeError(f"takeover requires autonomous phase, got {self._phase.value}")
            self._phase = ControlPhase.SYNCING
            self._active_id = self._next_intervention_id
            self._next_intervention_id += 1
            self._trigger_source = TriggerSource(source)
            self._recovery_start = None
            self._correction_start = None
            return self._active_id

    def confirm_synchronized(self, frame_index: int) -> None:
        with self._lock:
            if self._phase != ControlPhase.SYNCING:
                raise RuntimeError(f"synchronization confirmation is invalid in {self._phase.value}")
            if frame_index < 0:
                raise ValueError("recovery start frame must be non-negative")
            self._phase = ControlPhase.RECOVERY
            self._recovery_start = frame_index

    def begin_correction(self, frame_index: int) -> None:
        with self._lock:
            if self._phase != ControlPhase.RECOVERY:
                raise RuntimeError(f"correction requires recovery phase, got {self._phase.value}")
            if self._recovery_start is None or frame_index <= self._recovery_start:
                raise ValueError("record at least one recovery frame before starting correction")
            self._phase = ControlPhase.CORRECTION
            self._correction_start = frame_index

    def complete(self, next_frame_index: int) -> InterventionWindow:
        with self._lock:
            if self._phase != ControlPhase.CORRECTION:
                raise RuntimeError(f"completion requires correction phase, got {self._phase.value}")
            if self._recovery_start is None or self._correction_start is None:
                raise RuntimeError("intervention frame boundaries are incomplete")
            if next_frame_index <= self._correction_start:
                raise ValueError("record at least one correction frame before completing intervention")
            result = InterventionWindow(
                intervention_id=self._active_id,
                trigger_source=self._trigger_source,
                recovery_start_frame=self._recovery_start,
                recovery_end_frame=self._correction_start - 1,
                correction_start_frame=self._correction_start,
                correction_end_frame=next_frame_index - 1,
            )
            self._reset_locked()
            return result

    def cancel(self) -> None:
        with self._lock:
            if self._phase == ControlPhase.AUTONOMOUS:
                return
            self._reset_locked()

    def reset_episode(self) -> None:
        with self._lock:
            if self._phase != ControlPhase.AUTONOMOUS:
                raise RuntimeError("cannot rotate an episode while an intervention is active")
            self._next_intervention_id = 0

    def select_action(self, policy_action: Any, human_action: Any = None) -> ActionDecision:
        with self._lock:
            if self._phase == ControlPhase.AUTONOMOUS:
                if policy_action is None:
                    raise ValueError("autonomous control requires a policy action")
                return ActionDecision(self._phase, -1, policy_action, False)
            if self._phase in {ControlPhase.SYNCING, ControlPhase.PAUSED}:
                return ActionDecision(self._phase, -1, None, False, hold_position=True)
            if human_action is None:
                raise ValueError(f"{self._phase.value} control requires a human action")
            return ActionDecision(self._phase, self._active_id, human_action, True)

    def _reset_locked(self) -> None:
        self._phase = ControlPhase.AUTONOMOUS
        self._active_id = -1
        self._trigger_source = TriggerSource.NONE
        self._recovery_start = None
        self._correction_start = None


class InterventionEventQueue:
    def __init__(self) -> None:
        self._queue: SimpleQueue[InterventionCommand] = SimpleQueue()

    def push(self, command: InterventionCommand | str) -> None:
        self._queue.put(InterventionCommand(command))

    def pop(self) -> InterventionCommand | None:
        try:
            return self._queue.get_nowait()
        except Empty:
            return None

    def drain(self) -> list[InterventionCommand]:
        commands: list[InterventionCommand] = []
        while (command := self.pop()) is not None:
            commands.append(command)
        return commands


@dataclass(frozen=True)
class InterventionKeyConfig:
    toggle_control: str = "i"
    begin_correction: str = "c"
    cancel: str = "x"
    save_event: str = "h"
    stop: str = "esc"

    def validate(self) -> None:
        values = [
            self.toggle_control,
            self.begin_correction,
            self.cancel,
            self.save_event,
            self.stop,
        ]
        if any(not value.strip() for value in values):
            raise ValueError("intervention keys must not be empty")
        normalized = [value.strip().casefold() for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("intervention keys must be unique")


def create_intervention_key_listener(
    events: InterventionEventQueue, config: InterventionKeyConfig
) -> Any:
    """Create a small independent keyboard listener.

    The import stays optional so headless collection can use a pedal or another
    component that writes to InterventionEventQueue instead.
    """

    config.validate()
    try:
        from pynput import keyboard
    except ImportError as exc:
        raise RuntimeError("pynput is required for keyboard intervention controls") from exc

    key_map = {
        config.toggle_control.casefold(): InterventionCommand.TOGGLE_CONTROL,
        config.begin_correction.casefold(): InterventionCommand.BEGIN_CORRECTION,
        config.cancel.casefold(): InterventionCommand.CANCEL,
        config.save_event.casefold(): InterventionCommand.SAVE_EVENT,
        config.stop.casefold(): InterventionCommand.STOP,
    }
    pressed: set[str] = set()

    def key_name(key: Any) -> str:
        char = getattr(key, "char", None)
        if char:
            return str(char).casefold()
        name = getattr(key, "name", None)
        return str(name or key).replace("Key.", "").casefold()

    def on_press(key: Any) -> None:
        name = key_name(key)
        if name in pressed:
            return
        pressed.add(name)
        command = key_map.get(name)
        if command is not None:
            events.push(command)

    def on_release(key: Any) -> None:
        pressed.discard(key_name(key))

    try:
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("failed to start keyboard intervention controls") from exc
    return listener


class HandoverAdapter(Protocol):
    def prepare_takeover(self, robot: Any, teleop: Any, last_action: Any) -> None: ...

    def begin_human_control(self, robot: Any, teleop: Any) -> None: ...

    def end_human_control(self, robot: Any, teleop: Any) -> None: ...

    def resume_autonomous(self, robot: Any, teleop: Any) -> None: ...


@dataclass
class CapabilityHandoverAdapter:
    """Best-effort adapter for actuated leader devices.

    Safe handover requires a teleoperator that can accept positional feedback.
    Passive devices are rejected by default instead of moving the follower to an
    unknown pose.  Set require_feedback_alignment=False only after hardware-
    specific validation supplies another safe synchronization procedure.
    """

    synchronization_seconds: float = 0.5
    synchronization_steps: int = 20
    require_feedback_alignment: bool = True

    def prepare_takeover(self, robot: Any, teleop: Any, last_action: Any) -> None:
        send_feedback = getattr(teleop, "send_feedback", None)
        feedback_features = getattr(teleop, "feedback_features", None)
        supports_feedback = callable(send_feedback) and bool(feedback_features)
        if not supports_feedback:
            if self.require_feedback_alignment:
                raise RuntimeError(
                    "teleoperator cannot be aligned automatically; install a hardware-specific "
                    "HandoverAdapter before enabling human takeover"
                )
            logger.warning("Proceeding without verified teleoperator alignment")
            return
        if last_action is None:
            raise RuntimeError("cannot align teleoperator without the robot's last commanded action")
        target = _numeric_mapping(last_action)
        if not target:
            if self.require_feedback_alignment:
                raise RuntimeError("robot action cannot be converted into a teleoperator alignment target")
            logger.warning("Proceeding without a numeric teleoperator alignment target")
            return
        current = _numeric_mapping(teleop.get_action()) if hasattr(teleop, "get_action") else {}
        if set(current) != set(target):
            if self.require_feedback_alignment:
                raise RuntimeError(
                    "teleoperator and robot action keys differ; configure a hardware-specific "
                    "HandoverAdapter or a verified rename map"
                )
            logger.warning("Proceeding without verified action-key alignment")
            return
        steps = max(1, int(self.synchronization_steps))
        delay = max(0.0, float(self.synchronization_seconds)) / steps
        for step in range(1, steps + 1):
            ratio = step / steps
            interpolated = {
                key: current[key] + (target[key] - current[key]) * ratio for key in target
            }
            send_feedback(interpolated)
            if delay:
                time.sleep(delay)

    def begin_human_control(self, robot: Any, teleop: Any) -> None:
        disable_torque = getattr(teleop, "disable_torque", None)
        if callable(disable_torque):
            disable_torque()

    def end_human_control(self, robot: Any, teleop: Any) -> None:
        enable_torque = getattr(teleop, "enable_torque", None)
        if callable(enable_torque):
            enable_torque()

    def resume_autonomous(self, robot: Any, teleop: Any) -> None:
        disable_torque = getattr(teleop, "disable_torque", None)
        if callable(disable_torque):
            disable_torque()


def _numeric_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        try:
            scalar = float(item)
        except (TypeError, ValueError):
            return {}
        if not math.isfinite(scalar):
            return {}
        result[str(key)] = scalar
    return result
