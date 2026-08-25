"""Memory-bounded event capture and optional asynchronous trigger plugins."""

from __future__ import annotations

import importlib
import json
import math
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .contracts import ControlPhase, TriggerSource


@dataclass(frozen=True)
class CapturedPacket:
    timestamp: float
    payload: Any
    size_bytes: int | None = None

    def estimated_size(self) -> int:
        return int(self.size_bytes) if self.size_bytes is not None else estimate_payload_bytes(self.payload)


def estimate_payload_bytes(value: Any, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if value is None:
        return 0
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return max(0, nbytes)
    if isinstance(value, (bytes, bytearray, memoryview, str)):
        return len(value)
    if isinstance(value, Mapping):
        return sum(estimate_payload_bytes(key, seen) + estimate_payload_bytes(item, seen) for key, item in value.items())
    if isinstance(value, Sequence):
        return sum(estimate_payload_bytes(item, seen) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        # Dataclass packets (e.g. the feedback recorder's DatasetPacket) carry
        # camera frames in their fields; falling through to the 64-byte default
        # would let the buffer blow far past its configured memory cap.
        return sum(estimate_payload_bytes(getattr(value, f.name), seen) for f in fields(value))
    return 64


class BoundedCaptureBuffer:
    def __init__(self, *, max_seconds: float, fps: float, max_memory_mb: float) -> None:
        if max_seconds <= 0 or fps <= 0 or max_memory_mb <= 0:
            raise ValueError("capture buffer limits must be positive")
        self.max_frames = max(1, int(math.ceil(max_seconds * fps)))
        self.max_bytes = max(1, int(max_memory_mb * 1024 * 1024))
        self._packets: deque[tuple[CapturedPacket, int]] = deque()
        self._bytes = 0

    @property
    def memory_bytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._packets)

    def append(self, packet: CapturedPacket) -> None:
        if not math.isfinite(float(packet.timestamp)) or packet.timestamp < 0:
            raise ValueError("captured packet timestamp must be finite and non-negative")
        size = max(0, packet.estimated_size())
        self._packets.append((packet, size))
        self._bytes += size
        while len(self._packets) > self.max_frames or self._bytes > self.max_bytes:
            _, removed = self._packets.popleft()
            self._bytes -= removed

    def snapshot(self) -> tuple[CapturedPacket, ...]:
        return tuple(packet for packet, _ in self._packets)

    def flush(self) -> tuple[CapturedPacket, ...]:
        packets = self.snapshot()
        self.clear()
        return packets

    def clear(self) -> None:
        self._packets.clear()
        self._bytes = 0


@dataclass(frozen=True)
class TriggerSignal:
    triggered: bool
    source: TriggerSource | str = TriggerSource.NONE
    score: float | None = None
    reason: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        source = TriggerSource(self.source)
        if self.triggered and source == TriggerSource.NONE:
            raise ValueError("a triggered signal must identify its source")
        if self.score is not None and not math.isfinite(float(self.score)):
            raise ValueError("trigger score must be finite")
        json.dumps(dict(self.details), ensure_ascii=False)


@runtime_checkable
class WindowTriggerEvaluator(Protocol):
    name: str

    def evaluate_window(
        self, packets: Sequence[CapturedPacket]
    ) -> TriggerSignal | Mapping[str, Any]: ...


def normalize_trigger_signal(value: TriggerSignal | Mapping[str, Any]) -> TriggerSignal:
    if isinstance(value, TriggerSignal):
        signal = value
    elif isinstance(value, Mapping):
        signal = TriggerSignal(
            triggered=bool(value.get("triggered", False)),
            source=str(value.get("source") or TriggerSource.SEMANTIC.value),
            score=None if value.get("score") is None else float(value["score"]),
            reason=str(value.get("reason") or ""),
            details=dict(value.get("details") or {}),
        )
    else:
        raise TypeError("window trigger evaluator must return TriggerSignal or a mapping")
    signal.validate()
    return signal


def load_window_trigger(
    spec: str, config: Mapping[str, Any] | None = None
) -> WindowTriggerEvaluator:
    if ":" not in spec:
        raise ValueError("window trigger plugin must use module:attribute syntax")
    module_name, attribute_name = spec.split(":", 1)
    target = getattr(importlib.import_module(module_name), attribute_name)
    kwargs = dict(config or {})
    if isinstance(target, type):
        target = target(**kwargs)
    elif callable(target) and not hasattr(target, "evaluate_window"):
        target = target(kwargs)
    if not hasattr(target, "evaluate_window"):
        raise TypeError(f"window trigger plugin does not define evaluate_window(): {spec}")
    return target


def load_trigger_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("window trigger config must contain one JSON object")
    return value


class AsyncTriggerMonitor:
    """Run a slow trigger plugin off the robot control thread."""

    def __init__(self, evaluator: WindowTriggerEvaluator) -> None:
        self.evaluator = evaluator
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="capture-trigger")
        self._future: Future[Any] | None = None
        self._closed = False

    def submit(self, packets: Sequence[CapturedPacket]) -> bool:
        if self._closed:
            raise RuntimeError("trigger monitor is closed")
        if self._future is not None and not self._future.done():
            return False
        snapshot = tuple(packets)
        if not snapshot:
            return False
        self._future = self._executor.submit(self.evaluator.evaluate_window, snapshot)
        return True

    def poll(self) -> TriggerSignal | None:
        if self._future is None or not self._future.done():
            return None
        future = self._future
        self._future = None
        try:
            return normalize_trigger_signal(future.result())
        except Exception as exc:
            return TriggerSignal(
                triggered=False,
                source=TriggerSource.SEMANTIC,
                reason="trigger_evaluator_error",
                details={"error": f"{type(exc).__name__}:{str(exc)[:500]}"},
            )

    def close(self) -> None:
        if self._closed:
            return
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._closed = True

    def __enter__(self) -> "AsyncTriggerMonitor":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class EventCaptureController:
    """Manage pre-event buffering and a single live event window."""

    def __init__(self, buffer: BoundedCaptureBuffer) -> None:
        self.buffer = buffer
        self.recording = False
        self.active_signal: TriggerSignal | None = None

    def observe(self, packet: CapturedPacket) -> bool:
        if self.recording:
            return True
        self.buffer.append(packet)
        return False

    def start(self, signal: TriggerSignal) -> tuple[CapturedPacket, ...]:
        signal.validate()
        if not signal.triggered:
            raise ValueError("cannot start event capture from a non-triggered signal")
        if self.recording:
            return ()
        self.recording = True
        self.active_signal = signal
        return self.buffer.flush()

    def stop(self) -> TriggerSignal | None:
        signal = self.active_signal
        self.recording = False
        self.active_signal = None
        return signal


class EpisodeRotator:
    """Time-based episode rotation that never forces a human-control boundary."""

    def __init__(self, duration_seconds: float, now: float | None = None) -> None:
        if duration_seconds <= 0:
            raise ValueError("rotation duration must be positive")
        self.duration_seconds = float(duration_seconds)
        self.started_at = time.perf_counter() if now is None else float(now)

    def should_rotate(self, phase: ControlPhase | str, now: float | None = None) -> bool:
        current = time.perf_counter() if now is None else float(now)
        return (
            ControlPhase(phase) == ControlPhase.AUTONOMOUS
            and current - self.started_at >= self.duration_seconds
        )

    def reset(self, now: float | None = None) -> None:
        self.started_at = time.perf_counter() if now is None else float(now)
