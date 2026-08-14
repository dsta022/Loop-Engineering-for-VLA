"""Orbbec RGB-D backend (Femto Bolt, ...) via pyorbbecsdk.

Ported from the previous lerobot overlay (``cameras/orbbec/camera_orbbec.py``)
with the lerobot base class, decorators, and error types removed. A background
thread keeps the latest aligned (color, depth) pair; ``read()`` returns both
atomically as ``(H, W, 3)`` uint8 RGB and ``(H, W)`` uint16 millimeter depth.
"""

from __future__ import annotations

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import numpy as np

from ..config import OrbbecCameraConfig
from .base import Camera, ColorMode, DeviceNotConnectedError, get_cv2_rotation

logger = logging.getLogger(__name__)

try:  # pragma: no cover - hardware SDK
    import pyorbbecsdk as ob  # type: ignore

    _IMPORT_ERROR: Exception | None = None
except Exception as _err:  # noqa: BLE001
    try:
        import pyorbbecsdk2 as ob  # type: ignore

        _IMPORT_ERROR = None
    except Exception:
        ob = None  # type: ignore[assignment]
        _IMPORT_ERROR = _err


def _require_sdk() -> Any:
    if ob is None:
        raise ImportError(
            "pyorbbecsdk is required for Orbbec cameras. Install Orbbec's Python SDK "
            "(e.g. `pip install pyorbbecsdk2`) and retry."
        ) from _IMPORT_ERROR
    return ob


def _format_name(value: Any) -> str:
    return str(value).split(".")[-1]


class OrbbecCamera(Camera):
    def __init__(self, config: OrbbecCameraConfig):
        super().__init__(config)
        self.serial_number_or_name = config.serial_number_or_name
        self.warmup_s = config.warmup_s
        self.align_depth_to_color = config.align_depth_to_color
        self._rotation = get_cv2_rotation(config.rotation)

        self.device: Any | None = None
        self.pipeline: Any | None = None
        self.pipeline_config: Any | None = None
        self.align_filter: Any | None = None
        self.color_profile: Any | None = None
        self.depth_profile: Any | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock = Lock()
        self.latest_color_frame: np.ndarray | None = None
        self.latest_depth_frame: np.ndarray | None = None
        self.new_frame_event = Event()

        self._depth_out_w = config.output_depth_width
        self._depth_out_h = config.output_depth_height

    def __str__(self) -> str:
        return f"OrbbecCamera({self.serial_number_or_name or 'default'})"

    @property
    def is_connected(self) -> bool:
        return self.pipeline is not None and self.thread is not None and self.thread.is_alive()

    # -- device / profile selection --------------------------------------- #
    def _select_device(self) -> Any:
        sdk = _require_sdk()
        self.context = sdk.Context()
        devices = self.context.query_devices()
        count = devices.get_count() if hasattr(devices, "get_count") else len(devices)
        if count < 1:
            raise DeviceNotConnectedError("No Orbbec devices found.")

        def at(index):
            return devices.get_device_by_index(index) if hasattr(devices, "get_device_by_index") else devices[index]

        if not self.serial_number_or_name:
            if count > 1:
                raise ValueError("Multiple Orbbec cameras connected; set serial_number_or_name.")
            return at(0)
        for index in range(count):
            device = at(index)
            info = device.get_device_info()
            serial = str(getattr(info, "get_serial_number", lambda: None)())
            name = str(getattr(info, "get_name", lambda: None)())
            if str(self.serial_number_or_name) in {serial, name}:
                return device
        raise ValueError(f"No Orbbec camera found for '{self.serial_number_or_name}'.")

    def _select_video_profile(self, sensor_type, width, height, fps, preferred_formats) -> Any:
        profiles = self.pipeline.get_stream_profile_list(sensor_type)
        if width is None and height is None and fps is None:
            return profiles.get_default_video_stream_profile()
        for fmt in preferred_formats:
            try:
                return profiles.get_video_stream_profile(width, height, fmt, fps)
            except Exception:  # noqa: BLE001
                continue
        for index in range(int(profiles.get_count())):
            try:
                profile = profiles.get_stream_profile_by_index(index)
                if (
                    int(profile.get_width()) == width
                    and int(profile.get_height()) == height
                    and int(profile.get_fps()) == fps
                ):
                    return profile
            except Exception:  # noqa: BLE001
                continue
        raise RuntimeError(f"{self} could not find an Orbbec profile for {width}x{height}@{fps}.")

    def _preferred_color_formats(self, sdk) -> list[Any]:
        names = ["RGB", "BGR", "MJPG", "YUYV", "UYVY", "NV12", "NV21", "I420"]
        return [getattr(sdk.OBFormat, n, None) for n in names if getattr(sdk.OBFormat, n, None) is not None]

    def connect(self) -> None:
        sdk = _require_sdk()
        if self.is_connected:
            return
        self.device = self._select_device()
        self.pipeline = sdk.Pipeline(self.device)
        self.pipeline_config = sdk.Config()

        self.color_profile = self._select_video_profile(
            sdk.OBSensorType.COLOR_SENSOR,
            self.config.width,
            self.config.height,
            self.config.fps,
            self._preferred_color_formats(sdk),
        )
        self.pipeline_config.enable_stream(self.color_profile)

        if self.use_depth:
            self.depth_profile = self._select_video_profile(
                sdk.OBSensorType.DEPTH_SENSOR,
                self.config.depth_width,
                self.config.depth_height,
                self.config.depth_fps,
                [sdk.OBFormat.Y16],
            )
            self.pipeline_config.enable_stream(self.depth_profile)
            if hasattr(self.pipeline_config, "set_frame_aggregate_output_mode"):
                self.pipeline_config.set_frame_aggregate_output_mode(
                    sdk.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
                )
            if self.align_depth_to_color:
                self.align_filter = sdk.AlignFilter(align_to_stream=sdk.OBStreamType.COLOR_STREAM)

        try:
            self.pipeline.start(self.pipeline_config)
        except Exception as exc:  # noqa: BLE001
            self.pipeline = None
            self.pipeline_config = None
            raise DeviceNotConnectedError(f"Failed to open {self}: {exc}") from exc

        # Trust the configured resolution (align_depth_to_color makes depth match color).
        self.width = int(self.color_profile.get_width())
        self.height = int(self.color_profile.get_height())
        self.fps = int(self.color_profile.get_fps())

        self._start_read_thread()
        if self.warmup_s > 0:
            deadline = time.time() + self.warmup_s
            while time.time() < deadline:
                try:
                    self.read()
                except Exception:  # noqa: BLE001
                    time.sleep(0.05)
        logger.info("%s connected.", self)

    # -- frame conversion -------------------------------------------------- #
    def _color_to_rgb(self, frame: Any) -> np.ndarray:
        import cv2

        sdk = _require_sdk()
        width = int(frame.get_width())
        height = int(frame.get_height())
        fmt = frame.get_format()
        data = np.asanyarray(frame.get_data())

        def is_fmt(name):
            target = getattr(sdk.OBFormat, name, None)
            return target is not None and fmt == target

        if is_fmt("RGB"):
            return np.resize(data, (height, width, 3)).astype(np.uint8)
        if is_fmt("BGR"):
            return cv2.cvtColor(np.resize(data, (height, width, 3)).astype(np.uint8), cv2.COLOR_BGR2RGB)
        if is_fmt("MJPG"):
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"{self} failed to decode MJPG frame.")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for name, code in (
            ("YUYV", cv2.COLOR_YUV2RGB_YUYV),
            ("UYVY", cv2.COLOR_YUV2RGB_UYVY),
        ):
            if is_fmt(name):
                return cv2.cvtColor(np.resize(data, (height, width, 2)).astype(np.uint8), code)
        for name, code in (
            ("NV12", cv2.COLOR_YUV2RGB_NV12),
            ("NV21", cv2.COLOR_YUV2RGB_NV21),
            ("I420", cv2.COLOR_YUV2RGB_I420),
        ):
            if is_fmt(name):
                return cv2.cvtColor(np.resize(data, (height * 3 // 2, width)).astype(np.uint8), code)
        raise RuntimeError(f"{self} unsupported color format: {fmt}")

    def _depth_to_mm(self, frame: Any) -> np.ndarray:
        scale = float(frame.get_depth_scale())
        width = int(frame.get_width())
        height = int(frame.get_height())
        depth = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape((height, width))
        return np.clip(depth.astype(np.float32) * scale, 0, 65535).astype(np.uint16)

    def _postprocess(self, image: np.ndarray, *, depth: bool = False) -> np.ndarray:
        import cv2

        if self._rotation is not None:
            image = cv2.rotate(image, self._rotation)
        if not depth and self.color_mode == ColorMode.BGR:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def _read_loop(self) -> None:
        failures = 0
        while self.stop_event is not None and not self.stop_event.is_set():
            try:
                frames = self.pipeline.wait_for_frames(10000)
                if frames is None:
                    continue
                if self.use_depth and self.align_filter is not None:
                    frames = self.align_filter.process(frames)
                    if frames is None:
                        continue
                    frames = frames.as_frame_set()

                color_frame = frames.get_color_frame()
                if color_frame is None:
                    continue
                color = self._postprocess(self._color_to_rgb(color_frame))

                depth = None
                if self.use_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame is None:
                        continue
                    depth = self._postprocess(self._depth_to_mm(depth_frame), depth=True)

                with self.frame_lock:
                    self.latest_color_frame = color
                    if depth is not None:
                        self.latest_depth_frame = depth
                self.new_frame_event.set()
                failures = 0
            except Exception as exc:  # noqa: BLE001
                failures += 1
                if failures > 10:
                    logger.error("%s exceeded read failures: %s", self, exc)
                    break

    def _start_read_thread(self) -> None:
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read", daemon=True)
        self.thread.start()

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.stop_event = None

    def read(self) -> tuple[np.ndarray, np.ndarray | None]:
        if self.thread is None or not self.thread.is_alive():
            raise DeviceNotConnectedError(f"{self} read thread is not running.")
        if not self.new_frame_event.wait(timeout=10.0):
            raise TimeoutError(f"{self} timed out waiting for a frame.")
        with self.frame_lock:
            color = self.latest_color_frame
            depth = self.latest_depth_frame if self.use_depth else None
            self.new_frame_event.clear()
        if color is None:
            raise RuntimeError(f"{self} has no color frame available.")
        return self._validate_color(color), self._as_depth_mm(depth)

    def disconnect(self) -> None:
        self._stop_read_thread()
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            finally:
                self.pipeline = None
                self.pipeline_config = None
                self.align_filter = None
        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.new_frame_event.clear()
