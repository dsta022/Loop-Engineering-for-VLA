#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides an Orbbec RGB-D camera backend using pyorbbecsdk."""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2  # type: ignore  # TODO: add type stubs for OpenCV
import numpy as np  # type: ignore  # TODO: add type stubs for numpy
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

try:
    import pyorbbecsdk as ob  # type: ignore  # TODO: add type stubs for pyorbbecsdk

    _PYORBBEC_IMPORT_ERROR: Exception | None = None
except Exception as import_error:
    try:
        import pyorbbecsdk2 as ob  # type: ignore  # TODO: add type stubs for pyorbbecsdk2

        _PYORBBEC_IMPORT_ERROR = None
    except Exception:
        ob = None  # type: ignore[assignment]
        _PYORBBEC_IMPORT_ERROR = import_error

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_orbbec import OrbbecCameraConfig

logger = logging.getLogger(__name__)


def _require_orbbec_sdk() -> Any:
    if ob is None:
        raise ImportError(
            "pyorbbecsdk is required for Orbbec cameras. Install Orbbec's Python SDK package "
            "for your platform, for example `pip install pyorbbecsdk2`, then retry."
        ) from _PYORBBEC_IMPORT_ERROR
    return ob


def _format_name(value: Any) -> str:
    return str(value).split(".")[-1]


def _safe_call(obj: Any, method_name: str) -> Any | None:
    method = getattr(obj, method_name, None)
    if method is None:
        return None
    try:
        return method()
    except Exception:
        return None


class OrbbecCamera(Camera):
    """Manages Orbbec RGB-D cameras via pyorbbecsdk.

    Color frames are exposed through the standard `Camera.read*` APIs. When
    `use_depth=True`, the latest depth frame is also stored as `uint16`
    millimeters in `latest_depth_frame`, matching the RealSense RGB-D contract
    used by the SO follower.
    """

    def __init__(self, config: OrbbecCameraConfig):
        super().__init__(config)

        self.config = config
        self.serial_number_or_name = config.serial_number_or_name
        self.fps = config.fps
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s
        self.align_depth_to_color = config.align_depth_to_color

        self.context: Any | None = None
        self.device: Any | None = None
        self.pipeline: Any | None = None
        self.pipeline_config: Any | None = None
        self.align_filter: Any | None = None
        self.color_profile: Any | None = None
        self.depth_profile: Any | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_color_frame: NDArray[Any] | None = None
        self.latest_depth_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        self.capture_width: int | None = config.width
        self.capture_height: int | None = config.height
        self.depth_capture_width: int | None = config.output_depth_width
        self.depth_capture_height: int | None = config.output_depth_height
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            if self.capture_width is not None and self.capture_height is not None:
                self.capture_width, self.capture_height = self.capture_height, self.capture_width
            if self.depth_capture_width is not None and self.depth_capture_height is not None:
                self.depth_capture_width, self.depth_capture_height = (
                    self.depth_capture_height,
                    self.depth_capture_width,
                )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number_or_name or 'default'})"

    @property
    def is_connected(self) -> bool:
        return self.pipeline is not None and self.pipeline_config is not None

    @staticmethod
    def _device_list_count(device_list: Any) -> int:
        if hasattr(device_list, "get_count"):
            return int(device_list.get_count())
        return len(device_list)

    @staticmethod
    def _device_at(device_list: Any, index: int) -> Any:
        if hasattr(device_list, "get_device_by_index"):
            return device_list.get_device_by_index(index)
        return device_list[index]

    @staticmethod
    def _device_metadata(device: Any) -> dict[str, Any]:
        info = device.get_device_info()
        return {
            "name": _safe_call(info, "get_name"),
            "type": "Orbbec",
            "id": _safe_call(info, "get_serial_number"),
            "uid": _safe_call(info, "get_uid"),
            "pid": _safe_call(info, "get_pid"),
            "connection_type": _safe_call(info, "get_connection_type"),
        }

    @staticmethod
    def _profile_to_dict(profile: Any) -> dict[str, Any]:
        return {
            "format": _format_name(profile.get_format()),
            "width": int(profile.get_width()),
            "height": int(profile.get_height()),
            "fps": int(profile.get_fps()),
        }

    @classmethod
    def _find_default_profile_info(cls, device: Any, sensor_type: Any) -> dict[str, Any] | None:
        sdk = _require_orbbec_sdk()
        try:
            pipeline = sdk.Pipeline(device)
            profile_list = pipeline.get_stream_profile_list(sensor_type)
            profile = profile_list.get_default_video_stream_profile()
            return cls._profile_to_dict(profile)
        except Exception:
            return None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Detect available Orbbec cameras connected to the system."""
        sdk = _require_orbbec_sdk()
        context = sdk.Context()
        devices = context.query_devices()
        found_cameras_info: list[dict[str, Any]] = []

        for index in range(OrbbecCamera._device_list_count(devices)):
            device = OrbbecCamera._device_at(devices, index)
            camera_info = OrbbecCamera._device_metadata(device)
            profiles: dict[str, Any] = {}

            color_profile = OrbbecCamera._find_default_profile_info(device, sdk.OBSensorType.COLOR_SENSOR)
            if color_profile is not None:
                profiles["color"] = color_profile

            depth_profile = OrbbecCamera._find_default_profile_info(device, sdk.OBSensorType.DEPTH_SENSOR)
            if depth_profile is not None:
                profiles["depth"] = depth_profile

            if profiles:
                camera_info["default_stream_profiles"] = profiles

            found_cameras_info.append(camera_info)

        return found_cameras_info

    def _select_device(self) -> Any:
        sdk = _require_orbbec_sdk()
        self.context = sdk.Context()
        devices = self.context.query_devices()
        count = self._device_list_count(devices)
        if count < 1:
            raise ConnectionError("No Orbbec devices found.")

        if self.serial_number_or_name is None:
            if count > 1:
                raise ValueError(
                    "Multiple Orbbec cameras are connected. Set `serial_number_or_name` in the camera config."
                )
            return self._device_at(devices, 0)

        matches = []
        for index in range(count):
            device = self._device_at(devices, index)
            metadata = self._device_metadata(device)
            if self.serial_number_or_name in {str(metadata.get("id")), str(metadata.get("name"))}:
                matches.append(device)

        if not matches:
            available = [self._device_metadata(self._device_at(devices, index)) for index in range(count)]
            raise ValueError(
                f"No Orbbec camera found for '{self.serial_number_or_name}'. Available devices: {available}"
            )

        if len(matches) > 1:
            raise ValueError(
                f"Multiple Orbbec cameras matched '{self.serial_number_or_name}'. Use a unique serial number."
            )

        return matches[0]

    def _ob_format(self, name: str) -> Any | None:
        sdk = _require_orbbec_sdk()
        return getattr(sdk.OBFormat, name, None)

    def _preferred_color_formats(self) -> list[Any]:
        return [
            fmt
            for fmt in (
                self._ob_format("RGB"),
                self._ob_format("BGR"),
                self._ob_format("MJPG"),
                self._ob_format("YUYV"),
                self._ob_format("UYVY"),
                self._ob_format("NV12"),
                self._ob_format("NV21"),
                self._ob_format("I420"),
            )
            if fmt is not None
        ]

    def _select_video_profile(
        self,
        sensor_type: Any,
        *,
        width: int | None,
        height: int | None,
        fps: int | None,
        preferred_formats: list[Any],
    ) -> Any:
        if self.pipeline is None:
            raise RuntimeError(f"{self}: pipeline must be initialized before selecting profiles.")

        profile_list = self.pipeline.get_stream_profile_list(sensor_type)
        if width is None and height is None and fps is None:
            return profile_list.get_default_video_stream_profile()

        requested = f"{width}x{height}@{fps}"
        for stream_format in preferred_formats:
            try:
                return profile_list.get_video_stream_profile(width, height, stream_format, fps)
            except Exception:
                continue

        matching_profiles = []
        for index in range(int(profile_list.get_count())):
            try:
                profile = profile_list.get_stream_profile_by_index(index)
                if not all(hasattr(profile, attr) for attr in ["get_width", "get_height", "get_fps"]):
                    continue
                if (
                    int(profile.get_width()) == width
                    and int(profile.get_height()) == height
                    and int(profile.get_fps()) == fps
                ):
                    matching_profiles.append(profile)
            except Exception:
                continue

        if matching_profiles:
            preferred_names = [_format_name(fmt) for fmt in preferred_formats]
            matching_profiles.sort(
                key=lambda profile: preferred_names.index(_format_name(profile.get_format()))
                if _format_name(profile.get_format()) in preferred_names
                else len(preferred_names)
            )
            return matching_profiles[0]

        raise RuntimeError(f"{self} could not find an Orbbec stream profile for {requested}.")

    def _configure_pipeline(self) -> None:
        sdk = _require_orbbec_sdk()
        self.device = self._select_device()
        self.pipeline = sdk.Pipeline(self.device)
        self.pipeline_config = sdk.Config()

        self.color_profile = self._select_video_profile(
            sdk.OBSensorType.COLOR_SENSOR,
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
            preferred_formats=self._preferred_color_formats(),
        )
        self.pipeline_config.enable_stream(self.color_profile)

        if self.use_depth:
            self.depth_profile = self._select_video_profile(
                sdk.OBSensorType.DEPTH_SENSOR,
                width=self.config.depth_width,
                height=self.config.depth_height,
                fps=self.config.depth_fps,
                preferred_formats=[sdk.OBFormat.Y16],
            )
            self.pipeline_config.enable_stream(self.depth_profile)
            if hasattr(self.pipeline_config, "set_frame_aggregate_output_mode"):
                self.pipeline_config.set_frame_aggregate_output_mode(
                    sdk.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
                )
            if self.align_depth_to_color:
                self.align_filter = sdk.AlignFilter(align_to_stream=sdk.OBStreamType.COLOR_STREAM)

    def _configure_capture_settings(self) -> None:
        if self.color_profile is None:
            raise RuntimeError(f"{self}: color profile must be selected before configuring capture.")

        actual_width = int(self.color_profile.get_width())
        actual_height = int(self.color_profile.get_height())
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            self.width, self.height = actual_height, actual_width
            self.capture_width, self.capture_height = actual_width, actual_height
        else:
            self.width, self.height = actual_width, actual_height
            self.capture_width, self.capture_height = actual_width, actual_height

        self.fps = int(self.color_profile.get_fps())

        if self.use_depth:
            if self.align_depth_to_color:
                self.depth_capture_width = self.capture_width
                self.depth_capture_height = self.capture_height
            elif self.depth_profile is not None:
                self.depth_capture_width = int(self.depth_profile.get_width())
                self.depth_capture_height = int(self.depth_profile.get_height())

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        self._configure_pipeline()

        try:
            self.pipeline.start(self.pipeline_config)
        except Exception as e:
            self.pipeline = None
            self.pipeline_config = None
            raise ConnectionError(
                f"Failed to open {self}. Run `lerobot-find-cameras orbbec` to find available cameras."
            ) from e

        self._configure_capture_settings()
        self._start_read_thread()

        if warmup and self.warmup_s > 0:
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                self.async_read(timeout_ms=self.warmup_s * 1000)
                time.sleep(0.1)
            with self.frame_lock:
                if self.latest_color_frame is None or self.use_depth and self.latest_depth_frame is None:
                    raise ConnectionError(f"{self} failed to capture frames during warmup.")

        logger.info(f"{self} connected.")

    def _read_from_hardware(self) -> Any:
        if self.pipeline is None:
            raise DeviceNotConnectedError(f"{self} pipeline is not initialized")

        frames = self.pipeline.wait_for_frames(10000)
        if frames is None:
            raise RuntimeError(f"{self} read failed: no frames returned.")
        return frames

    def _color_frame_to_rgb(self, frame: Any) -> NDArray[Any]:
        sdk = _require_orbbec_sdk()
        width = int(frame.get_width())
        height = int(frame.get_height())
        color_format = frame.get_format()
        data = np.asanyarray(frame.get_data())

        rgb_format = getattr(sdk.OBFormat, "RGB", None)
        bgr_format = getattr(sdk.OBFormat, "BGR", None)
        mjpg_format = getattr(sdk.OBFormat, "MJPG", None)
        yuyv_format = getattr(sdk.OBFormat, "YUYV", None)
        uyvy_format = getattr(sdk.OBFormat, "UYVY", None)
        nv12_format = getattr(sdk.OBFormat, "NV12", None)
        nv21_format = getattr(sdk.OBFormat, "NV21", None)
        i420_format = getattr(sdk.OBFormat, "I420", None)

        if rgb_format is not None and color_format == rgb_format:
            return np.resize(data, (height, width, 3)).astype(np.uint8)
        if bgr_format is not None and color_format == bgr_format:
            image = np.resize(data, (height, width, 3)).astype(np.uint8)
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if mjpg_format is not None and color_format == mjpg_format:
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"{self} failed to decode MJPG color frame.")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if yuyv_format is not None and color_format == yuyv_format:
            yuyv = np.resize(data, (height, width, 2)).astype(np.uint8)
            return cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUYV)
        if uyvy_format is not None and color_format == uyvy_format:
            uyvy = np.resize(data, (height, width, 2)).astype(np.uint8)
            return cv2.cvtColor(uyvy, cv2.COLOR_YUV2RGB_UYVY)
        if nv12_format is not None and color_format == nv12_format:
            nv12 = np.resize(data, (height * 3 // 2, width)).astype(np.uint8)
            return cv2.cvtColor(nv12, cv2.COLOR_YUV2RGB_NV12)
        if nv21_format is not None and color_format == nv21_format:
            nv21 = np.resize(data, (height * 3 // 2, width)).astype(np.uint8)
            return cv2.cvtColor(nv21, cv2.COLOR_YUV2RGB_NV21)
        if i420_format is not None and color_format == i420_format:
            i420 = np.resize(data, (height * 3 // 2, width)).astype(np.uint8)
            return cv2.cvtColor(i420, cv2.COLOR_YUV2RGB_I420)

        raise RuntimeError(f"{self} unsupported Orbbec color format: {color_format}.")

    def _depth_frame_to_mm(self, frame: Any) -> NDArray[Any]:
        sdk = _require_orbbec_sdk()
        if frame.get_format() != sdk.OBFormat.Y16:
            raise RuntimeError(f"{self} expected Y16 depth, got {frame.get_format()}.")

        width = int(frame.get_width())
        height = int(frame.get_height())
        scale = float(frame.get_depth_scale())
        depth = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape((height, width))
        depth_mm = np.clip(depth.astype(np.float32) * scale, 0, 65535).astype(np.uint16)
        return depth_mm

    def _postprocess_image(self, image: NDArray[Any], *, depth_frame: bool = False) -> NDArray[Any]:
        if depth_frame:
            h, w = image.shape
            expected_width = self.depth_capture_width
            expected_height = self.depth_capture_height
        else:
            h, w, c = image.shape
            if c != 3:
                raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels.")
            expected_width = self.capture_width
            expected_height = self.capture_height

        if expected_width is not None and expected_height is not None:
            if h != expected_height or w != expected_width:
                raise RuntimeError(
                    f"{self} frame width={w} or height={h} do not match configured "
                    f"width={expected_width} or height={expected_height}."
                )

        processed_image = image
        if not depth_frame and self.color_mode == ColorMode.BGR:
            processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    def _read_loop(self) -> None:
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        failure_count = 0
        while not self.stop_event.is_set():
            try:
                frames = self._read_from_hardware()
                if self.use_depth and self.align_filter is not None:
                    frames = self.align_filter.process(frames)
                    if frames is None:
                        continue
                    frames = frames.as_frame_set()

                color_frame = frames.get_color_frame()
                if color_frame is None:
                    raise RuntimeError(f"{self} received a frameset without a color frame.")
                color_image = self._color_frame_to_rgb(color_frame)
                processed_color_frame = self._postprocess_image(color_image)

                if self.use_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame is None:
                        raise RuntimeError(f"{self} received a frameset without a depth frame.")
                    depth_mm = self._depth_frame_to_mm(depth_frame)
                    processed_depth_frame = self._postprocess_image(depth_mm, depth_frame=True)

                capture_time = time.perf_counter()
                with self.frame_lock:
                    self.latest_color_frame = processed_color_frame
                    if self.use_depth:
                        self.latest_depth_frame = processed_depth_frame
                    self.latest_timestamp = capture_time
                self.new_frame_event.set()
                failure_count = 0

            except DeviceNotConnectedError:
                break
            except Exception as e:
                if failure_count <= 10:
                    failure_count += 1
                    logger.warning(f"Error reading frame in background thread for {self}: {e}")
                else:
                    raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from e

    def _start_read_thread(self) -> None:
        self._stop_read_thread()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 0) -> NDArray[Any]:
        if color_mode is not None:
            logger.warning(
                f"{self} read() color_mode parameter is deprecated and will be removed in future versions."
            )
        if timeout_ms:
            logger.warning(
                f"{self} read() timeout_ms parameter is deprecated and will be removed in future versions."
            )

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        self.new_frame_event.clear()
        return self.async_read(timeout_ms=10000)

    @check_if_not_connected
    def read_depth(self, timeout_ms: int = 200) -> NDArray[Any]:
        if not self.use_depth:
            raise RuntimeError(f"Depth stream is not enabled for {self}.")
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        self.new_frame_event.clear()
        _ = self.async_read(timeout_ms=timeout_ms)

        with self.frame_lock:
            depth_map = self.latest_depth_frame

        if depth_map is None:
            raise RuntimeError(f"No depth frame available for {self}.")
        return depth_map

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {self.thread.is_alive()}."
            )

        with self.frame_lock:
            frame = self.latest_color_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        with self.frame_lock:
            frame = self.latest_color_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )

        return frame

    def disconnect(self) -> None:
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.thread is not None:
            self._stop_read_thread()

        if self.pipeline is not None:
            self.pipeline.stop()

        self.pipeline = None
        self.pipeline_config = None
        self.align_filter = None
        self.color_profile = None
        self.depth_profile = None

        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

        logger.info(f"{self} disconnected.")
