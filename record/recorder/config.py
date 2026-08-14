"""Configuration dataclasses + YAML/CLI loader (replaces draccus ``RecordConfig``).

The stock lerobot entry points parse configuration with ``draccus`` (a heavy
dataclass-CLI framework). This module reproduces just the slice the recorder
needs, with zero third-party dependency beyond PyYAML:

* ``--config_path=<file.yaml>`` selects a YAML config (same flag as before).
* ``--a.b.c=value`` dotted flags override individual fields after the YAML load,
  so existing command lines such as ``--dataset.num_episodes=5`` keep working.
* values are parsed with ``yaml.safe_load`` so ``5`` -> int, ``false`` -> bool.

The dataclasses intentionally mirror the field names in the existing YAML
configs (``robot``/``teleop``/``dataset``/``policy``/``feedback`` + camera blocks)
so no user config has to change.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH_FLAG = "--config_path"

# Backends whose ``use_depth`` flag can be honoured (mirrors camera_profiles).
DEPTH_CAPABLE_TYPES = {"orbbec", "intelrealsense", "mock"}
FOLLOWER_TYPES = {"so100_follower", "so101_follower", "so_follower"}
LEADER_TYPES = {"so100_leader", "so101_leader", "so_leader"}


class ConfigError(ValueError):
    """Raised when a recording configuration is invalid."""


# --------------------------------------------------------------------------- #
# Camera configs
# --------------------------------------------------------------------------- #
@dataclass
class BaseCameraConfig:
    type: str
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"
    rotation: int = 0

    # Depth defaults: overridden by depth-capable backends.
    use_depth: bool = False

    @property
    def output_depth_width(self) -> int | None:
        return self.width

    @property
    def output_depth_height(self) -> int | None:
        return self.height


@dataclass
class OpenCVCameraConfig(BaseCameraConfig):
    index_or_path: Any = None

    def __post_init__(self) -> None:
        if self.index_or_path is None:
            raise ConfigError("opencv camera requires 'index_or_path'")
        self.use_depth = False


@dataclass
class RealSenseCameraConfig(BaseCameraConfig):
    serial_number_or_name: str | None = None

    def __post_init__(self) -> None:
        if not self.serial_number_or_name:
            raise ConfigError("intelrealsense camera requires 'serial_number_or_name'")
        # RealSense depth always shares the color stream resolution/fps.


@dataclass
class OrbbecCameraConfig(BaseCameraConfig):
    serial_number_or_name: str | None = None
    depth_fps: int | None = None
    depth_width: int | None = None
    depth_height: int | None = None
    align_depth_to_color: bool = True
    warmup_s: int = 1

    def __post_init__(self) -> None:
        if not self.serial_number_or_name:
            raise ConfigError("orbbec camera requires 'serial_number_or_name'")
        if self.use_depth and self.align_depth_to_color and (self.width is None or self.height is None):
            raise ConfigError(
                "orbbec align_depth_to_color=true requires explicit RGB width/height "
                "so dataset depth feature shapes are known before connecting."
            )

    @property
    def output_depth_width(self) -> int | None:
        if self.align_depth_to_color:
            return self.width
        return self.depth_width or self.width

    @property
    def output_depth_height(self) -> int | None:
        if self.align_depth_to_color:
            return self.height
        return self.depth_height or self.height


@dataclass
class MockCameraConfig(BaseCameraConfig):
    """Synthetic camera for offline testing (no hardware, no SDK)."""

    seed: int = 0

    def __post_init__(self) -> None:
        if self.width is None or self.height is None or self.fps is None:
            raise ConfigError("mock camera requires fps/width/height")


_CAMERA_CLASSES: dict[str, type[BaseCameraConfig]] = {
    "opencv": OpenCVCameraConfig,
    "intelrealsense": RealSenseCameraConfig,
    "orbbec": OrbbecCameraConfig,
    "mock": MockCameraConfig,
}


def _build_dataclass(cls: type, payload: dict[str, Any], *, context: str) -> Any:
    """Instantiate ``cls`` from a mapping, ignoring unknown keys with a warning."""
    if not isinstance(payload, dict):
        raise ConfigError(f"{context} must be a mapping, got {type(payload).__name__}")
    field_names = {f.name for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, value in payload.items():
        if key in field_names:
            kwargs[key] = value
        else:
            logger.warning("ignoring unknown %s field %r", context, key)
    return cls(**kwargs)


def build_camera_config(name: str, payload: dict[str, Any]) -> BaseCameraConfig:
    if not isinstance(payload, dict) or not payload.get("type"):
        raise ConfigError(f"camera '{name}' must be a mapping with a 'type' field")
    cam_type = str(payload["type"])
    cls = _CAMERA_CLASSES.get(cam_type)
    if cls is None:
        raise ConfigError(
            f"camera '{name}' has unsupported type '{cam_type}'; "
            f"supported: {sorted(_CAMERA_CLASSES)}"
        )
    return _build_dataclass(cls, payload, context=f"camera '{name}'")


# --------------------------------------------------------------------------- #
# Robot / teleop / dataset / feedback configs
# --------------------------------------------------------------------------- #
@dataclass
class RobotConfig:
    type: str
    port: str = ""
    id: str | None = None
    use_degrees: bool = False
    max_relative_target: float | None = None
    disable_torque_on_disconnect: bool = True
    mock: bool = False
    cameras: dict[str, BaseCameraConfig] = field(default_factory=dict)

    @property
    def name(self) -> str:
        # Dataset robot_type label; keep the concrete model when provided.
        return self.type


@dataclass
class TeleopConfig:
    type: str
    port: str = ""
    id: str | None = None
    # Leader normalization must match the follower for correct teleop mapping.
    # ``None`` means "inherit the follower's use_degrees" (resolved by the entrypoint).
    use_degrees: bool | None = None
    mock: bool = False


@dataclass
class PolicyConfig:
    path: str | None = None
    device: str = "cuda"
    # Extra keys are preserved for the policy runtime adapter.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    repo_id: str
    single_task: str
    root: str | None = None
    fps: int = 30
    episode_time_s: float = 60.0
    reset_time_s: float = 60.0
    num_episodes: int = 50
    video: bool = True
    push_to_hub: bool = False
    private: bool = False
    tags: list[str] | None = None
    vcodec: str = "libsvtav1"
    # Kept for CLI/back-compat; the framework-free encoder ignores streaming.
    streaming_encoding: bool = False
    encoder_threads: int | None = None
    num_image_writer_threads_per_camera: int = 4

    def __post_init__(self) -> None:
        if not self.single_task:
            raise ConfigError("dataset.single_task is required")
        if not self.repo_id:
            raise ConfigError("dataset.repo_id is required")


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetConfig
    teleop: TeleopConfig | None = None
    policy: PolicyConfig | None = None
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    play_sounds: bool = False
    resume: bool = False
    # Populated only for the feedback recorder (opaque to base recording).
    feedback: dict[str, Any] | None = None

    @property
    def fps(self) -> int:
        return self.dataset.fps


# --------------------------------------------------------------------------- #
# YAML + CLI loading
# --------------------------------------------------------------------------- #
def _coerce_value(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def extract_config_path(argv: list[str]) -> tuple[list[str], str | None]:
    """Return (remaining_argv, config_path). Accepts ``--config_path=x`` or ``--config_path x``."""
    remaining: list[str] = []
    value: str | None = None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == CONFIG_PATH_FLAG:
            if index + 1 >= len(argv):
                raise ConfigError(f"{CONFIG_PATH_FLAG} requires a value")
            value = argv[index + 1]
            index += 2
            continue
        if arg.startswith(f"{CONFIG_PATH_FLAG}="):
            value = arg.split("=", 1)[1]
            index += 1
            continue
        remaining.append(arg)
        index += 1
    return remaining, value


def apply_overrides(payload: dict[str, Any], argv: list[str]) -> dict[str, Any]:
    """Apply ``--a.b.c=value`` dotted overrides onto a nested mapping."""
    result = dict(payload)
    for arg in argv:
        if not arg.startswith("--"):
            raise ConfigError(f"unexpected argument: {arg!r}")
        body = arg[2:]
        if "=" not in body:
            raise ConfigError(f"override {arg!r} must be of the form --key.path=value")
        dotted, raw = body.split("=", 1)
        keys = dotted.split(".")
        node = result
        for key in keys[:-1]:
            child = node.get(key)
            if not isinstance(child, dict):
                child = {}
                node[key] = child
            node = child
        node[keys[-1]] = _coerce_value(raw)
    return result


def _build_cameras(payload: Any) -> dict[str, BaseCameraConfig]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigError("robot.cameras must be a mapping of camera name -> config")
    return {name: build_camera_config(name, block) for name, block in payload.items()}


def build_record_config(payload: dict[str, Any]) -> RecordConfig:
    if not isinstance(payload, dict):
        raise ConfigError("record config must be a mapping")
    if "robot" not in payload:
        raise ConfigError("record config must define 'robot'")

    robot_payload = dict(payload["robot"])
    cameras = _build_cameras(robot_payload.pop("cameras", None))
    robot = _build_dataclass(RobotConfig, robot_payload, context="robot")
    robot.cameras = cameras

    # ``dataset`` is optional so teleoperate-only configs (robot + teleop + fps)
    # load. A placeholder is synthesized for that path; recording always ships a
    # real dataset block.
    if payload.get("dataset") is not None:
        dataset = _build_dataclass(DatasetConfig, payload["dataset"], context="dataset")
        if "fps" in payload and "fps" not in payload["dataset"]:
            dataset.fps = int(payload["fps"])
    else:
        dataset = DatasetConfig(
            repo_id="_teleoperate_only", single_task="_teleoperate_only", fps=int(payload.get("fps", 30))
        )

    teleop = None
    if payload.get("teleop") is not None:
        teleop = _build_dataclass(TeleopConfig, payload["teleop"], context="teleop")

    policy = None
    if payload.get("policy") is not None:
        policy_payload = dict(payload["policy"])
        known = {f.name for f in dataclasses.fields(PolicyConfig)}
        extra = {k: policy_payload.pop(k) for k in list(policy_payload) if k not in known}
        policy = _build_dataclass(PolicyConfig, policy_payload, context="policy")
        policy.extra = extra

    top = {
        k: payload[k]
        for k in ("display_data", "display_ip", "display_port", "play_sounds", "resume")
        if k in payload
    }
    cfg = RecordConfig(
        robot=robot,
        dataset=dataset,
        teleop=teleop,
        policy=policy,
        feedback=payload.get("feedback"),
        **top,
    )
    return cfg


def load_record_config(argv: list[str]) -> RecordConfig:
    """Parse argv (``--config_path`` + dotted overrides) into a :class:`RecordConfig`."""
    remaining, config_path = extract_config_path(argv)
    if config_path is None:
        raise ConfigError(f"{CONFIG_PATH_FLAG} is required")
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    payload = apply_overrides(payload, remaining)
    return build_record_config(payload)


# --------------------------------------------------------------------------- #
# Capability guards
# --------------------------------------------------------------------------- #
def has_depth_cameras(robot: RobotConfig) -> list[str]:
    return [name for name, cam in robot.cameras.items() if getattr(cam, "use_depth", False)]


def enforce_rgb_only(robot: RobotConfig) -> None:
    """Reject depth-enabled cameras on the RGB-only recording path."""
    depth = has_depth_cameras(robot)
    if depth:
        raise ConfigError(
            f"cameras {depth} enable depth, but this recorder is RGB-only. "
            "Use the rgbd_record path for depth capture, or disable use_depth."
        )
    for name, cam in robot.cameras.items():
        if cam.type in DEPTH_CAPABLE_TYPES and cam.type not in {"mock"} and getattr(cam, "use_depth", False):
            raise ConfigError(f"camera '{name}' enables depth on the RGB-only path")
