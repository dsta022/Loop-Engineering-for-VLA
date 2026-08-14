"""Build the LeRobot v3.0 ``info.json`` features block + frame vector helpers.

Motor observation/action scalars (``{motor}.pos``) are aggregated into single
``observation.state`` / ``action`` vectors; RGB cameras become ``video`` features
(separate mp4) and depth cameras become ``image`` features (PNG struct in parquet
+ depth sidecar), exactly matching what the downstream audit/merge expect.
"""

from __future__ import annotations

from typing import Any

# Feetech encoder name -> the codec_name ffprobe reports (audit checks this).
ENCODER_TO_CODEC = {
    "libsvtav1": "av1",
    "libaom-av1": "av1",
    "av1": "av1",
    "libx264": "h264",
    "h264": "h264",
    "libx265": "hevc",
    "hevc": "hevc",
    "mjpeg": "mjpeg",
}

STATE_KEY = "observation.state"
ACTION_KEY = "action"
IMAGE_PREFIX = "observation.images."


def codec_name_for(vcodec: str) -> str:
    return ENCODER_TO_CODEC.get(str(vcodec).lower(), "av1")


def rgb_key(camera: str) -> str:
    return f"{IMAGE_PREFIX}{camera}"


def depth_key(camera: str) -> str:
    return f"{IMAGE_PREFIX}{camera}_depth"


def motor_pos_names(motor_names: list[str]) -> list[str]:
    return [f"{motor}.pos" for motor in motor_names]


def _scalar_feature(dtype: str = "int64") -> dict[str, Any]:
    return {"dtype": dtype, "shape": [1], "names": None}


def build_features(
    motor_names: list[str],
    cameras: dict[str, Any],
    *,
    fps: int,
    use_video: bool = True,
    vcodec: str = "libsvtav1",
) -> dict[str, dict[str, Any]]:
    """Return the ``features`` mapping for ``meta/info.json``."""
    names = motor_pos_names(motor_names)
    dim = len(names)
    features: dict[str, dict[str, Any]] = {
        ACTION_KEY: {"dtype": "float32", "shape": [dim], "names": names},
        STATE_KEY: {"dtype": "float32", "shape": [dim], "names": names},
    }

    codec = codec_name_for(vcodec)
    for camera, cam_cfg in cameras.items():
        height, width = int(cam_cfg.height), int(cam_cfg.width)
        if use_video:
            features[rgb_key(camera)] = {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": height,
                    "video.width": width,
                    "video.channels": 3,
                    "video.codec": codec,
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": float(fps),
                    "has_audio": False,
                },
            }
        else:
            features[rgb_key(camera)] = {
                "dtype": "image",
                "shape": [height, width, 3],
                "names": ["height", "width", "channels"],
            }
        if getattr(cam_cfg, "use_depth", False):
            d_h = int(cam_cfg.output_depth_height)
            d_w = int(cam_cfg.output_depth_width)
            features[depth_key(camera)] = {
                "dtype": "image",
                "shape": [d_h, d_w, 1],
                "names": ["height", "width", "channels"],
            }

    features["timestamp"] = _scalar_feature("float32")
    features["frame_index"] = _scalar_feature("int64")
    features["episode_index"] = _scalar_feature("int64")
    features["index"] = _scalar_feature("int64")
    features["task_index"] = _scalar_feature("int64")
    return features


def state_vector(obs: dict[str, Any], motor_names: list[str]) -> list[float]:
    return [float(obs[f"{motor}.pos"]) for motor in motor_names]


def action_vector(action: dict[str, Any], motor_names: list[str]) -> list[float]:
    return [float(action[f"{motor}.pos"]) for motor in motor_names]


def video_keys(features: dict[str, dict[str, Any]]) -> list[str]:
    return [key for key, spec in features.items() if spec.get("dtype") == "video"]


def image_keys(features: dict[str, dict[str, Any]]) -> list[str]:
    return [key for key, spec in features.items() if spec.get("dtype") == "image"]
