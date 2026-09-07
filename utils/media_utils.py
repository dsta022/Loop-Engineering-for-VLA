#!/usr/bin/env python3
"""Shared PNG, depth, and ffmpeg helpers for LeRobot dataset tools."""

from __future__ import annotations

import binascii
import json
import struct
import subprocess
import zlib
from pathlib import Path

import numpy as np

from utils.dataset_io import find_env_executable


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
VIDEO_SAMPLE_WIDTH = 160
VIDEO_SAMPLE_HEIGHT = 120


def png_channels(color_type: int) -> int:
    if color_type == 0:
        return 1
    if color_type == 2:
        return 3
    if color_type == 4:
        return 2
    if color_type == 6:
        return 4
    raise ValueError(f"unsupported_png_color_type:{color_type}")


def validate_png_full(path: Path, expected_width: int, expected_height: int) -> None:
    idat = bytearray()
    with path.open("rb") as handle:
        if handle.read(8) != PNG_SIGNATURE:
            raise ValueError("bad_png_signature")
        saw_ihdr = False
        saw_iend = False
        while True:
            length_raw = handle.read(4)
            if len(length_raw) == 0:
                break
            chunk_type = handle.read(4)
            if len(length_raw) != 4 or len(chunk_type) != 4:
                raise ValueError("truncated_chunk_header")
            length = struct.unpack(">I", length_raw)[0]
            data = handle.read(length)
            crc_raw = handle.read(4)
            if len(data) != length or len(crc_raw) != 4:
                raise ValueError("truncated_chunk_data")
            expected_crc = struct.unpack(">I", crc_raw)[0]
            actual_crc = binascii.crc32(chunk_type)
            actual_crc = binascii.crc32(data, actual_crc) & 0xFFFFFFFF
            if actual_crc != expected_crc:
                raise ValueError(f"bad_crc:{chunk_type.decode('ascii', errors='replace')}")
            if chunk_type == b"IHDR":
                saw_ihdr = True
                width, height, *_ = struct.unpack(">IIBBBBB", data)
                if width != expected_width or height != expected_height:
                    raise ValueError(f"bad_dimensions:{width}x{height}")
            elif chunk_type == b"IDAT":
                idat.extend(data)
            elif chunk_type == b"IEND":
                saw_iend = True
                break
        if not saw_ihdr:
            raise ValueError("missing_ihdr")
        if not saw_iend:
            raise ValueError("missing_iend")
        if not idat:
            raise ValueError("missing_idat")
        zlib.decompress(bytes(idat))


def validate_png(path: Path, expected_shape: list[int] | None, mode: str) -> None:
    if mode == "none":
        return
    if expected_shape is None or len(expected_shape) < 2:
        raise ValueError("missing_expected_shape")
    expected_height = int(expected_shape[0])
    expected_width = int(expected_shape[1])
    if mode == "header":
        with path.open("rb") as handle:
            if handle.read(8) != PNG_SIGNATURE:
                raise ValueError("bad_png_signature")
            length = struct.unpack(">I", handle.read(4))[0]
            chunk_type = handle.read(4)
            data = handle.read(length)
            if chunk_type != b"IHDR" or length != 13:
                raise ValueError("missing_ihdr")
            width, height, *_ = struct.unpack(">IIBBBBB", data)
            if width != expected_width or height != expected_height:
                raise ValueError(f"bad_dimensions:{width}x{height}")
        return
    if mode == "full":
        validate_png_full(path, expected_width, expected_height)
        return
    raise ValueError(f"unsupported_png_check:{mode}")


def unfilter_png_scanlines(raw: bytes, width: int, height: int, bit_depth: int, color_type: int) -> bytes:
    channels = png_channels(color_type)
    if bit_depth not in {8, 16}:
        raise ValueError(f"unsupported_png_bit_depth:{bit_depth}")
    bytes_per_sample = bit_depth // 8
    bpp = channels * bytes_per_sample
    row_bytes = width * bpp
    expected = height * (row_bytes + 1)
    if len(raw) < expected:
        raise ValueError(f"png_decompressed_too_short:{len(raw)}<{expected}")

    rows: list[bytes] = []
    prev = bytearray(row_bytes)
    offset = 0
    for _ in range(height):
        filter_type = raw[offset]
        offset += 1
        cur = bytearray(raw[offset : offset + row_bytes])
        offset += row_bytes
        for i in range(row_bytes):
            left = cur[i - bpp] if i >= bpp else 0
            up = prev[i]
            up_left = prev[i - bpp] if i >= bpp else 0
            if filter_type == 0:
                pass
            elif filter_type == 1:
                cur[i] = (cur[i] + left) & 0xFF
            elif filter_type == 2:
                cur[i] = (cur[i] + up) & 0xFF
            elif filter_type == 3:
                cur[i] = (cur[i] + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                p = left + up - up_left
                pa = abs(p - left)
                pb = abs(p - up)
                pc = abs(p - up_left)
                predictor = left if pa <= pb and pa <= pc else up if pb <= pc else up_left
                cur[i] = (cur[i] + predictor) & 0xFF
            else:
                raise ValueError(f"unsupported_png_filter:{filter_type}")
        rows.append(bytes(cur))
        prev = cur
    return b"".join(rows)


def read_png_array(path: Path) -> np.ndarray:
    """Decode a PNG to an (H, W, C) array.

    OpenCV decodes in C when it is installed; the pure-Python reference
    decoder below is the fallback and the source of the structured error
    messages, so a file OpenCV rejects still fails with a specific reason.
    """
    fast = read_png_with_cv2(path)
    if fast is not None:
        return fast
    return read_png_pure_python(path)


def read_png_with_cv2(path: Path) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:
        return None
    try:
        payload = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if payload.size == 0:
        return None
    arr = cv2.imdecode(payload, cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
    return arr


def read_png_pure_python(path: Path) -> np.ndarray:
    """Standard-library PNG decoder used when OpenCV is absent or refuses a file."""
    with path.open("rb") as handle:
        if handle.read(8) != PNG_SIGNATURE:
            raise ValueError("bad_png_signature")
        width = height = bit_depth = color_type = None
        idat = bytearray()
        while True:
            length_raw = handle.read(4)
            if not length_raw:
                break
            chunk_type = handle.read(4)
            if len(length_raw) != 4 or len(chunk_type) != 4:
                raise ValueError("truncated_png_chunk_header")
            length = struct.unpack(">I", length_raw)[0]
            data = handle.read(length)
            crc = handle.read(4)
            if len(data) != length or len(crc) != 4:
                raise ValueError("truncated_png_chunk_data")
            if chunk_type == b"IHDR":
                width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", data)
            elif chunk_type == b"IDAT":
                idat.extend(data)
            elif chunk_type == b"IEND":
                break
        if width is None or height is None or bit_depth is None or color_type is None:
            raise ValueError("missing_png_ihdr")
        raw = zlib.decompress(bytes(idat))
        scanlines = unfilter_png_scanlines(raw, int(width), int(height), int(bit_depth), int(color_type))
        channels = png_channels(int(color_type))
        if bit_depth == 16:
            arr = np.frombuffer(scanlines, dtype=">u2").reshape((height, width, channels))
        else:
            arr = np.frombuffer(scanlines, dtype=np.uint8).reshape((height, width, channels))
        return arr


def depth_content_metrics(path: Path) -> dict[str, float | bool]:
    arr = read_png_array(path)
    arr2 = arr[:, :, 0] if arr.ndim == 3 and arr.shape[2] == 1 else arr.reshape(-1)
    invalid_ratio = float(np.mean(arr2 == 0))
    nonzero = arr2[arr2 != 0]
    constant = bool(arr2.size > 0 and np.min(arr2) == np.max(arr2))
    nonzero_constant = bool(nonzero.size > 0 and np.min(nonzero) == np.max(nonzero))
    return {
        "invalid_ratio": invalid_ratio,
        "constant": constant,
        "nonzero_constant": nonzero_constant,
    }


def ffmpeg_available() -> str | None:
    return find_env_executable("ffmpeg")


def ffprobe_available() -> str | None:
    return find_env_executable("ffprobe")


def ffprobe_video(
    path: Path,
    ffprobe_path: str,
    expected_shape: list[int] | None,
    expected_codec: str | None,
) -> tuple[bool, str]:
    result = subprocess.run(
        [
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height",
            "-of",
            "json",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False, result.stderr.strip() or "ffprobe_failed"
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False, "ffprobe_bad_json"
    streams = payload.get("streams", [])
    if not streams:
        return False, "ffprobe_no_video_stream"
    stream = streams[0]
    width = stream.get("width")
    height = stream.get("height")
    codec = stream.get("codec_name")
    if width is None or height is None:
        return False, "ffprobe_missing_dimensions"
    if expected_shape is not None and len(expected_shape) >= 2:
        expected_height = int(expected_shape[0])
        expected_width = int(expected_shape[1])
        if int(width) != expected_width or int(height) != expected_height:
            return False, f"ffprobe_bad_dimensions:{width}x{height}!={expected_width}x{expected_height}"
    if expected_codec and codec and str(codec).lower() != str(expected_codec).lower():
        return False, f"ffprobe_bad_codec:{codec}!={expected_codec}"
    return True, ""


def ffmpeg_decode_video(path: Path, ffmpeg_path: str) -> tuple[bool, str]:
    result = subprocess.run(
        [
            ffmpeg_path,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "ffmpeg_decode_failed"
        return False, message.replace("\r", " ").replace("\n", " ")[:500]
    return True, ""


def sample_video_file(
    video_path: Path,
    ffmpeg_path: str,
    sample_frames: int,
    stride: int,
    start: float | None = None,
    duration: float | None = None,
) -> tuple[np.ndarray | None, str]:
    """Sample strided frames, optionally restricted to a [start, start+duration) segment.

    Variable frame-rate output (``-fps_mode vfr``, or ``-vsync vfr`` on ffmpeg
    older than 5.1) is required: without it ffmpeg pads the select filter's
    output back to the input frame rate by duplicating the first surviving
    frame, so the ``-frames:v`` cap would return identical copies of frame 0
    instead of strided samples spanning the video. Recent ffmpeg builds have
    removed ``-vsync`` entirely, so the modern flag is tried first.
    """
    stride = max(int(stride), 1)
    selector = f"eq(n\\,0)+not(mod(n\\,{stride}))"
    prefix = [ffmpeg_path, "-hide_banner", "-nostdin", "-v", "error"]
    if start is not None and start > 0:
        prefix += ["-ss", format_seconds(start)]
    if duration is not None:
        if duration <= 0:
            return None, "non_positive_video_segment_duration"
        prefix += ["-t", format_seconds(duration)]
    prefix += [
        "-i",
        str(video_path),
        "-vf",
        f"select='{selector}',scale={VIDEO_SAMPLE_WIDTH}:{VIDEO_SAMPLE_HEIGHT}",
    ]
    suffix = [
        "-frames:v",
        str(sample_frames),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    result = None
    for vfr_args in (["-fps_mode", "vfr"], ["-vsync", "vfr"]):
        result = subprocess.run(
            [*prefix, *vfr_args, *suffix], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
        )
        if result.returncode == 0:
            break
        stderr = result.stderr.decode("utf-8", errors="replace")
        if "Unrecognized option" not in stderr or vfr_args[0].lstrip("-") not in stderr:
            break
    assert result is not None
    if result.returncode != 0:
        msg = result.stderr.decode("utf-8", errors="replace").strip() or "ffmpeg_file_sample_failed"
        return None, msg.replace("\r", " ").replace("\n", " ")[:500]
    frame_size = VIDEO_SAMPLE_WIDTH * VIDEO_SAMPLE_HEIGHT * 3
    if len(result.stdout) < frame_size:
        return None, "ffmpeg_file_sample_no_frames"
    usable = len(result.stdout) - (len(result.stdout) % frame_size)
    frames = np.frombuffer(result.stdout[:usable], dtype=np.uint8)
    frames = frames.reshape((-1, VIDEO_SAMPLE_HEIGHT, VIDEO_SAMPLE_WIDTH, 3))
    return frames, ""


def video_content_metrics(frames: np.ndarray) -> dict[str, float]:
    flat = frames.reshape((frames.shape[0], -1, 3)).astype(np.float32)
    means = flat.mean(axis=(1, 2))
    stds = flat.std(axis=(1, 2))
    black_ratio = float(np.mean((means < 5.0) & (stds < 3.0)))
    white_ratio = float(np.mean((means > 250.0) & (stds < 3.0)))
    constant_ratio = float(np.mean(stds < 2.0))
    freeze_ratio = 0.0
    if frames.shape[0] > 1:
        diffs = np.mean(
            np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16)),
            axis=(1, 2, 3),
        )
        freeze_ratio = float(np.mean(diffs < 1.0))
    return {
        "black_ratio": black_ratio,
        "white_ratio": white_ratio,
        "constant_ratio": constant_ratio,
        "freeze_ratio": freeze_ratio,
        "sampled_frames": float(frames.shape[0]),
    }


def format_seconds(value: float) -> str:
    if value < 0:
        value = 0.0
    text = f"{value:.9f}".rstrip("0").rstrip(".")
    return text or "0"


def available_video_encoders(ffmpeg_path: str) -> set[str]:
    result = subprocess.run(
        [ffmpeg_path, "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    payload = result.stdout + "\n" + result.stderr
    encoders: set[str] = set()
    for line in payload.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1])
    return encoders


def choose_encoder(ffmpeg_path: str, codec: str | None) -> tuple[str, list[str]]:
    codec_text = (codec or "").lower()
    encoders = available_video_encoders(ffmpeg_path)
    candidates: list[tuple[str, list[str]]]
    if codec_text in {"av1"}:
        candidates = [
            ("libsvtav1", ["-preset", "8"]),
            ("libaom-av1", ["-cpu-used", "6", "-b:v", "0"]),
        ]
    elif codec_text in {"h264", "avc1"}:
        candidates = [("libx264", ["-preset", "veryfast"])]
    elif codec_text in {"hevc", "h265"}:
        candidates = [("libx265", ["-preset", "fast"])]
    elif codec_text in {"mjpeg"}:
        candidates = [("mjpeg", [])]
    else:
        candidates = [("libx264", ["-preset", "veryfast"])]

    for encoder, args in candidates:
        if encoder in encoders:
            return encoder, args
    raise RuntimeError(f"no ffmpeg encoder found for codec {codec or 'unknown'}; tried {[c[0] for c in candidates]}")


def run_ffmpeg_trim(
    ffmpeg_path: str,
    src_video: Path,
    dst_video: Path,
    start: float,
    duration: float,
    mode: str,
    codec: str | None,
    crf: int,
) -> None:
    if duration <= 0:
        raise ValueError(f"non-positive video segment duration: {duration}")
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    base = [
        ffmpeg_path,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-ss",
        format_seconds(start),
        "-t",
        format_seconds(duration),
        "-i",
        str(src_video),
        "-map",
        "0:v:0",
        "-an",
    ]
    if mode == "trim-copy":
        codec_args = ["-c:v", "copy", "-avoid_negative_ts", "make_zero", "-reset_timestamps", "1"]
    elif mode == "trim-reencode":
        encoder, extra = choose_encoder(ffmpeg_path, codec)
        codec_args = ["-c:v", encoder, "-crf", str(crf), *extra, "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    else:
        raise ValueError(f"unsupported video mode: {mode}")
    result = subprocess.run(
        [*base, *codec_args, str(dst_video)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout).replace("\r", " ").replace("\n", " ")[:1000]
        raise RuntimeError(f"ffmpeg trim failed: {src_video} -> {dst_video}: {message}")
