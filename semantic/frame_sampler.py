"""Decode evenly spaced RGB frames from one episode's video time-segment.

Episodes share a single video file (many episodes per chunk file), so a segment
is addressed by ``[from_timestamp, to_timestamp)`` inside that file. Frames are
returned as an ``(N, H, W, 3)`` uint8 array in RGB order, ready to embed.

Only ffmpeg is required; no model runtime is imported here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


class FrameSamplingError(RuntimeError):
    """Raised when a segment cannot be decoded into the requested frames."""


def _even_stride(segment_frames: int, requested: int) -> int:
    """Frame stride that yields about ``requested`` frames from the segment."""
    if segment_frames <= requested:
        return 1
    return max(1, segment_frames // requested)


def sample_segment_frames(
    video_path: Path,
    ffmpeg_path: str,
    from_timestamp: float,
    to_timestamp: float,
    *,
    num_frames: int,
    size: int,
    segment_frames: int | None = None,
) -> np.ndarray:
    """Return ``(N, size, size, 3)`` uint8 RGB frames spanning the segment.

    ``segment_frames`` is the episode length in frames when known; it lets the
    sampler pick a stride in a single ffmpeg call instead of probing. The output
    is square-scaled to ``size`` for a vision encoder; aspect ratio is not
    preserved, which is standard for CLIP-style preprocessing.
    """
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    if size < 1:
        raise ValueError("size must be >= 1")
    duration = float(to_timestamp) - float(from_timestamp)
    if not duration > 0.0:
        raise FrameSamplingError(
            f"non-positive segment duration for {video_path.name}: "
            f"[{from_timestamp}, {to_timestamp})"
        )

    stride = _even_stride(int(segment_frames), num_frames) if segment_frames else 1
    # Input seeking (`-ss`/`-to` before `-i`) resets the frame counter to 0 at
    # the segment start, so `mod(n, stride)` selects within the segment only.
    selector = f"not(mod(n\\,{stride}))"
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-ss",
        f"{float(from_timestamp):.6f}",
        "-to",
        f"{float(to_timestamp):.6f}",
        "-i",
        str(video_path),
        "-vf",
        f"select='{selector}',scale={size}:{size}",
        "-vsync",
        "0",
        "-frames:v",
        str(num_frames),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", "replace").strip().splitlines()
        tail = message[-1] if message else "unknown ffmpeg error"
        raise FrameSamplingError(f"ffmpeg failed for {video_path.name}: {tail}")

    frame_bytes = size * size * 3
    total = len(result.stdout)
    if total == 0 or total % frame_bytes != 0:
        raise FrameSamplingError(
            f"ffmpeg returned {total} bytes for {video_path.name}, "
            f"not a whole multiple of {frame_bytes}"
        )
    count = total // frame_bytes
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(count, size, size, 3)
    return np.ascontiguousarray(frames)
