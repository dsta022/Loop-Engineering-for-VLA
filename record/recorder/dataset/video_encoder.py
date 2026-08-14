"""Encode raw RGB frames to an mp4 whose codec matches ``info.json``.

Streams frames straight into ffmpeg over stdin (no intermediate PNGs) and reuses
the toolkit's own :func:`utils.media_utils.choose_encoder`, so the produced codec
matches what the quality audit's ffprobe check expects.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np

from utils.media_utils import choose_encoder, ffmpeg_available

from .features import codec_name_for


class VideoEncodeError(RuntimeError):
    pass


def _build_ffmpeg_cmd(
    ffmpeg: str,
    out_path: Path,
    *,
    fps: int,
    width: int,
    height: int,
    vcodec: str,
    encoder_threads: int | None,
) -> list[str]:
    encoder, extra = choose_encoder(ffmpeg, codec_name_for(vcodec))
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{int(width)}x{int(height)}",
        "-r",
        str(int(fps)),
        "-i",
        "-",
        "-an",
        "-c:v",
        encoder,
        *extra,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if encoder_threads:
        cmd += ["-threads", str(int(encoder_threads))]
    cmd.append(str(out_path))
    return cmd


class VideoStream:
    """Incrementally encode ``(H, W, 3)`` uint8 RGB frames into one mp4."""

    def __init__(
        self,
        out_path: Path,
        *,
        fps: int,
        width: int,
        height: int,
        vcodec: str = "libsvtav1",
        encoder_threads: int | None = None,
        ffmpeg_path: str | None = None,
    ):
        self.out_path = Path(out_path)
        self.width = int(width)
        self.height = int(height)
        ffmpeg = ffmpeg_path or ffmpeg_available()
        if ffmpeg is None:
            raise VideoEncodeError("ffmpeg not found on PATH or in the environment.")
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = _build_ffmpeg_cmd(
            ffmpeg, self.out_path, fps=fps, width=width, height=height,
            vcodec=vcodec, encoder_threads=encoder_threads,
        )
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self._frames = 0

    def write(self, frame: np.ndarray) -> None:
        array = np.ascontiguousarray(frame, dtype=np.uint8)
        if array.shape != (self.height, self.width, 3):
            raise VideoEncodeError(f"frame shape {array.shape} != {(self.height, self.width, 3)}")
        try:
            self._proc.stdin.write(array.tobytes())
        except BrokenPipeError as exc:
            _, stderr = self._proc.communicate()
            raise VideoEncodeError(
                f"ffmpeg pipe broke after {self._frames} frames: "
                f"{stderr.decode('utf-8', 'replace')[:500]}"
            ) from exc
        self._frames += 1

    def close(self) -> int:
        """Flush and finalize; returns the frame count written."""
        if self._proc.stdin:
            self._proc.stdin.close()
        _, stderr = self._proc.communicate()
        if self._proc.returncode != 0:
            raise VideoEncodeError(f"ffmpeg failed: {stderr.decode('utf-8', 'replace')[:800]}")
        return self._frames

    def abort(self) -> None:
        """Kill the encoder and remove the partial file (for re-recorded episodes)."""
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.kill()
            self._proc.communicate()
        except Exception:  # noqa: BLE001
            pass
        self.out_path.unlink(missing_ok=True)


def encode_video(
    frames: Iterable[np.ndarray],
    out_path: Path,
    *,
    fps: int,
    width: int,
    height: int,
    vcodec: str = "libsvtav1",
    encoder_threads: int | None = None,
    ffmpeg_path: str | None = None,
) -> int:
    """Batch-encode an iterable of RGB frames to ``out_path`` (mp4)."""
    stream = VideoStream(
        out_path, fps=fps, width=width, height=height,
        vcodec=vcodec, encoder_threads=encoder_threads, ffmpeg_path=ffmpeg_path,
    )
    for frame in frames:
        stream.write(frame)
    return stream.close()
