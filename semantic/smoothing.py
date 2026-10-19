"""Temporal smoothing for per-frame alignment signals.

Raw per-frame CLIP similarity is spiky: a single frame with an odd crop or
motion blur swings the score, which turns into spurious regressions once the
audit's progress-shape check looks at frame-to-frame drops. These filters
denoise along the frame axis (axis 0) while preserving genuine sustained
transitions, so real regressions survive and single-frame spikes do not.

Every filter is shape-preserving and works on a 1-D series or a 2-D
``(frames, channels)`` array smoothed independently per channel. None of them
import a model runtime.
"""

from __future__ import annotations

import numpy as np


SMOOTHING_METHODS = ("none", "median", "moving_average", "ema")


def _as_2d(values: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array[:, None], True
    if array.ndim == 2:
        return array, False
    raise ValueError("smoothing expects a 1-D or 2-D array")


def median_filter(values: np.ndarray, window: int) -> np.ndarray:
    """Sliding median with edge clamping; best at removing single-frame spikes."""
    array, was_1d = _as_2d(values)
    window = max(1, int(window))
    if window == 1 or array.shape[0] < 2:
        return array[:, 0] if was_1d else array
    half = window // 2
    padded = np.pad(array, ((half, half), (0, 0)), mode="edge")
    out = np.empty_like(array)
    for i in range(array.shape[0]):
        out[i] = np.median(padded[i : i + 2 * half + 1], axis=0)
    return out[:, 0] if was_1d else out


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered mean with edge clamping; smooth but softens sharp transitions."""
    array, was_1d = _as_2d(values)
    window = max(1, int(window))
    if window == 1 or array.shape[0] < 2:
        return array[:, 0] if was_1d else array
    half = window // 2
    padded = np.pad(array, ((half, half), (0, 0)), mode="edge")
    kernel = np.ones(2 * half + 1, dtype=np.float64) / (2 * half + 1)
    out = np.vstack(
        [np.convolve(padded[:, c], kernel, mode="valid") for c in range(array.shape[1])]
    ).T
    return out[:, 0] if was_1d else out


def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    """Causal exponential moving average; introduces lag, so a regression the
    filter sees is delayed rather than erased."""
    array, was_1d = _as_2d(values)
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("ema alpha must be within (0, 1]")
    out = np.empty_like(array)
    out[0] = array[0]
    for i in range(1, array.shape[0]):
        out[i] = alpha * array[i] + (1.0 - alpha) * out[i - 1]
    return out[:, 0] if was_1d else out


def smooth(
    values: np.ndarray,
    method: str,
    *,
    window: int = 3,
    alpha: float = 0.5,
) -> np.ndarray:
    """Dispatch to a named smoothing filter. ``method='none'`` is a no-op."""
    if method == "none":
        return np.asarray(values, dtype=np.float64)
    if method == "median":
        return median_filter(values, window)
    if method == "moving_average":
        return moving_average(values, window)
    if method == "ema":
        return ema(values, alpha)
    raise ValueError(f"unknown smoothing method: {method!r}; expected one of {SMOOTHING_METHODS}")
