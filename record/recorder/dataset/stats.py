"""Per-episode feature statistics in LeRobot's shape.

Stats are stored per episode in ``meta/episodes/...parquet`` as
``stats/<feature>/<stat>`` columns and aggregated into ``meta/stats.json`` by
the toolkit's :func:`utils.dataset_metadata.build_stats_json` (weighted mean +
pooled std). Vector features keep per-dimension arrays; scalar features keep
length-1 arrays; image/video features keep per-channel ``[C, 1, 1]`` arrays.
"""

from __future__ import annotations

import numpy as np

QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def vector_stats(matrix: np.ndarray, count: int) -> dict[str, object]:
    """Stats for a ``[n, dim]`` (or ``[n]``) numeric feature -> per-dim arrays."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    stats: dict[str, object] = {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(count)],
    }
    for name, q in QUANTILES.items():
        stats[name] = np.quantile(arr, q, axis=0).tolist()
    return stats


def scalar_stats(values: np.ndarray, count: int) -> dict[str, object]:
    """Stats for a scalar feature -> length-1 arrays (matches lerobot layout)."""
    return vector_stats(np.asarray(values, dtype=np.float64), count)


class ImageStatAccumulator:
    """Running per-channel mean/std/min/max over normalized image frames."""

    def __init__(self, channels: int, *, divisor: float = 255.0):
        self.channels = channels
        self.divisor = float(divisor)
        self._sum = np.zeros(channels, dtype=np.float64)
        self._sumsq = np.zeros(channels, dtype=np.float64)
        self._min = np.full(channels, np.inf, dtype=np.float64)
        self._max = np.full(channels, -np.inf, dtype=np.float64)
        self._pixels = 0
        self._frames = 0

    def update(self, frame: np.ndarray) -> None:
        arr = np.asarray(frame, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        arr = arr / self.divisor
        flat = arr.reshape(-1, arr.shape[-1])
        self._sum += flat.sum(axis=0)
        self._sumsq += np.square(flat).sum(axis=0)
        self._min = np.minimum(self._min, flat.min(axis=0))
        self._max = np.maximum(self._max, flat.max(axis=0))
        self._pixels += flat.shape[0]
        self._frames += 1

    def result(self) -> dict[str, object]:
        if self._pixels == 0:
            zeros = [[[0.0]] for _ in range(self.channels)]
            return {"min": zeros, "max": zeros, "mean": zeros, "std": zeros, "count": [0]}
        mean = self._sum / self._pixels
        variance = np.maximum(self._sumsq / self._pixels - np.square(mean), 0.0)
        std = np.sqrt(variance)

        def nested(values: np.ndarray) -> list:
            return [[[float(v)]] for v in values]

        return {
            "min": nested(self._min),
            "max": nested(self._max),
            "mean": nested(mean),
            "std": nested(std),
            "count": [int(self._frames)],
        }
