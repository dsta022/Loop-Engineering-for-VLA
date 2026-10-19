"""Tests for temporal smoothing of alignment signals."""

from __future__ import annotations

import unittest

import numpy as np

from semantic.smoothing import ema, median_filter, moving_average, smooth


class MedianFilterTest(unittest.TestCase):
    def test_removes_single_frame_spike(self):
        series = np.array([0.2, 0.2, 0.9, 0.2, 0.2])
        out = median_filter(series, window=3)
        self.assertLess(out[2], 0.5)  # spike pulled down toward neighbours

    def test_preserves_sustained_transition(self):
        series = np.array([0.1, 0.1, 0.1, 0.8, 0.8, 0.8])
        out = median_filter(series, window=3)
        self.assertLess(out[1], 0.3)
        self.assertGreater(out[4], 0.6)

    def test_shape_preserved_1d_and_2d(self):
        self.assertEqual(median_filter(np.zeros(5), 3).shape, (5,))
        self.assertEqual(median_filter(np.zeros((5, 2)), 3).shape, (5, 2))

    def test_window_one_is_noop(self):
        series = np.array([0.2, 0.9, 0.1])
        np.testing.assert_allclose(median_filter(series, 1), series)


class MovingAverageTest(unittest.TestCase):
    def test_reduces_variance(self):
        rng = np.linspace(0, 1, 12) + np.array([0.3, -0.3] * 6)
        out = moving_average(rng, window=3)
        self.assertLess(float(np.var(out)), float(np.var(rng)))

    def test_endpoints_finite(self):
        out = moving_average(np.array([0.0, 1.0, 0.0, 1.0]), window=3)
        self.assertTrue(np.all(np.isfinite(out)))


class EmaTest(unittest.TestCase):
    def test_lags_toward_target(self):
        series = np.array([0.0, 1.0, 1.0, 1.0])
        out = ema(series, alpha=0.5)
        self.assertTrue(out[1] < out[2] < out[3])
        self.assertLessEqual(out[-1], 1.0)

    def test_alpha_one_is_identity(self):
        series = np.array([0.1, 0.7, 0.2])
        np.testing.assert_allclose(ema(series, alpha=1.0), series)

    def test_invalid_alpha_raises(self):
        with self.assertRaises(ValueError):
            ema(np.array([0.0, 1.0]), alpha=0.0)


class SmoothDispatchTest(unittest.TestCase):
    def test_none_is_noop(self):
        series = np.array([0.2, 0.9, 0.1])
        np.testing.assert_allclose(smooth(series, "none"), series)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            smooth(np.array([0.0, 1.0]), "bogus")

    def test_all_methods_preserve_length(self):
        series = np.array([0.1, 0.4, 0.2, 0.9, 0.3, 0.5])
        for method in ("none", "median", "moving_average", "ema"):
            self.assertEqual(smooth(series, method, window=3, alpha=0.5).shape, (6,))


if __name__ == "__main__":
    unittest.main()
