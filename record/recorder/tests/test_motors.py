import tempfile
import unittest
from pathlib import Path

from recorder.motors.calibration import MotorCalibration, load_calibration, save_calibration
from recorder.motors.feetech_bus import FeetechBus, Motor, MotorNormMode
from recorder.motors.tables import decode_sign_magnitude, encode_sign_magnitude


class SignMagnitudeTests(unittest.TestCase):
    def test_roundtrip(self):
        for bit in (11, 15):
            for value in (-2000, -1, 0, 1, 2000):
                if abs(value) <= (1 << bit) - 1:
                    self.assertEqual(decode_sign_magnitude(encode_sign_magnitude(value, bit), bit), value)

    def test_positive_unchanged(self):
        self.assertEqual(encode_sign_magnitude(1234, 15), 1234)
        self.assertEqual(decode_sign_magnitude(1234, 15), 1234)


def _bus(mode):
    motors = {"j": Motor(id=1, model="sts3215", norm_mode=mode)}
    cal = {"j": MotorCalibration(id=1, drive_mode=0, homing_offset=0, range_min=0, range_max=4095)}
    return FeetechBus("MOCK", motors, cal)


class NormalizationTests(unittest.TestCase):
    def test_range_m100_100_midpoint(self):
        bus = _bus(MotorNormMode.RANGE_M100_100)
        self.assertAlmostEqual(bus._normalize("j", 2047), (2047 / 4095) * 200 - 100, places=4)

    def test_roundtrip_all_modes(self):
        for mode in (MotorNormMode.RANGE_M100_100, MotorNormMode.RANGE_0_100, MotorNormMode.DEGREES):
            bus = _bus(mode)
            for raw in (0, 1000, 2047, 3000, 4095):
                norm = bus._normalize("j", raw)
                back = bus._unnormalize("j", norm)
                self.assertLessEqual(abs(back - raw), 2, f"{mode} raw={raw} -> {norm} -> {back}")

    def test_read_write_helpers_use_calibration(self):
        bus = _bus(MotorNormMode.RANGE_0_100)
        # 0..4095 -> 0..100
        self.assertAlmostEqual(bus._normalize("j", 0), 0.0, places=4)
        self.assertAlmostEqual(bus._normalize("j", 4095), 100.0, places=4)


class CalibrationIoTests(unittest.TestCase):
    def test_save_load_roundtrip(self):
        cal = {
            "shoulder_pan": MotorCalibration(1, 0, 42, 900, 3200),
            "gripper": MotorCalibration(6, 0, -13, 1000, 3000),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            save_calibration(path, cal)
            loaded = load_calibration(path)
        self.assertEqual(loaded, cal)


if __name__ == "__main__":
    unittest.main()
