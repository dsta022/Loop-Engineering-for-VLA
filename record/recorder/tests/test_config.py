import unittest

from recorder import config


def base_payload():
    return {
        "robot": {
            "type": "so100_follower", "id": "r", "port": "COM5", "use_degrees": True,
            "cameras": {
                "front": {"type": "opencv", "index_or_path": 0, "fps": 30, "width": 64, "height": 48},
            },
        },
        "teleop": {"type": "so100_leader", "port": "COM6"},
        "dataset": {"repo_id": "u/d", "single_task": "t", "fps": 30, "num_episodes": 3},
    }


class ConfigTests(unittest.TestCase):
    def test_build_basic(self):
        cfg = config.build_record_config(base_payload())
        self.assertEqual(cfg.robot.type, "so100_follower")
        self.assertEqual(cfg.dataset.num_episodes, 3)
        self.assertIn("front", cfg.robot.cameras)
        self.assertFalse(cfg.robot.cameras["front"].use_depth)
        self.assertEqual(cfg.teleop.type, "so100_leader")

    def test_dotted_overrides(self):
        payload = config.apply_overrides(base_payload(), ["--dataset.num_episodes=7", "--robot.use_degrees=false"])
        cfg = config.build_record_config(payload)
        self.assertEqual(cfg.dataset.num_episodes, 7)
        self.assertFalse(cfg.robot.use_degrees)

    def test_extract_config_path(self):
        remaining, path = config.extract_config_path(["prog", "--config_path=x.yaml", "--dataset.fps=15"])
        self.assertEqual(path, "x.yaml")
        self.assertEqual(remaining, ["prog", "--dataset.fps=15"])

    def test_depth_camera_builds(self):
        payload = base_payload()
        payload["robot"]["cameras"]["front"] = {
            "type": "orbbec", "serial_number_or_name": "S", "fps": 30, "width": 64, "height": 48,
            "use_depth": True, "depth_fps": 30, "depth_width": 32, "depth_height": 24,
            "align_depth_to_color": False,
        }
        cfg = config.build_record_config(payload)
        cam = cfg.robot.cameras["front"]
        self.assertTrue(cam.use_depth)
        self.assertEqual((cam.output_depth_height, cam.output_depth_width), (24, 32))

    def test_enforce_rgb_only_rejects_depth(self):
        payload = base_payload()
        payload["robot"]["cameras"]["front"] = {
            "type": "intelrealsense", "serial_number_or_name": "S", "fps": 30,
            "width": 64, "height": 48, "use_depth": True,
        }
        cfg = config.build_record_config(payload)
        with self.assertRaises(config.ConfigError):
            config.enforce_rgb_only(cfg.robot)

    def test_unknown_camera_type_rejected(self):
        payload = base_payload()
        payload["robot"]["cameras"]["front"] = {"type": "nope", "fps": 30, "width": 64, "height": 48}
        with self.assertRaises(config.ConfigError):
            config.build_record_config(payload)


if __name__ == "__main__":
    unittest.main()
