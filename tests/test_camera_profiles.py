from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "camera_profiles", REPO_ROOT / "record" / "camera_profiles.py"
)
camera_profiles = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(camera_profiles)


PROFILES_PAYLOAD = {
    "default_profile": "orbbec_rig",
    "profiles": {
        "rgb_rig": {
            "head": {"type": "opencv", "index_or_path": "/dev/video2", "fps": 30, "width": 640, "height": 480},
            "wrist": {"type": "opencv", "index_or_path": "/dev/video0", "fps": 30, "width": 640, "height": 480},
        },
        "orbbec_rig": {
            "front": {
                "type": "orbbec",
                "serial_number_or_name": "SERIAL",
                "fps": 30,
                "width": 1280,
                "height": 720,
                "use_depth": True,
                "depth_fps": 30,
                "depth_width": 640,
                "depth_height": 576,
                "align_depth_to_color": True,
            },
        },
        "realsense_rig": {
            "front": {
                "type": "intelrealsense",
                "serial_number_or_name": "D435_SERIAL",
                "fps": 30,
                "width": 848,
                "height": 480,
                "use_depth": True,
            },
        },
    },
}

RECORD_CONFIG = {
    "robot": {
        "type": "so100_follower",
        "port": "/dev/ttyACM0",
        "cameras": {"front": {"type": "opencv", "index_or_path": "/dev/video9"}},
    },
    "dataset": {"repo_id": "user/dataset", "fps": 30, "single_task": "task"},
}


class CameraProfileTests(unittest.TestCase):
    def _write_inputs(self, root: Path) -> tuple[Path, Path]:
        profiles_path = root / "camera_profiles.yaml"
        profiles_path.write_text(yaml.safe_dump(PROFILES_PAYLOAD), encoding="utf-8")
        config_path = root / "record.yaml"
        config_path.write_text(yaml.safe_dump(RECORD_CONFIG), encoding="utf-8")
        return profiles_path, config_path

    def test_rewrite_argv_merges_selected_profile_into_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, config_path = self._write_inputs(root)
            argv = [
                "record.py",
                f"--config_path={config_path}",
                "--camera_profile=realsense_rig",
            ]

            rewritten = camera_profiles.rewrite_argv_with_camera_profile(
                argv, allow_depth=True, search_root=root
            )

            self.assertEqual(len(rewritten), 2)
            self.assertNotIn("--camera_profile=realsense_rig", rewritten)
            merged_path = Path(rewritten[1].split("=", 1)[1])
            merged = yaml.safe_load(merged_path.read_text(encoding="utf-8"))
            self.assertEqual(
                merged["robot"]["cameras"], PROFILES_PAYLOAD["profiles"]["realsense_rig"]
            )
            self.assertEqual(merged["robot"]["port"], "/dev/ttyACM0")
            self.assertEqual(merged["dataset"], RECORD_CONFIG["dataset"])
            merged_path.unlink()

    def test_default_profile_and_space_separated_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_path, config_path = self._write_inputs(root)
            argv = [
                "record.py",
                "--config_path",
                str(config_path),
                "--camera_profile",
                "default",
                "--camera_profiles_path",
                str(profiles_path),
            ]

            rewritten = camera_profiles.rewrite_argv_with_camera_profile(
                argv, allow_depth=True, search_root=root
            )

            self.assertEqual(rewritten[1], "--config_path")
            merged = yaml.safe_load(Path(rewritten[2]).read_text(encoding="utf-8"))
            self.assertEqual(
                merged["robot"]["cameras"], PROFILES_PAYLOAD["profiles"]["orbbec_rig"]
            )
            Path(rewritten[2]).unlink()

    def test_argv_unchanged_without_profile_flag(self) -> None:
        argv = ["record.py", "--config_path=record.yaml"]
        rewritten = camera_profiles.rewrite_argv_with_camera_profile(
            argv, allow_depth=True, search_root=Path(".")
        )
        self.assertEqual(rewritten, argv)

    def test_rgb_only_recorder_rejects_depth_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, config_path = self._write_inputs(root)
            argv = [
                "record.py",
                f"--config_path={config_path}",
                "--camera_profile=orbbec_rig",
            ]
            with self.assertRaisesRegex(camera_profiles.CameraProfileError, "RGB-only"):
                camera_profiles.rewrite_argv_with_camera_profile(
                    argv, allow_depth=False, search_root=root
                )

    def test_rgb_profile_passes_rgb_only_recorder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, config_path = self._write_inputs(root)
            argv = [
                "record.py",
                f"--config_path={config_path}",
                "--camera_profile=rgb_rig",
            ]
            rewritten = camera_profiles.rewrite_argv_with_camera_profile(
                argv, allow_depth=False, search_root=root
            )
            merged_path = Path(rewritten[1].split("=", 1)[1])
            merged = yaml.safe_load(merged_path.read_text(encoding="utf-8"))
            self.assertEqual(merged["robot"]["cameras"], PROFILES_PAYLOAD["profiles"]["rgb_rig"])
            merged_path.unlink()

    def test_unknown_profile_and_missing_config_path_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, config_path = self._write_inputs(root)
            with self.assertRaisesRegex(camera_profiles.CameraProfileError, "unknown camera profile"):
                camera_profiles.rewrite_argv_with_camera_profile(
                    ["record.py", f"--config_path={config_path}", "--camera_profile=nope"],
                    allow_depth=True,
                    search_root=root,
                )
            with self.assertRaisesRegex(camera_profiles.CameraProfileError, "--config_path"):
                camera_profiles.rewrite_argv_with_camera_profile(
                    ["record.py", "--camera_profile=rgb_rig"],
                    allow_depth=True,
                    search_root=root,
                )

    def test_depth_on_opencv_camera_is_rejected(self) -> None:
        payload = {
            "profiles": {
                "bad": {"cam": {"type": "opencv", "index_or_path": 0, "use_depth": True}},
            }
        }
        with self.assertRaisesRegex(camera_profiles.CameraProfileError, "no depth stream"):
            camera_profiles.validate_profile_cameras("bad", payload["profiles"]["bad"], allow_depth=True)

    def test_repo_camera_profiles_yaml_is_valid(self) -> None:
        payload = camera_profiles.load_camera_profiles(REPO_ROOT / "record" / "camera_profiles.yaml")
        for name, cameras in payload["profiles"].items():
            allow_depth = any(cam.get("use_depth") for cam in cameras.values())
            camera_profiles.validate_profile_cameras(name, cameras, allow_depth=allow_depth or True)
        self.assertIn(payload["default_profile"], payload["profiles"])


if __name__ == "__main__":
    unittest.main()
