import unittest

import numpy as np

from recorder.config import OpenCVCameraConfig, OrbbecCameraConfig
from recorder.dataset import depth_sidecar, features, stats

# repo-native PNG reader proves cross-tool depth compatibility
from utils.media_utils import read_png_array


class FeaturesTests(unittest.TestCase):
    def setUp(self):
        self.cameras = {
            "front": OrbbecCameraConfig(
                type="orbbec", serial_number_or_name="S", fps=30, width=64, height=48,
                use_depth=True, depth_fps=30, depth_width=64, depth_height=48, align_depth_to_color=True,
            ),
            "wrist": OpenCVCameraConfig(type="opencv", index_or_path=0, fps=30, width=32, height=24),
        }
        self.feats = features.build_features(
            ["shoulder_pan", "gripper"], self.cameras, fps=30, use_video=True, vcodec="libsvtav1"
        )

    def test_action_state_dims(self):
        self.assertEqual(self.feats["action"]["shape"], [2])
        self.assertEqual(self.feats["observation.state"]["names"], ["shoulder_pan.pos", "gripper.pos"])

    def test_rgb_is_video_with_codec(self):
        spec = self.feats["observation.images.front"]
        self.assertEqual(spec["dtype"], "video")
        self.assertEqual(spec["shape"], [48, 64, 3])
        self.assertEqual(spec["info"]["video.codec"], "av1")

    def test_depth_is_image(self):
        spec = self.feats["observation.images.front_depth"]
        self.assertEqual(spec["dtype"], "image")
        self.assertEqual(spec["shape"], [48, 64, 1])

    def test_video_and_image_keys(self):
        self.assertEqual(sorted(features.video_keys(self.feats)),
                         ["observation.images.front", "observation.images.wrist"])
        self.assertEqual(features.image_keys(self.feats), ["observation.images.front_depth"])

    def test_vectors(self):
        obs = {"shoulder_pan.pos": 1.5, "gripper.pos": -2.0}
        self.assertEqual(features.state_vector(obs, ["shoulder_pan", "gripper"]), [1.5, -2.0])


class DepthSidecarTests(unittest.TestCase):
    def test_png_roundtrip_via_repo_reader(self):
        import tempfile
        from pathlib import Path

        depth = (np.arange(48 * 64, dtype=np.uint16).reshape(48, 64) % 4000)
        with tempfile.TemporaryDirectory() as tmp:
            path = depth_sidecar.save_depth_sidecar_frame(tmp, "observation.images.front_depth", 0, 0, depth)
            self.assertTrue(Path(path).exists())
            read = read_png_array(Path(path))
        self.assertEqual(read.shape, (48, 64, 1))
        self.assertTrue(np.array_equal(read[:, :, 0].astype(np.uint16), depth))

    def test_key_parsing(self):
        self.assertEqual(depth_sidecar.depth_camera_name_from_key("observation.images.front_depth"), "front")
        self.assertIsNone(depth_sidecar.depth_camera_name_from_key("observation.images.front"))
        self.assertEqual(
            str(depth_sidecar.depth_sidecar_relpath("k", 2, 5)).replace("\\", "/"),
            "depth_sidecar/k/episode-000002/frame-000005.png",
        )


class StatsTests(unittest.TestCase):
    def test_vector_stats_shapes(self):
        matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        s = stats.vector_stats(matrix, 3)
        self.assertEqual(s["min"], [1.0, 2.0])
        self.assertEqual(s["max"], [5.0, 6.0])
        self.assertEqual(s["count"], [3])

    def test_image_accumulator_channels(self):
        acc = stats.ImageStatAccumulator(3, divisor=255.0)
        acc.update(np.full((4, 4, 3), 255, dtype=np.uint8))
        result = acc.result()
        self.assertEqual(result["count"], [1])
        # normalized white -> mean 1.0 per channel, nested [C,1,1]
        self.assertEqual(result["mean"], [[[1.0]], [[1.0]], [[1.0]]])


if __name__ == "__main__":
    unittest.main()
