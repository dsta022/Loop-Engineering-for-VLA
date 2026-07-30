"""Tests for the alignment-based semantic evaluator.

These tests use a deterministic fake vision-language backend, so they exercise
the full scoring path — frame handling, prompt construction, stage-anchored and
instruction-only curves, and the SemanticResult contract — without torch, a GPU,
or a model download. ffmpeg is stubbed by injecting frames through a fake
sampler on the evaluator.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from semantic.backends import VisionLanguageBackend, l2_normalize
from semantic.clip_alignment_evaluator import ClipAlignmentEvaluator
from utils.semantic_evaluator import (
    SemanticEpisode,
    VideoSegment,
    normalize_result,
)


class KeywordBackend:
    """Fake backend: image/text vectors placed on a fixed keyword basis.

    Each frame and each text is mapped to a unit vector in a small space whose
    axes are keywords. A frame "resembles" a stage when they share a keyword, so
    scripted frame sequences produce predictable, monotone progress curves.
    """

    def __init__(self, vocabulary: list[str]) -> None:
        self.name = "keyword_fake_backend"
        self.embedding_dim = len(vocabulary)
        self._vocabulary = vocabulary
        self._index = {word: i for i, word in enumerate(vocabulary)}

    def _vector_for_words(self, words: list[str]) -> np.ndarray:
        vector = np.zeros(self.embedding_dim, dtype=np.float32)
        for word in words:
            if word in self._index:
                vector[self._index[word]] += 1.0
        if not vector.any():
            vector[:] = 1.0
        return vector

    def embed_images(self, frames):
        # Each fake frame encodes its keyword in pixel [0, 0, 0] as a vocab index.
        vectors = []
        for frame in frames:
            code = int(np.asarray(frame).reshape(-1)[0])
            word = self._vocabulary[code % self.embedding_dim]
            vectors.append(self._vector_for_words([word]))
        return l2_normalize(np.stack(vectors, axis=0))

    def embed_texts(self, texts):
        vectors = [
            self._vector_for_words([w for w in self._vocabulary if w in text.lower()])
            for text in texts
        ]
        return l2_normalize(np.stack(vectors, axis=0))


def frame_with_code(code: int, size: int = 4) -> np.ndarray:
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[0, 0, 0] = code
    return frame


class InjectedSampler:
    """Replaces frame sampling with a scripted list of coded frames."""

    def __init__(self, codes: list[int]) -> None:
        self.codes = codes

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return np.stack([frame_with_code(code) for code in self.codes], axis=0)


def make_episode(profile=None, tasks=("reach grasp lift",)) -> SemanticEpisode:
    return SemanticEpisode(
        dataset_root=Path("/dataset"),
        episode_index=0,
        length=len(("f",)) * 8,
        tasks=tuple(tasks),
        videos=(
            VideoSegment(
                key="observation.images.global",
                path=Path("/dataset/videos/observation.images.global/chunk-000/file-000.mp4"),
                from_timestamp=0.0,
                to_timestamp=2.0,
            ),
        ),
        task_profile=profile,
    )


STAGE_PROFILE = {
    "task_key": "reach_grasp_lift",
    "task_name": "reach grasp lift",
    "instruction": "reach the cube, grasp it, and lift it",
    "stages": [
        {"name": "reach", "description": "arm approaches", "observable_condition": "reach", "weight": 1.0},
        {"name": "grasp", "description": "gripper closes", "observable_condition": "grasp", "weight": 1.0},
        {"name": "lift", "description": "cube leaves table", "observable_condition": "lift", "weight": 1.0},
    ],
}


class StageAnchoredCurveTest(unittest.TestCase):
    def _evaluator(self, codes):
        backend = KeywordBackend(["reach", "grasp", "lift"])
        evaluator = ClipAlignmentEvaluator(
            backend,
            num_frames=len(codes),
            frame_size=4,
            ffmpeg_path="/usr/bin/true",
            prompt_template="{condition}",
        )
        return evaluator

    def setUp(self):
        # Patch the module-level sampler the evaluator calls.
        import semantic.clip_alignment_evaluator as mod

        self._mod = mod
        self._real_sampler = mod.sample_segment_frames

    def tearDown(self):
        self._mod.sample_segment_frames = self._real_sampler

    def _run(self, codes):
        evaluator = self._evaluator(codes)
        self._mod.sample_segment_frames = InjectedSampler(codes)
        return evaluator.evaluate(make_episode(STAGE_PROFILE))

    def test_progressing_episode_rises_to_final_stage(self):
        # frames move reach(0) -> grasp(1) -> lift(2)
        result = self._run([0, 0, 1, 1, 2, 2])
        normalized = normalize_result(result)
        self.assertEqual(normalized.details["method"], "stage_anchored")
        self.assertGreater(normalized.final_progress, normalized.progress_curve[0])
        self.assertGreater(normalized.final_progress, 0.7)

    def test_stalled_episode_stays_low(self):
        # every frame still looks like the first stage
        result = self._run([0, 0, 0, 0, 0, 0])
        normalized = normalize_result(result)
        self.assertLess(normalized.final_progress, 0.5)

    def test_regression_shows_in_curve(self):
        # reaches lift then falls back to reach
        result = self._run([0, 1, 2, 2, 0, 0])
        normalized = normalize_result(result)
        peak = max(normalized.progress_curve)
        self.assertGreaterEqual(peak - normalized.final_progress, 0.15)

    def test_curve_within_unit_interval(self):
        result = self._run([0, 1, 2, 1, 2, 0])
        for value in result.progress_curve:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


class InstructionOnlyCurveTest(unittest.TestCase):
    def setUp(self):
        import semantic.clip_alignment_evaluator as mod

        self._mod = mod
        self._real_sampler = mod.sample_segment_frames

    def tearDown(self):
        self._mod.sample_segment_frames = self._real_sampler

    def _run(self, codes, tasks):
        backend = KeywordBackend(["reach", "grasp", "lift"])
        evaluator = ClipAlignmentEvaluator(
            backend, num_frames=len(codes), frame_size=4, ffmpeg_path="/usr/bin/true"
        )
        self._mod.sample_segment_frames = InjectedSampler(codes)
        return evaluator.evaluate(make_episode(profile=None, tasks=tasks))

    def test_instruction_only_when_no_profile(self):
        result = self._run([0, 1, 2], tasks=("lift the object",))
        normalized = normalize_result(result)
        self.assertEqual(normalized.details["method"], "instruction_only")
        # min-max normalized: at least one 0.0 and one 1.0 unless degenerate
        self.assertAlmostEqual(min(normalized.progress_curve), 0.0, places=6)
        self.assertAlmostEqual(max(normalized.progress_curve), 1.0, places=6)

    def test_no_task_text_raises(self):
        with self.assertRaises(RuntimeError):
            self._run([0, 1], tasks=("",))


class ContractTest(unittest.TestCase):
    def setUp(self):
        import semantic.clip_alignment_evaluator as mod

        self._mod = mod
        self._real_sampler = mod.sample_segment_frames

    def tearDown(self):
        self._mod.sample_segment_frames = self._real_sampler

    def test_result_passes_normalize(self):
        backend = KeywordBackend(["reach", "grasp", "lift"])
        evaluator = ClipAlignmentEvaluator(
            backend, num_frames=4, frame_size=4, ffmpeg_path="/usr/bin/true"
        )
        self._mod.sample_segment_frames = InjectedSampler([0, 1, 2, 2])
        result = evaluator.evaluate(make_episode(STAGE_PROFILE))
        normalized = normalize_result(result)  # must not raise
        import json

        json.dumps(normalized.details)  # details must be JSON-serializable

    def test_missing_video_raises(self):
        backend = KeywordBackend(["reach"])
        evaluator = ClipAlignmentEvaluator(
            backend, num_frames=4, frame_size=4, ffmpeg_path="/usr/bin/true"
        )
        episode = SemanticEpisode(
            dataset_root=Path("/dataset"),
            episode_index=0,
            length=8,
            tasks=("reach",),
            videos=(),
        )
        with self.assertRaises(ValueError):
            evaluator.evaluate(episode)

    def test_backend_protocol_satisfied(self):
        self.assertIsInstance(KeywordBackend(["a", "b"]), VisionLanguageBackend)


if __name__ == "__main__":
    unittest.main()
