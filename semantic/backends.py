"""Vision-language embedding backends for alignment-based semantic scoring.

The evaluator depends only on the :class:`VisionLanguageBackend` protocol: given
frames it returns L2-normalized image embeddings, and given texts it returns
L2-normalized text embeddings in the same space. This keeps the evaluator's
scoring logic testable with a lightweight fake and independent of any specific
model runtime.

:class:`OpenClipBackend` is the real implementation. It imports ``torch`` and
``open_clip`` lazily inside ``__init__`` so that importing this module — during
the base audit or the unit tests — never pulls in a model runtime.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VisionLanguageBackend(Protocol):
    """A shared image/text embedding space with L2-normalized outputs."""

    name: str
    embedding_dim: int

    def embed_images(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        """Embed ``(H, W, 3)`` uint8 RGB frames as ``(N, D)`` unit vectors."""

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed text prompts as ``(M, D)`` unit vectors."""


def l2_normalize(matrix: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """Return ``matrix`` scaled to unit L2 norm along ``axis``."""
    norms = np.linalg.norm(matrix, axis=axis, keepdims=True)
    return matrix / np.maximum(norms, eps)


class OpenClipBackend:
    """open_clip CLIP/SigLIP embedding backend (lazy torch import).

    Config keys (all optional):
      - ``model_name``: open_clip architecture, e.g. ``"ViT-B-32"`` or
        ``"ViT-SO400M-14-SigLIP-384"``.
      - ``pretrained``: pretrained tag, e.g. ``"laion2b_s34b_b79k"`` or
        ``"webli"``.
      - ``device``: ``"cuda"``, ``"cpu"``, or ``None`` to auto-select.
      - ``batch_size``: image embedding batch size.
    """

    def __init__(
        self,
        *,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        try:
            import torch
            import open_clip
        except Exception as exc:  # pragma: no cover - exercised only with the runtime
            raise ImportError(
                "OpenClipBackend requires torch and open_clip; install them into the "
                "model environment (e.g. the 'hao' conda env) with "
                "`pip install open_clip_torch`"
            ) from exc

        self._torch = torch
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=resolved_device
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._device = resolved_device
        self.name = f"openclip:{model_name}:{pretrained}"
        self.embedding_dim = int(model.text_projection.shape[1]) if hasattr(
            model, "text_projection"
        ) else int(model.visual.output_dim)
        self._batch_size = max(1, int(batch_size))

    def embed_images(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        from PIL import Image

        torch = self._torch
        tensors = [
            self._preprocess(Image.fromarray(np.ascontiguousarray(frame)))
            for frame in frames
        ]
        if not tensors:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(tensors), self._batch_size):
                batch = torch.stack(tensors[start : start + self._batch_size]).to(
                    self._device
                )
                features = self._model.encode_image(batch)
                outputs.append(features.float().cpu().numpy())
        return l2_normalize(np.concatenate(outputs, axis=0))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        tokens = self._tokenizer(list(texts)).to(self._device)
        with torch.no_grad():
            features = self._model.encode_text(tokens).float().cpu().numpy()
        return l2_normalize(features)
