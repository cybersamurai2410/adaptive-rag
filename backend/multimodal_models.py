from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image


@dataclass
class QueryVector:
    vector: List[float]
    token: str


class MultiModalEmbedder:
    """ColPali-only multi-vector embedder (no fallback backend)."""

    def __init__(self, model_name: str = "vidore/colpali-v1.2"):
        self.model_name = model_name

        try:
            from colpali_engine.models import ColPali, ColPaliProcessor  # type: ignore
            import torch  # type: ignore

            self._torch = torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self.colpali = ColPali.from_pretrained(self.model_name).to(self._device).eval()
            self.processor = ColPaliProcessor.from_pretrained(self.model_name)
        except Exception as e:
            raise RuntimeError(
                "ColPali is required. Install compatible colpali-engine/torch stack."
            ) from e

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norm

    def embed_text_multi(self, text: str) -> List[List[float]]:
        with self._torch.inference_mode():
            batch = self.processor.process_queries([text]).to(self._device)
            output = self.colpali(**batch)
            vecs = output[0].detach().float().cpu().numpy()
            vecs = self._normalize(vecs)
            return vecs.tolist()

    def embed_query_multi(self, query: str) -> List[QueryVector]:
        vectors = self.embed_text_multi(query)
        return [QueryVector(vector=v, token=f"q{i}") for i, v in enumerate(vectors)]

    def embed_image_multi(self, image: Image.Image, grid: int = 2) -> List[List[float]]:
        # grid is unused for ColPali but kept for interface compatibility.
        _ = grid
        with self._torch.inference_mode():
            batch = self.processor.process_images([image]).to(self._device)
            output = self.colpali(**batch)
            vecs = output[0].detach().float().cpu().numpy()
            vecs = self._normalize(vecs)
            return vecs.tolist()
