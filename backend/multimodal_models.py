from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


@dataclass
class QueryVector:
    vector: List[float]
    token: str


class MultiModalEmbedder:
    """
    CLIP-based shared embedding space for text + images.
    Produces multi-vector outputs for late-interaction retrieval.
    """

    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norm

    def embed_text_multi(self, text: str, window_words: int = 24, overlap: int = 8) -> List[List[float]]:
        words = text.split()
        if not words:
            return []

        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i : i + window_words])
            if chunk.strip():
                chunks.append(chunk)
            if i + window_words >= len(words):
                break
            i += max(1, window_words - overlap)

        embs = self.model.encode(chunks, normalize_embeddings=True)
        return embs.tolist() if len(chunks) > 1 else [embs.tolist()] if embs.ndim == 1 else embs.tolist()

    def embed_query_multi(self, query: str) -> List[QueryVector]:
        tokens = [t.strip() for t in query.replace("?", " ").split() if t.strip()]
        if not tokens:
            return []

        # token-level (coarse ColBERT-style approximation)
        token_embs = self.model.encode(tokens, normalize_embeddings=True)
        vectors = token_embs.tolist() if token_embs.ndim > 1 else [token_embs.tolist()]
        return [QueryVector(vector=v, token=tokens[i]) for i, v in enumerate(vectors)]

    def embed_image_multi(self, image: Image.Image, grid: int = 2) -> List[List[float]]:
        w, h = image.size
        patches = []
        for gx in range(grid):
            for gy in range(grid):
                left = int(gx * w / grid)
                right = int((gx + 1) * w / grid)
                top = int(gy * h / grid)
                bottom = int((gy + 1) * h / grid)
                patches.append(image.crop((left, top, right, bottom)))

        embs = self.model.encode(patches, normalize_embeddings=True)
        return embs.tolist() if embs.ndim > 1 else [embs.tolist()]
