from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


@dataclass
class QueryVector:
    vector: List[float]
    token: str


class MultiModalEmbedder:
    """
    Multi-vector embedder with two backends:
    - colpali (preferred): document-native late-interaction embeddings
    - clip (fallback): sentence-transformers CLIP shared embedding space
    """

    def __init__(self, backend: str = "colpali", model_name: str = "vidore/colpali-v1.2"):
        self.backend = backend.lower().strip()
        self.model_name = model_name

        if self.backend == "colpali":
            # Optional dependency path; kept lazy/guarded to avoid hard import failures.
            try:
                from colpali_engine.models import ColPali, ColPaliProcessor  # type: ignore
                import torch  # type: ignore

                self._torch = torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self.colpali = ColPali.from_pretrained(self.model_name).to(self._device).eval()
                self.processor = ColPaliProcessor.from_pretrained(self.model_name)
                return
            except Exception as e:
                raise RuntimeError(
                    "ColPali backend requested but unavailable. Install compatible colpali-engine/torch stack "
                    "or switch MM_BACKEND=clip."
                ) from e

        # fallback backend
        self.backend = "clip"
        self.clip_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32")

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norm

    def embed_text_multi(self, text: str, window_words: int = 24, overlap: int = 8) -> List[List[float]]:
        if self.backend == "colpali":
            with self._torch.inference_mode():
                batch = self.processor.process_queries([text]).to(self._device)
                output = self.colpali(**batch)
                # [tokens, dim] multi-vector query/doc representation
                vecs = output[0].detach().float().cpu().numpy()
                vecs = self._normalize(vecs)
                return vecs.tolist()

        # CLIP fallback
        words = text.split()
        if not words:
            return []

        chunks = []
        i = 0
        while i < len(words):
            chunks.append(" ".join(words[i : i + window_words]))
            if i + window_words >= len(words):
                break
            i += max(1, window_words - overlap)

        embs = self.clip_model.encode(chunks, normalize_embeddings=True)
        if isinstance(embs, np.ndarray) and embs.ndim == 1:
            return [embs.tolist()]
        return embs.tolist()

    def embed_query_multi(self, query: str) -> List[QueryVector]:
        if self.backend == "colpali":
            vectors = self.embed_text_multi(query)
            # token ids are internal; use positional tokens for stable rerank bookkeeping
            return [QueryVector(vector=v, token=f"q{i}") for i, v in enumerate(vectors)]

        tokens = [t.strip() for t in query.replace("?", " ").split() if t.strip()]
        if not tokens:
            return []
        token_embs = self.clip_model.encode(tokens, normalize_embeddings=True)
        vectors = token_embs.tolist() if getattr(token_embs, "ndim", 1) > 1 else [token_embs.tolist()]
        return [QueryVector(vector=v, token=tokens[i]) for i, v in enumerate(vectors)]

    def embed_image_multi(self, image: Image.Image, grid: int = 2) -> List[List[float]]:
        if self.backend == "colpali":
            with self._torch.inference_mode():
                batch = self.processor.process_images([image]).to(self._device)
                output = self.colpali(**batch)
                vecs = output[0].detach().float().cpu().numpy()
                vecs = self._normalize(vecs)
                return vecs.tolist()

        # CLIP fallback multi-patch vectors
        w, h = image.size
        patches = []
        for gx in range(grid):
            for gy in range(grid):
                left = int(gx * w / grid)
                right = int((gx + 1) * w / grid)
                top = int(gy * h / grid)
                bottom = int((gy + 1) * h / grid)
                patches.append(image.crop((left, top, right, bottom)))

        embs = self.clip_model.encode(patches, normalize_embeddings=True)
        if isinstance(embs, np.ndarray) and embs.ndim == 1:
            return [embs.tolist()]
        return embs.tolist()
