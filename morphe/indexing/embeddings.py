"""
Embedding service for MORPHE.
Supports:
- Local SentenceTransformers model (e.g., all-MiniLM-L6-v2)
- Lightweight Deterministic Fallback Embedding generator (for unit tests / offline execution)
- L2-normalized numpy float32 embeddings
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence, Union
import numpy as np


class EmbeddingService:
    """
    Service for generating dense embeddings.
    Defaults to all-MiniLM-L6-v2, with an automatic or selectable fast deterministic fallback.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
        use_fallback_only: bool = False,
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.use_fallback_only = use_fallback_only
        self._model = None
        self._initialized = False

    def _init_model(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        if self.use_fallback_only:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            # Infer actual dimension
            test_emb = self._model.encode("test", normalize_embeddings=True)
            self.dimension = test_emb.shape[0]
        except Exception:
            # Fallback gracefully if model cannot be downloaded or initialized
            self._model = None

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string and return an L2-normalized float32 vector."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Embed multiple text strings and return an (N, dimension) float32 array."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        self._init_model()

        if self._model is not None:
            vectors = self._model.encode(
                list(texts),
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return np.asarray(vectors, dtype=np.float32)

        # Deterministic fast fallback (hashing-based pseudo-semantic vector)
        return self._fallback_embed_batch(texts)

    def _fallback_embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """
        Deterministic pseudo-semantic embedding based on token n-grams and hashing.
        Guarantees reproducible, L2-normalized vectors of shape (N, dimension).
        """
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            words = text.lower().split()
            if not words:
                words = ["<empty>"]
            vec = np.zeros(self.dimension, dtype=np.float32)
            for w in words:
                # Seed deterministically by word MD5
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
                rng = np.random.default_rng(h % (2**32))
                component = rng.standard_normal(self.dimension).astype(np.float32)
                vec += component
            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec /= norm
            out[i] = vec
        return out

