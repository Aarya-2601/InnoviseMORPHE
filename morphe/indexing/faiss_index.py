"""
Per-module FAISS Index management for MORPHE.
Features:
- Individual FAISS IndexFlatIP (Inner Product = Cosine Similarity with normalized vectors)
- Bidirectional mapping between internal FAISS integer IDs and immutable record SHA-256 IDs
- Dynamic add, remove (rebuild/reindex), and search
- Serialization and disk persistence (.faiss and mapping json)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import faiss
import numpy as np


class ModuleFaissIndex:
    """
    Dedicated FAISS vector index for a single MORPHE memory module.
    Uses cosine similarity (IndexFlatIP with L2-normalized embeddings).
    """

    def __init__(self, module_id: str, dimension: int = 384):
        self.module_id = module_id
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        # Bidirectional ID mappings
        self.int_to_record_id: Dict[int, str] = {}
        self.record_id_to_int: Dict[str, int] = {}
        # Keep an in-memory cache of normalized vectors for rebuilding upon record removals
        self._vectors: Dict[str, np.ndarray] = {}

    def add(
        self,
        record_id: str,
        embedding: Union[np.ndarray, Sequence[float]],
    ) -> int:
        """
        Add a record embedding to the module index.
        If already present, updates the embedding.
        Returns the internal integer index.
        """
        emb_arr = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if emb_arr.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {emb_arr.shape[1]}"
            )

        # L2-normalize for Inner Product (Cosine similarity)
        norm = np.linalg.norm(emb_arr)
        if norm > 1e-9:
            emb_arr /= norm

        if record_id in self.record_id_to_int:
            # Already exists: update vector cache and rebuild index
            self._vectors[record_id] = emb_arr.squeeze(0)
            self._rebuild()
            return self.record_id_to_int[record_id]

        new_int_id = self.index.ntotal
        self.index.add(emb_arr)
        self.int_to_record_id[new_int_id] = record_id
        self.record_id_to_int[record_id] = new_int_id
        self._vectors[record_id] = emb_arr.squeeze(0)
        return new_int_id

    def add_batch(
        self,
        record_ids: Sequence[str],
        embeddings: Union[np.ndarray, Sequence[Sequence[float]]],
    ) -> List[int]:
        """
        Add a batch of record embeddings to the module index.
        """
        if len(record_ids) == 0:
            return []

        emb_matrix = np.asarray(embeddings, dtype=np.float32)
        if emb_matrix.ndim == 1:
            emb_matrix = emb_matrix.reshape(1, -1)

        if emb_matrix.shape[0] != len(record_ids):
            raise ValueError(
                f"Count mismatch: {len(record_ids)} record IDs vs {emb_matrix.shape[0]} embeddings"
            )
        if emb_matrix.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension mismatch: expected {self.dimension}, got {emb_matrix.shape[1]}"
            )

        # L2 normalize rows
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        emb_matrix /= norms

        added_indices: List[int] = []
        for rid, vec in zip(record_ids, emb_matrix):
            self._vectors[rid] = vec
        self._rebuild()
        for rid in record_ids:
            added_indices.append(self.record_id_to_int[rid])

        return added_indices

    def remove(self, record_id: str) -> bool:
        """
        Remove a record from the module index.
        Rebuilds the index from the remaining vector cache.
        """
        if record_id not in self._vectors:
            return False

        del self._vectors[record_id]
        self._rebuild()
        return True

    def _rebuild(self) -> None:
        """Reconstruct the FAISS IndexFlatIP cleanly from the vector cache."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.int_to_record_id.clear()
        self.record_id_to_int.clear()

        if not self._vectors:
            return

        record_ids = list(self._vectors.keys())
        matrix = np.stack([self._vectors[rid] for rid in record_ids], axis=0)
        self.index.add(matrix)

        for int_id, rid in enumerate(record_ids):
            self.int_to_record_id[int_id] = rid
            self.record_id_to_int[rid] = int_id

    def search(
        self,
        query_embedding: Union[np.ndarray, Sequence[float]],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Search for top_k most similar records inside this module.
        Returns list of (record_id, cosine_score) sorted by descending similarity.
        """
        if self.index.ntotal == 0:
            return []

        q_vec = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(q_vec)
        if norm > 1e-9:
            q_vec /= norm

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_vec, k)

        results: List[Tuple[str, float]] = []
        for score, int_id in zip(scores[0], indices[0]):
            if int_id in self.int_to_record_id:
                results.append((self.int_to_record_id[int_id], float(score)))

        return results

    def compute_centroid(self) -> np.ndarray:
        """
        Compute module centroid (mean of all vectors in module, L2-normalized).
        Used by the MORPHE router to score module affinity.
        """
        if not self._vectors:
            return np.zeros(self.dimension, dtype=np.float32)

        matrix = np.stack(list(self._vectors.values()), axis=0)
        centroid = np.mean(matrix, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-9:
            centroid /= norm
        return centroid

    def save(self, directory: Union[str, Path]) -> None:
        """Save index and mapping to disk."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        index_file = dir_path / f"module_{self.module_id}.faiss"
        meta_file = dir_path / f"module_{self.module_id}.json"

        faiss.write_index(self.index, str(index_file))
        meta_data = {
            "module_id": self.module_id,
            "dimension": self.dimension,
            "record_id_to_int": self.record_id_to_int,
            "int_to_record_id": {str(k): v for k, v in self.int_to_record_id.items()},
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)

    @classmethod
    def load(cls, directory: Union[str, Path], module_id: str) -> ModuleFaissIndex:
        """Load index from disk."""
        dir_path = Path(directory)
        index_file = dir_path / f"module_{module_id}.faiss"
        meta_file = dir_path / f"module_{module_id}.json"

        if not index_file.exists() or not meta_file.exists():
            raise FileNotFoundError(f"Index files for module {module_id} not found in {dir_path}")

        with open(meta_file, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        inst = cls(module_id=meta_data["module_id"], dimension=meta_data["dimension"])
        inst.index = faiss.read_index(str(index_file))
        inst.record_id_to_int = meta_data["record_id_to_int"]
        inst.int_to_record_id = {int(k): v for k, v in meta_data["int_to_record_id"].items()}

        # Reconstruct internal vector cache from FAISS index
        for int_id, rid in inst.int_to_record_id.items():
            vec = inst.index.reconstruct(int_id)
            inst._vectors[rid] = np.asarray(vec, dtype=np.float32)

        return inst

    @property
    def size(self) -> int:
        """Total records indexed in this module."""
        return self.index.ntotal

