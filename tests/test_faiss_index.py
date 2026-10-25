"""
Tests for SentenceTransformers embedding service and per-module FAISS indices.
"""

import numpy as np
import pytest
from morphe.indexing.embeddings import EmbeddingService
from morphe.indexing.faiss_index import ModuleFaissIndex


def test_embedding_service_dimensions_and_normalization():
    # Use fallback mode for instant, deterministic testing
    service = EmbeddingService(dimension=64, use_fallback_only=True)
    text = "Reinforcement learning optimizes memory topology."
    
    vec = service.embed_text(text)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (64,)
    assert vec.dtype == np.float32
    # Verify L2 normalization
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, abs=1e-5) == 1.0


def test_embedding_service_batch_embedding():
    service = EmbeddingService(dimension=64, use_fallback_only=True)
    texts = [
        "Module A handles authentication.",
        "Module B coordinates database sharding.",
        "Module C processes telemetry metrics."
    ]
    batch = service.embed_batch(texts)
    assert batch.shape == (3, 64)
    for i in range(3):
        assert pytest.approx(np.linalg.norm(batch[i]), abs=1e-5) == 1.0


def test_module_faiss_index_add_and_search():
    index = ModuleFaissIndex(module_id="mod_auth", dimension=32)
    
    # 3 synthetic vectors
    v1 = np.ones(32, dtype=np.float32)
    v2 = np.zeros(32, dtype=np.float32)
    v2[0] = 1.0
    v3 = -np.ones(32, dtype=np.float32)

    index.add("rec_auth_1", v1)
    index.add("rec_auth_2", v2)
    index.add("rec_auth_3", v3)

    assert index.size == 3

    # Query with something most similar to v1
    q = np.ones(32, dtype=np.float32)
    results = index.search(q, top_k=2)

    assert len(results) == 2
    # Top result should be rec_auth_1 with cosine ~ 1.0
    assert results[0][0] == "rec_auth_1"
    assert pytest.approx(results[0][1], abs=1e-4) == 1.0


def test_module_faiss_centroid_computation():
    index = ModuleFaissIndex(module_id="mod_db", dimension=4)
    # Vectors along axis 0
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    index.add("r1", v1)
    index.add("r2", v2)

    centroid = index.compute_centroid()
    assert centroid.shape == (4,)
    np.testing.assert_allclose(centroid, [1.0, 0.0, 0.0, 0.0], atol=1e-5)


def test_module_faiss_record_removal_and_rebuild():
    index = ModuleFaissIndex(module_id="mod_telemetry", dimension=16)
    v1 = np.random.randn(16).astype(np.float32)
    v2 = np.random.randn(16).astype(np.float32)
    v3 = np.random.randn(16).astype(np.float32)

    index.add("r1", v1)
    index.add("r2", v2)
    index.add("r3", v3)
    assert index.size == 3

    # Remove r2
    removed = index.remove("r2")
    assert removed is True
    assert index.size == 2

    # Search should no longer return r2
    results = index.search(v2, top_k=3)
    returned_ids = [r[0] for r in results]
    assert "r2" not in returned_ids
    assert len(results) == 2


def test_module_faiss_disk_persistence(tmp_path):
    index = ModuleFaissIndex(module_id="mod_persist", dimension=8)
    v1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    index.add("rec_p1", v1)
    index.add("rec_p2", v2)

    index.save(tmp_path)

    # Load into a new index instance
    loaded = ModuleFaissIndex.load(tmp_path, module_id="mod_persist")
    assert loaded.size == 2
    assert loaded.module_id == "mod_persist"
    
    # Verify search fidelity
    results = loaded.search(v2, top_k=1)
    assert len(results) == 1
    assert results[0][0] == "rec_p2"
    assert pytest.approx(results[0][1], abs=1e-4) == 1.0

