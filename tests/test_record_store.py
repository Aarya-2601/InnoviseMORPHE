"""
Unit and integration tests for MORPHE Content-Addressed Record Store.
"""

import hashlib
import numpy as np
import pytest

from morphe.storage.record_store import Record, RecordStore


def test_record_sha256_id_generation():
    text = "PostgreSQL is used for relational metadata in MORPHE."
    expected_id = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
    
    rec = Record.create(text=text, metadata={"project": "MORPHE", "type": "tech_spec"})
    assert rec.id == expected_id
    assert rec.text == text
    assert rec.metadata["project"] == "MORPHE"
    assert rec.metadata["type"] == "tech_spec"
    assert rec.archived is False


def test_record_immutability():
    rec = Record.create(text="Immutable test record")
    with pytest.raises(Exception):
        # Attempting to modify frozen dataclass attribute must fail
        rec.text = "Altered text"  # type: ignore


def test_record_store_put_and_get():
    store = RecordStore(":memory:")
    sample_text = "Dense + BM25 + RRF forms the hybrid retrieval core."
    embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    rec, is_new = store.put(
        text=sample_text,
        metadata={"domain": "retrieval", "priority": "high"},
        embedding=embedding,
    )

    assert is_new is True
    assert store.contains(rec.id)
    assert store.count() == 1

    fetched = store.get(rec.id)
    assert fetched is not None
    assert fetched.id == rec.id
    assert fetched.text == sample_text
    assert fetched.metadata["domain"] == "retrieval"
    assert fetched.embedding is not None
    np.testing.assert_allclose(fetched.embedding, embedding, atol=1e-6)


def test_record_store_deduplication():
    store = RecordStore(":memory:")
    text = "Exact identical content ingested multiple times."

    rec1, is_new1 = store.put(text=text, metadata={"version": 1})
    rec2, is_new2 = store.put(text=text, metadata={"version": 2})

    assert is_new1 is True
    assert is_new2 is False
    assert rec1.id == rec2.id
    # Store count must remain 1 because it's content-addressed
    assert store.count() == 1


def test_record_store_batch_and_get_many():
    store = RecordStore(":memory:")
    items = [
        ("Chunk A: Memory-Oriented Reconfigurable Plasticity", {"idx": 1}),
        ("Chunk B: Hierarchical Evolution of Modules", {"idx": 2}),
        ("Chunk C: Verifiable Ground Truth Reward", {"idx": 3}),
        ("Chunk A: Memory-Oriented Reconfigurable Plasticity", {"idx": 1}),  # duplicate
    ]

    results = store.put_batch(items)
    assert len(results) == 4
    assert results[0][1] is True  # new
    assert results[1][1] is True  # new
    assert results[2][1] is True  # new
    assert results[3][1] is False  # duplicate!

    assert store.count() == 3

    all_ids = store.list_all_ids()
    assert len(all_ids) == 3

    many = store.get_many(all_ids[:2])
    assert len(many) == 2
    assert all_ids[0] in many
    assert all_ids[1] in many


def test_record_store_archive_and_soft_retention():
    store = RecordStore(":memory:")
    rec, _ = store.put("Stale fact that will be superseded.")
    
    assert store.count(include_archived=False) == 1
    
    # Soft archive
    success = store.set_archived(rec.id, archived=True)
    assert success is True
    
    # Immutable record still exists in store
    assert store.count(include_archived=True) == 1
    assert store.count(include_archived=False) == 0

    fetched = store.get(rec.id)
    assert fetched is not None
    assert fetched.archived is True


def test_record_store_disk_persistence(tmp_path):
    db_file = tmp_path / "test_records.db"
    store1 = RecordStore(db_file)
    emb = np.array([0.5, -0.5, 1.0], dtype=np.float32)
    rec, _ = store1.put("Persisted memory chunk", metadata={"persistent": True}, embedding=emb)
    store1.close()

    # Re-open from disk
    store2 = RecordStore(db_file)
    assert store2.contains(rec.id)
    fetched = store2.get(rec.id)
    assert fetched is not None
    assert fetched.text == "Persisted memory chunk"
    assert fetched.metadata["persistent"] is True
    assert fetched.embedding is not None
    np.testing.assert_allclose(fetched.embedding, emb, atol=1e-6)
    store2.close()

