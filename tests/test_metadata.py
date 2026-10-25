"""
Tests for metadata tagging (project, memory type, timestamp) and querying.
"""

import time
import pytest
from morphe.models.metadata import MemoryType, RecordMetadata
from morphe.storage.record_store import Record, RecordStore


def test_record_metadata_defaults_and_parsing():
    meta = RecordMetadata(project="InnoviseAlpha", memory_type=MemoryType.PREFERENCE)
    assert meta.project == "InnoviseAlpha"
    assert meta.memory_type == MemoryType.PREFERENCE
    assert meta.timestamp > 0
    assert "T" in meta.iso_time  # valid ISO format

    d = meta.to_dict()
    assert d["project"] == "InnoviseAlpha"
    assert d["memory_type"] == "preference"

    restored = RecordMetadata.from_dict(d)
    assert restored.project == "InnoviseAlpha"
    assert restored.memory_type == MemoryType.PREFERENCE


def test_record_creation_with_metadata_dict_or_object():
    # Pass as dict
    rec1 = Record.create(
        "User prefers concise answers with Python code.",
        metadata={"project": "Assistant", "memory_type": "preference", "tags": ["ui", "concise"]}
    )
    assert rec1.parsed_metadata.project == "Assistant"
    assert rec1.parsed_metadata.memory_type == MemoryType.PREFERENCE
    assert "concise" in rec1.parsed_metadata.tags

    # Pass as typed RecordMetadata
    meta_obj = RecordMetadata(project="InnoviseBeta", memory_type=MemoryType.TECH_SPEC, tags=["rl", "bandit"])
    rec2 = Record.create("LinUCB maintains upper confidence bounds.", metadata=meta_obj)
    assert rec2.parsed_metadata.project == "InnoviseBeta"
    assert rec2.parsed_metadata.memory_type == MemoryType.TECH_SPEC


def test_record_store_indexing_and_filtering_by_metadata():
    store = RecordStore(":memory:")
    
    t0 = 1700000000.0
    t1 = 1700001000.0
    t2 = 1700002000.0

    store.put(
        "Record 1: Project Alpha doc",
        metadata=RecordMetadata(project="Alpha", memory_type=MemoryType.DOC, timestamp=t0),
    )
    store.put(
        "Record 2: Project Alpha preference",
        metadata=RecordMetadata(project="Alpha", memory_type=MemoryType.PREFERENCE, timestamp=t1),
    )
    store.put(
        "Record 3: Project Beta doc",
        metadata=RecordMetadata(project="Beta", memory_type=MemoryType.DOC, timestamp=t2),
    )

    # 1. Filter by project
    alpha_recs = store.find_by_metadata(project="Alpha")
    assert len(alpha_recs) == 2
    assert all(r.parsed_metadata.project == "Alpha" for r in alpha_recs)

    # 2. Filter by memory_type
    doc_recs = store.find_by_metadata(memory_type=MemoryType.DOC)
    assert len(doc_recs) == 2
    assert {r.parsed_metadata.project for r in doc_recs} == {"Alpha", "Beta"}

    # 3. Filter by project + memory_type
    alpha_pref = store.find_by_metadata(project="Alpha", memory_type=MemoryType.PREFERENCE)
    assert len(alpha_pref) == 1
    assert "preference" in alpha_pref[0].text

    # 4. Filter by timestamp range
    time_window = store.find_by_metadata(since_timestamp=t0 + 500, until_timestamp=t2 + 500)
    assert len(time_window) == 2
    assert time_window[0].parsed_metadata.timestamp == t1
    assert time_window[1].parsed_metadata.timestamp == t2

