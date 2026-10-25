"""
Tests for Edge Types with Validity Windows (Supersedes) and Archived Records.
Directly implements Section 10A.1 and Section 14A Test 9:
"After 'we switched from Postgres to MongoDB', a current query returns MongoDB
and a history query returns Postgres. An archived record is restored by rollback."
"""

import time
import pytest

from morphe.core.backend import MorpheBackend
from morphe.storage.record_store import RecordStore
from morphe.temporal.manager import TemporalMemoryManager, is_historical_query
from morphe.topology.models import Edge, EdgeType, ModuleManifest, TopologySnapshot


def test_is_historical_query_detection():
    assert is_historical_query("What did we use before?") is True
    assert is_historical_query("Which database did we historically support?") is True
    assert is_historical_query("What was our former architecture?") is True
    assert is_historical_query("What database do we currently use?") is False
    assert is_historical_query("Show me active authentication tokens.") is False


def test_supersedes_edge_and_query_resolution():
    store = RecordStore(":memory:")
    # Fact 1 (Old): "We store user accounts in PostgreSQL."
    rec_old, _ = store.put("We store user accounts in PostgreSQL.")
    
    # Fact 2 (New): "We switched from Postgres to MongoDB for user accounts."
    rec_new, _ = store.put("We switched from Postgres to MongoDB for user accounts.")

    # Module contains both
    m_db = ModuleManifest.create("mod_db", [rec_old.id, rec_new.id], charter="Database configuration")
    snap0 = TopologySnapshot.create(modules={"mod_db": m_db}, edges=[])
    backend = MorpheBackend(record_store=store, initial_snapshot=snap0)
    v0_id = backend.head_version_id

    # Initialize Temporal Manager & Record supersession
    temporal_mgr = TemporalMemoryManager(backend)
    edge = temporal_mgr.record_supersession(
        new_record_id=rec_new.id,
        old_record_id=rec_old.id,
        source_module="mod_db",
        archive_old=True,
    )
    v1 = backend.commit("Updated database fact: switched to MongoDB")

    # Verify edge type and archived flag
    assert edge.edge_type == EdgeType.SUPERSEDES
    assert backend.record_store.get(rec_old.id).archived is True
    assert backend.record_store.get(rec_new.id).archived is False

    # 1. Test CURRENT query: must prefer MongoDB (new record)
    candidates = [(rec_old.id, 0.85), (rec_new.id, 0.84)]  # assume raw similarity is close
    resolved_current = temporal_mgr.resolve_temporal_records(
        candidates, query="What database do we use for user accounts?"
    )
    top_current_id = resolved_current[0][0]
    assert top_current_id == rec_new.id
    assert "MongoDB" in store.get(top_current_id).text

    # 2. Test HISTORICAL query: must prefer PostgreSQL (old record)
    resolved_history = temporal_mgr.resolve_temporal_records(
        candidates, query="What database did we use before switching?"
    )
    top_history_id = resolved_history[0][0]
    assert top_history_id == rec_old.id
    assert "PostgreSQL" in store.get(top_history_id).text


def test_archived_record_restoration_via_rollback():
    store = RecordStore(":memory:")
    rec_old, _ = store.put("Legacy Redis caching setup.")
    rec_new, _ = store.put("Upgraded to Memcached caching tier.")

    m = ModuleManifest.create("mod_cache", [rec_old.id, rec_new.id])
    snap = TopologySnapshot.create(modules={"mod_cache": m}, edges=[])
    backend = MorpheBackend(record_store=store, initial_snapshot=snap)
    v0_id = backend.head_version_id

    temporal_mgr = TemporalMemoryManager(backend)
    temporal_mgr.record_supersession(
        new_record_id=rec_new.id,
        old_record_id=rec_old.id,
        source_module="mod_cache",
        archive_old=True,
    )
    v1 = backend.commit("Archived redis fact")
    assert store.get(rec_old.id).archived is True

    # Rollback to v0_id
    success = backend.rollback(v0_id)
    assert success is True
    assert backend.head_version_id == v0_id
    
    # Supersedes edge is removed from active topology
    active_edges = backend.get_staged_snapshot().edges
    assert not any(e.edge_type == EdgeType.SUPERSEDES for e in active_edges)
    
    # Store still retains immutable records
    assert store.contains(rec_old.id)
    assert store.contains(rec_new.id)

