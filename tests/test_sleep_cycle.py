"""
Tests for Sleep-Cycle Backend:
- Near-duplicate record merging
- Stale-memory down-weighting & soft-archiving
- Compacted module manifests
"""

import time
import numpy as np
import pytest

from morphe.consolidation.sleep_cycle import SleepCycleConsolidator
from morphe.core.backend import MorpheBackend
from morphe.storage.record_store import RecordStore
from morphe.topology.models import ModuleManifest, TopologySnapshot


def test_near_duplicate_merging_in_sleep_cycle():
    store = RecordStore(":memory:")
    # 2 near-identical records (cosine ~ 0.99)
    v1 = np.array([1.0, 0.05, 0.0], dtype=np.float32)
    v1 /= np.linalg.norm(v1)

    v2 = np.array([1.0, 0.04, 0.0], dtype=np.float32)
    v2 /= np.linalg.norm(v2)

    # 1 distinct record
    v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    r1, _ = store.put("User authentication uses OAuth2 with JWT tokens.", embedding=v1)
    r2, _ = store.put("User auth employs OAuth2 bearer JWT tokens.", embedding=v2)
    r3, _ = store.put("PostgreSQL manages transactional data schemas.", embedding=v3)

    m = ModuleManifest.create("mod_mixed", [r1.id, r2.id, r3.id], charter="Mixed module")
    snap = TopologySnapshot.create(modules={"mod_mixed": m}, edges=[])
    backend = MorpheBackend(record_store=store, initial_snapshot=snap)

    consolidator = SleepCycleConsolidator(backend, near_duplicate_cosine_threshold=0.98)
    report = consolidator.run_sleep_cycle(auto_commit=True)

    # Check report
    assert len(report.merged_near_duplicates) == 1
    pair = report.merged_near_duplicates[0]
    assert pair.primary_record_id == r1.id
    assert pair.duplicate_record_id == r2.id
    assert pair.similarity >= 0.98

    # Invariants check
    # 1. Duplicate record is soft-archived
    assert store.get(r2.id).archived is True
    assert store.get(r1.id).archived is False
    assert store.get(r3.id).archived is False

    # 2. Module manifest is compacted (r2 removed from active manifest)
    updated_manifest = backend.get_staged_snapshot().get_module("mod_mixed")
    assert r1.id in updated_manifest.record_ids
    assert r3.id in updated_manifest.record_ids
    assert r2.id not in updated_manifest.record_ids
    assert updated_manifest.record_count == 2


def test_stale_memory_downweighting_and_archiving():
    store = RecordStore(":memory:")
    r_fresh, _ = store.put("Fresh interaction memory from today.")
    r_stale, _ = store.put("Stale interaction memory from 30 days ago.")

    m = ModuleManifest.create("mod_interactions", [r_fresh.id, r_stale.id])
    snap = TopologySnapshot.create(modules={"mod_interactions": m}, edges=[])
    backend = MorpheBackend(record_store=store, initial_snapshot=snap)

    now = 1700000000.0
    access_map = {
        r_fresh.id: now - 3600.0,            # 1 hour ago
        r_stale.id: now - (20 * 86400.0),    # 20 days ago
    }

    # Staleness threshold = 7 days
    consolidator = SleepCycleConsolidator(
        backend,
        staleness_threshold_seconds=7 * 86400.0,
    )

    report = consolidator.run_sleep_cycle(last_accessed_map=access_map, now=now, auto_commit=True)

    assert len(report.downweighted_stale_records) == 1
    stale_rec = report.downweighted_stale_records[0]
    assert stale_rec.record_id == r_stale.id

    # Verify stale record was archived
    assert store.get(r_stale.id).archived is True
    assert store.get(r_fresh.id).archived is False

    # Manifest retains only fresh record
    mod = backend.get_staged_snapshot().get_module("mod_interactions")
    assert mod.record_count == 1
    assert r_fresh.id in mod.record_ids
    assert r_stale.id not in mod.record_ids

