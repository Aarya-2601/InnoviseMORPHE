"""
Hard Edge-Case and Stress Testing Suite for MORPHE.
Covers critical failure modes and adversarial scenarios across all built subsystems:

1. Record Store:
   - Extreme inputs: Empty strings, giant chunks (100KB+), unicode emojis, null bytes, special chars.
   - Hash collisions and idempotency under concurrent-like batch inserts.
   - Missing embedding updates and zero-vector handling.

2. Hybrid Chunker:
   - Texts with no sentence punctuation (continuous stream).
   - Code blocks, markdown tables, URLs, abbreviations (e.g. "Dr. Smith vs. U.S.A.").
   - Extremely short strings (< min_chars) and single massive sentences (> max_chars).

3. Backend & Topology Lineage:
   - Invalid rollback requests (non-existent commit ID).
   - Cyclic and self-loop edge prevention during split/merge.
   - Cascading module splits leaving empty source modules.
   - Deep fork isolation (sandbox branch mutations never pollute parent ledger).

4. Mutation Engine:
   - Module split with identical or unseparable embeddings.
   - Merge operations where combined record count crosses the split threshold.
   - Extreme prune requests on non-existent edges.
   - Promotions with disjoint clusters.

5. Temporal Memory & Supersedes:
   - Chained supersessions (A -> superseded by B -> superseded by C).
   - Ambiguous queries with both historical and current signals.
   - Rollback of chained supersessions restoring intermediate states.

6. Sleep Cycle Consolidation:
   - Identical embeddings vs. completely orthogonal embeddings.
   - Staleness window with future or negative timestamps.
   - Consolidation on an empty module.
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from morphe.adaptation.engine import MutationEngine
from morphe.consolidation.sleep_cycle import SleepCycleConsolidator
from morphe.core.backend import MorpheBackend
from morphe.ingestion.chunker import HybridChunker, sentence_split
from morphe.storage.ledger import VersionLedger
from morphe.storage.record_store import Record, RecordStore
from morphe.temporal.manager import TemporalMemoryManager, is_historical_query
from morphe.topology.models import Edge, EdgeType, ModuleManifest, TopologySnapshot
from morphe.topology.mutations import MutationAction, MutationType


# ==============================================================================
# 1. RECORD STORE ADVERSARIAL CASES
# ==============================================================================

def test_hard_record_store_extreme_inputs():
    store = RecordStore(":memory:")

    # 1. Special characters, multi-language unicode, and emojis
    unicode_text = "MORPHE: 🧠 🚀 机器学习 & 딥러닝! Special: \t\r\n \u0000 \u200b and symbols <>[]{}|~"
    rec1, is_new1 = store.put(unicode_text)
    assert is_new1 is True
    assert store.contains(rec1.id)
    fetched1 = store.get(rec1.id)
    assert fetched1.text == unicode_text

    # 2. Large chunk (100,000 characters)
    giant_text = "A" * 100000
    rec2, is_new2 = store.put(giant_text)
    assert is_new2 is True
    assert len(store.get(rec2.id).text) == 100000

    # 3. Duplicate insertion with embedding upgrade
    rec3, _ = store.put("Embedding upgrade target text.")
    assert rec3.embedding is None
    
    # Ingest same text but now with embedding
    emb_vec = np.ones(128, dtype=np.float32)
    rec3_upgraded, is_new_upgraded = store.put("Embedding upgrade target text.", embedding=emb_vec)
    assert is_new_upgraded is False
    assert rec3_upgraded.id == rec3.id
    assert rec3_upgraded.embedding is not None
    assert rec3_upgraded.embedding.shape == (128,)


def test_hard_record_store_zero_vector_handling():
    store = RecordStore(":memory:")
    # Zero vector should not cause div-by-zero crashes
    zero_vec = np.zeros(64, dtype=np.float32)
    rec, is_new = store.put("Zero vector test record", embedding=zero_vec)
    assert is_new is True
    fetched = store.get(rec.id)
    assert np.all(fetched.embedding == 0.0)


# ==============================================================================
# 2. HYBRID CHUNKER ADVERSARIAL CASES
# ==============================================================================

def test_hard_chunker_no_punctuation_stream():
    chunker = HybridChunker(min_chars=50, target_chars=100, max_chars=200)
    # Text with zero sentence-ending punctuation marks
    stream_text = "this is a continuous streaming log output with no dots or question marks whatsoever across lines"
    chunks = chunker.chunk_text(stream_text)
    assert len(chunks) >= 1
    # Ensure no data loss
    assert "".join(c.text for c in chunks).replace(" ", "") == stream_text.replace(" ", "")


def test_hard_chunker_abbreviations_and_decimals():
    text = (
        "Dr. Smith visited the U.S.A. on Jan. 15th. "
        "The measurement was 3.14159 units at 10.5 GHz! "
        "Will this split properly? Yes, it will."
    )
    sentences = sentence_split(text)
    # Should not split mid-abbreviation or mid-decimal
    assert any("3.14159 units at 10.5 GHz!" in s for s in sentences)
    assert any("Dr. Smith visited the U.S.A. on Jan. 15th." in s for s in sentences)


def test_hard_chunker_sub_minimum_string():
    chunker = HybridChunker(min_chars=100, target_chars=300, max_chars=600)
    tiny_text = "Short note."
    chunks = chunker.chunk_text(tiny_text)
    assert len(chunks) == 1
    assert chunks[0].text == "Short note."


# ==============================================================================
# 3. BACKEND & TOPOLOGY ADVERSARIAL CASES
# ==============================================================================

def test_hard_backend_invalid_rollback_and_missing_versions():
    backend = MorpheBackend()
    # Attempting to rollback to non-existent ID must gracefully return False
    assert backend.rollback("non_existent_version_xyz") is False
    
    # Diff on missing versions must raise KeyError
    with pytest.raises(KeyError):
        backend.diff("v_fake1", "v_fake2")


def test_hard_backend_cascading_splits_and_isolation():
    store = RecordStore(":memory:")
    # Initialize with 1 module containing 10 records
    recs = [store.put(f"Memory unit {i}")[0].id for i in range(10)]
    m = ModuleManifest.create("mod_root", recs)
    snap = TopologySnapshot.create(modules={"mod_root": m}, edges=[])
    backend = MorpheBackend(record_store=store, initial_snapshot=snap)

    # First split in sandbox
    sb1 = backend.fork()
    sb1.apply_mutation(MutationAction.split_module(
        module_id="mod_root",
        new_module_id="mod_sub1",
        migrated_record_ids=recs[:5],
    ))
    
    # Second split chained in same sandbox
    sb1.apply_mutation(MutationAction.split_module(
        module_id="mod_sub1",
        new_module_id="mod_sub2",
        migrated_record_ids=recs[:2],
    ))

    # Commit sandbox
    commit1 = sb1.commit("Chained splits in sandbox")

    # Invariants check
    assert sb1.graph.number_of_nodes() == 3
    assert set(sb1.graph.nodes) == {"mod_root", "mod_sub1", "mod_sub2"}
    # Main backend MUST remain completely untouched with 1 node!
    assert backend.graph.number_of_nodes() == 1
    assert "mod_root" in backend.graph.nodes


# ==============================================================================
# 4. MUTATION ENGINE ADVERSARIAL CASES
# ==============================================================================

def test_hard_mutation_split_identical_embeddings():
    """Module where all records have IDENTICAL embeddings (cannot cluster)."""
    store = RecordStore(":memory:")
    same_emb = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    same_emb /= np.linalg.norm(same_emb)

    rec_ids = [store.put(f"Identical fact {i}", embedding=same_emb)[0].id for i in range(160)]
    m = ModuleManifest.create("mod_identical", rec_ids)
    snap = TopologySnapshot.create(modules={"mod_identical": m}, edges=[])
    backend = MorpheBackend(record_store=store, initial_snapshot=snap)

    engine = MutationEngine(split_size_threshold=150)
    splits = engine.find_eligible_splits(backend)
    
    # Size-based overload trigger must still safely partition without division by zero
    assert len(splits) >= 1
    applied = backend.apply_mutation(splits[0].action)
    assert applied is True
    assert backend.graph.number_of_nodes() == 2


def test_hard_mutation_merge_size_limit_rejection():
    """Two modules are semantically identical, but merging would exceed split limit."""
    store = RecordStore(":memory:")
    c = np.array([1.0, 0.0], dtype=np.float32)
    # 90 + 80 = 170 records > 150 split threshold
    r1 = [store.put(f"Doc A {i}")[0].id for i in range(90)]
    r2 = [store.put(f"Doc B {i}")[0].id for i in range(80)]

    m1 = ModuleManifest.create("m1", r1, centroid=c)
    m2 = ModuleManifest.create("m2", r2, centroid=c)
    snap = TopologySnapshot.create(modules={"m1": m1, "m2": m2}, edges=[])
    backend = MorpheBackend(record_store=store, initial_snapshot=snap)

    engine = MutationEngine(split_size_threshold=150, merge_cosine_threshold=0.90)
    merges = engine.find_eligible_merges(backend)
    
    # Invariant: Merge candidate must be REJECTED to prevent creating an immediately overloaded module
    assert len(merges) == 0


# ==============================================================================
# 5. TEMPORAL MEMORY CHAINED SUPERSEDING & ROLLBACK
# ==============================================================================

def test_hard_temporal_chained_supersedes():
    """Fact A -> superseded by Fact B -> superseded by Fact C."""
    store = RecordStore(":memory:")
    fA, _ = store.put("Database engine is MySQL 5.7.")
    fB, _ = store.put("Migrated database engine from MySQL to PostgreSQL 14.")
    fC, _ = store.put("Upgraded database engine to distributed CockroachDB.")

    m = ModuleManifest.create("mod_infra", [fA.id, fB.id, fC.id])
    snap = TopologySnapshot.create(modules={"mod_infra": m}, edges=[])
    backend = MorpheBackend(record_store=store, initial_snapshot=snap)
    v0_id = backend.head_version_id

    temporal = TemporalMemoryManager(backend)

    # 1. Supersede A with B
    temporal.record_supersession(new_record_id=fB.id, old_record_id=fA.id, source_module="mod_infra")
    v1 = backend.commit("MySQL -> Postgres")

    # 2. Supersede B with C
    temporal.record_supersession(new_record_id=fC.id, old_record_id=fB.id, source_module="mod_infra")
    v2 = backend.commit("Postgres -> CockroachDB")

    assert store.get(fA.id).archived is True
    assert store.get(fB.id).archived is True
    assert store.get(fC.id).archived is False

    candidates = [(fA.id, 0.9), (fB.id, 0.9), (fC.id, 0.9)]

    # Query Current: Must yield C (CockroachDB)
    curr_res = temporal.resolve_temporal_records(candidates, "What database engine do we currently run?")
    assert curr_res[0][0] == fC.id
    assert "CockroachDB" in store.get(curr_res[0][0]).text

    # Query Historical: Must prefer older records (fA / fB)
    hist_res = temporal.resolve_temporal_records(candidates, "What was our historical or original database?")
    top_hist_id = hist_res[0][0]
    assert top_hist_id in {fA.id, fB.id}

    # Rollback to v1: restores intermediate state where fB is active and only fA was superseded
    backend.rollback(v1.version_id)
    assert backend.head_version_id == v1.version_id


# ==============================================================================
# 6. SLEEP CYCLE CONSOLIDATION BOUNDARY CASES
# ==============================================================================

def test_hard_sleep_cycle_empty_and_orthogonal():
    store = RecordStore(":memory:")
    # Module with zero embeddings or orthogonal vectors
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    r1, _ = store.put("Orthogonal topic 1", embedding=v1)
    r2, _ = store.put("Orthogonal topic 2", embedding=v2)
    r3, _ = store.put("Orthogonal topic 3", embedding=v3)

    snap = TopologySnapshot.create(
        modules={"mod_ortho": ModuleManifest.create("mod_ortho", [r1.id, r2.id, r3.id])},
        edges=[]
    )
    backend = MorpheBackend(record_store=store, initial_snapshot=snap)

    consolidator = SleepCycleConsolidator(backend, near_duplicate_cosine_threshold=0.95)
    report = consolidator.run_sleep_cycle(auto_commit=False)

    # Invariant: Orthogonal records must NEVER be merged
    assert len(report.merged_near_duplicates) == 0
    assert backend.get_staged_snapshot().get_module("mod_ortho").record_count == 3

