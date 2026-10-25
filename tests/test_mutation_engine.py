"""
Tests for Mutation Engine: Split, Merge, Shortcut, Prune, Promote.
Verifies Section 5.1 & 5.1a eligibility rules and execution.
"""

import numpy as np
import pytest

from morphe.adaptation.engine import MutationEngine
from morphe.core.backend import MorpheBackend
from morphe.storage.record_store import Record, RecordStore
from morphe.topology.models import Edge, EdgeType, ModuleManifest, TopologySnapshot
from morphe.topology.mutations import MutationAction, MutationType


def test_mutation_engine_split_eligibility_and_execution():
    # Setup record store with an overloaded module (> split_threshold)
    store = RecordStore(":memory:")
    rec_ids = []
    # Create 160 records to cross the 150 split threshold
    for i in range(160):
        # Generate two synthetic clusters based on prefix
        vec = np.zeros(16, dtype=np.float32)
        if i < 80:
            vec[0] = 1.0  # Cluster A
            text = f"Cluster A document item {i}"
        else:
            vec[1] = 1.0  # Cluster B
            text = f"Cluster B document item {i}"
        rec, _ = store.put(text=text, embedding=vec)
        rec_ids.append(rec.id)

    manifest = ModuleManifest.create("mod_overloaded", rec_ids, charter="Overloaded multi-topic module")
    snap = TopologySnapshot.create(modules={"mod_overloaded": manifest}, edges=[])
    backend = MorpheBackend(record_store=store, initial_snapshot=snap)

    engine = MutationEngine(split_size_threshold=150)
    split_candidates = engine.find_eligible_splits(backend)

    assert len(split_candidates) >= 1
    top_split = split_candidates[0]
    assert top_split.action.mutation_type == MutationType.SPLIT_MODULE
    assert top_split.action.source_module == "mod_overloaded"
    assert top_split.symptom == "overloaded"

    # Execute split in sandbox
    sandbox = backend.fork()
    applied = sandbox.apply_mutation(top_split.action)
    assert applied is True
    assert sandbox.graph.number_of_nodes() == 2
    assert "mod_overloaded" in sandbox.graph.nodes
    assert "mod_overloaded_sub1" in sandbox.graph.nodes


def test_mutation_engine_merge_eligibility_and_execution():
    store = RecordStore(":memory:")
    # Two redundant modules with high centroid cosine similarity
    c1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    c2 = np.array([0.98, 0.1, 0.0], dtype=np.float32)
    c2 /= np.linalg.norm(c2)

    m1 = ModuleManifest.create("mod_1", ["r1", "r2"], charter="Auth tokens", centroid=c1)
    m2 = ModuleManifest.create("mod_2", ["r3", "r4"], charter="Token authentication", centroid=c2)

    snap = TopologySnapshot.create(modules={"mod_1": m1, "mod_2": m2}, edges=[])
    backend = MorpheBackend(record_store=store, initial_snapshot=snap)

    engine = MutationEngine(merge_cosine_threshold=0.90)
    merge_candidates = engine.find_eligible_merges(backend)

    assert len(merge_candidates) == 1
    cand = merge_candidates[0]
    assert cand.action.mutation_type == MutationType.MERGE_MODULES
    assert cand.symptom == "redundant"
    assert cand.evidence["cosine_similarity"] >= 0.90

    # Execute merge
    backend.apply_mutation(cand.action)
    assert backend.graph.number_of_nodes() == 1
    assert "mod_1" in backend.graph.nodes  # target module retained
    assert "mod_2" not in backend.graph.nodes


def test_mutation_engine_shortcut_eligibility_and_execution():
    store = RecordStore(":memory:")
    m1 = ModuleManifest.create("mod_search", ["r1"], charter="Search")
    m2 = ModuleManifest.create("mod_analytics", ["r2"], charter="Analytics")

    snap = TopologySnapshot.create(modules={"mod_search": m1, "mod_analytics": m2}, edges=[])
    backend = MorpheBackend(record_store=store, initial_snapshot=snap)

    engine = MutationEngine(shortcut_coaccess_threshold=10)

    # Co-access history without direct edge
    coaccess = {("mod_search", "mod_analytics"): 15}
    candidates = engine.find_eligible_shortcuts(backend, coaccess_counts=coaccess)

    assert len(candidates) == 1
    sc = candidates[0]
    assert sc.action.mutation_type == MutationType.ADD_EDGE
    assert sc.action.edge_type == EdgeType.SHORTCUT
    assert sc.action.source_module == "mod_search"
    assert sc.action.target_module == "mod_analytics"

    # Execute shortcut
    backend.apply_mutation(sc.action)
    assert backend.graph.has_edge("mod_search", "mod_analytics")


def test_mutation_engine_prune_eligibility_and_execution():
    store = RecordStore(":memory:")
    m1 = ModuleManifest.create("mod_a", ["r1"])
    m2 = ModuleManifest.create("mod_b", ["r2"])
    cold_edge = Edge("mod_a", "mod_b", weight=0.04)  # below epsilon 0.10

    snap = TopologySnapshot.create(modules={"mod_a": m1, "mod_b": m2}, edges=[cold_edge])
    backend = MorpheBackend(record_store=store, initial_snapshot=snap)

    engine = MutationEngine(prune_epsilon=0.10)
    prune_cands = engine.find_eligible_prunes(backend)

    assert len(prune_cands) == 1
    cand = prune_cands[0]
    assert cand.action.mutation_type == MutationType.REMOVE_EDGE
    assert cand.symptom == "cold_edge"

    # Execute prune
    backend.apply_mutation(cand.action)
    assert not backend.graph.has_edge("mod_a", "mod_b")


def test_mutation_engine_promote_hierarchical_cluster():
    store = RecordStore(":memory:")
    m1 = ModuleManifest.create("mod_x", ["r1"])
    m2 = ModuleManifest.create("mod_y", ["r2"])
    m3 = ModuleManifest.create("mod_z", ["r3"])

    snap = TopologySnapshot.create(modules={"mod_x": m1, "mod_y": m2, "mod_z": m3}, edges=[])
    backend = MorpheBackend(record_store=store, initial_snapshot=snap)

    engine = MutationEngine(shortcut_coaccess_threshold=5, promote_min_cluster_size=3)

    # Persistent co-access connecting all 3
    coaccess = {
        ("mod_x", "mod_y"): 8,
        ("mod_y", "mod_z"): 10,
    }

    promote_cands = engine.find_eligible_promotions(backend, coaccess_counts=coaccess)
    assert len(promote_cands) >= 1
    prom = promote_cands[0]
    assert prom.action.mutation_type == MutationType.ADD_MODULE
    assert prom.symptom == "cluster_convergence"
    assert prom.evidence["cluster_size"] == 3

    # Execute promotion
    backend.apply_mutation(prom.action)
    assert prom.action.new_module_id in backend.graph.nodes

