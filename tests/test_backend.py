"""
Unit tests for MorpheBackend contract:
- fork()
- apply_mutation(action) [split, merge, shortcut, prune, promote]
- evaluate(queryset)
- commit() and rollback(version_id)
- list_versions()
- diff(v1, v2)
"""

import pytest
import numpy as np

from morphe.backend.morphe_backend import MorpheBackend
from morphe.models.topology import Mutation
from morphe.storage.record_store import RecordStore


@pytest.fixture
def populated_backend():
    """Create a backend with sample records and 2 modules."""
    store = RecordStore(":memory:")
    # Add 4 records via store.put
    rec1, _ = store.put("Authentication security policy OAuth2", metadata={"project": "Auth"})
    rec2, _ = store.put("Password hashing with argon2id", metadata={"project": "Auth"})
    rec3, _ = store.put("Database connection pooling in PostgreSQL", metadata={"project": "DB"})
    rec4, _ = store.put("Index optimization with B-tree", metadata={"project": "DB"})

    backend = MorpheBackend(record_store=store)
    backend.add_module("mod_auth", charter="Authentication & Security", record_ids=[rec1.id, rec2.id])
    backend.add_module("mod_db", charter="Database & Indexing", record_ids=[rec3.id, rec4.id])
    backend.add_edge("mod_auth", "mod_db", weight=0.5, edge_type="associative")
    backend.commit(reason_card="Baseline v1: 2 modules, 1 edge")
    return backend


def test_initial_state_and_commit(populated_backend):
    assert populated_backend.head_version == "v1"
    assert len(populated_backend.graph.nodes) == 2
    assert len(populated_backend.graph.edges) == 1
    versions = populated_backend.list_versions()
    assert len(versions) == 1
    assert versions[0].version_id == "v1"


def test_fork_isolation(populated_backend):
    forked = populated_backend.fork()
    assert forked.head_version == "v1"
    assert len(forked.graph.nodes) == 2

    # Mutate the fork
    forked.apply_mutation(Mutation(action_type="shortcut", target_modules=["mod_db", "mod_auth"], parameters={"weight": 0.9}))
    
    # Fork has 2 edges (or 3 if bidirectional), parent must still have 1 edge!
    assert populated_backend.graph.number_of_edges() == 1
    assert forked.graph.number_of_edges() > 1


def test_apply_split_mutation(populated_backend):
    target = "mod_auth"
    manifest = populated_backend.manifests[target]
    rec_list = list(manifest.record_ids)
    
    split_action = Mutation(
        action_type="split",
        target_modules=[target],
        parameters={
            "sub_module_1": "mod_oauth",
            "sub_module_2": "mod_crypto",
            "charter_1": "OAuth Submodule",
            "charter_2": "Crypto Submodule",
            "split_records": {
                "mod_oauth": [rec_list[0]],
                "mod_crypto": [rec_list[1]],
            }
        },
        reason="Module overloaded",
    )
    success = populated_backend.apply_mutation(split_action)
    assert success is True
    assert "mod_auth" not in populated_backend.manifests
    assert "mod_oauth" in populated_backend.manifests
    assert "mod_crypto" in populated_backend.manifests
    # Edges between new submodules created
    assert populated_backend.graph.has_edge("mod_oauth", "mod_crypto")


def test_apply_merge_mutation(populated_backend):
    merge_action = Mutation(
        action_type="merge",
        target_modules=["mod_auth", "mod_db"],
        parameters={"merged_id": "mod_core", "charter": "Unified Core Services"},
        reason="Modules closely related",
    )
    success = populated_backend.apply_mutation(merge_action)
    assert success is True
    assert "mod_auth" not in populated_backend.manifests
    assert "mod_db" not in populated_backend.manifests
    assert "mod_core" in populated_backend.manifests
    assert len(populated_backend.manifests["mod_core"].record_ids) == 4


def test_apply_shortcut_and_prune(populated_backend):
    # Add shortcut
    shortcut = Mutation(
        action_type="shortcut",
        target_modules=["mod_db", "mod_auth"],
        parameters={"weight": 1.5, "bidirectional": False},
    )
    populated_backend.apply_mutation(shortcut)
    edge = populated_backend.get_edge("mod_db", "mod_auth")
    assert edge is not None
    assert edge.weight == 1.5
    assert edge.edge_type == "shortcut"

    # Prune edge
    prune = Mutation(
        action_type="prune",
        target_modules=[],
        parameters={"edge": ["mod_db", "mod_auth"]},
    )
    populated_backend.apply_mutation(prune)
    assert not populated_backend.graph.has_edge("mod_db", "mod_auth")


def test_rollback_and_diff(populated_backend):
    v1_edges = populated_backend.graph.number_of_edges()

    # Apply mutation and commit v2
    populated_backend.apply_mutation(Mutation(
        action_type="shortcut",
        target_modules=["mod_db", "mod_auth"],
        parameters={"weight": 1.2, "bidirectional": False},
        reason="Frequent co-access",
    ))
    commit_v2 = populated_backend.commit(
        reason_card="Added shortcut edge between DB and Auth",
        metrics={"recall_at_5": 0.95, "latency_ms": 12.0},
    )
    assert commit_v2.version_id == "v2"
    assert populated_backend.head_version == "v2"

    # Verify diff
    d = populated_backend.diff("v1", "v2")
    assert len(d.added_edges) == 1
    assert d.added_edges[0]["source"] == "mod_db"
    assert d.added_edges[0]["target"] == "mod_auth"

    # Rollback to v1
    reverted = populated_backend.rollback("v1")
    assert reverted is True
    assert populated_backend.head_version == "v1"
    assert populated_backend.graph.number_of_edges() == v1_edges
    assert not populated_backend.graph.has_edge("mod_db", "mod_auth")


def test_evaluation(populated_backend):
    target_rec_id = list(populated_backend.manifests["mod_auth"].record_ids)[0]
    queryset = [
        {"query": "Authentication OAuth2 security", "gold_id": target_rec_id, "gold_module": "mod_auth"},
        {"query": "Database PostgreSQL indexing", "gold_module": "mod_db"},
    ]
    metrics = populated_backend.evaluate(queryset)
    assert "recall_at_5" in metrics
    assert "mrr" in metrics
    assert "latency_ms" in metrics
    assert metrics["recall_at_5"] >= 0.5
