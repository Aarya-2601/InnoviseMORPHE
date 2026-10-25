"""
Tests for Copy-on-Write Sandbox Fork, Commit, Rollback, and Diff.
Directly implements Contract Stub test from Step 1 of Section 14A.
"""

import pytest

from morphe.core.backend import MorpheBackend
from morphe.topology.models import Edge, EdgeType, ModuleManifest, TopologySnapshot
from morphe.topology.mutations import MutationAction, MutationType


@pytest.fixture
def initial_backend():
    # Module A: Auth, Module B: Database
    mA = ModuleManifest.create("mod_auth", ["rec_1", "rec_2"], charter="Auth module")
    mB = ModuleManifest.create("mod_db", ["rec_3", "rec_4"], charter="Database module")
    edge = Edge("mod_auth", "mod_db", weight=0.5, edge_type=EdgeType.REGULAR)

    snap0 = TopologySnapshot.create(modules={"mod_auth": mA, "mod_db": mB}, edges=[edge])
    backend = MorpheBackend(initial_snapshot=snap0)
    return backend


def test_commit_and_list_versions(initial_backend):
    backend = initial_backend
    assert len(backend.list_versions()) == 1
    root_id = backend.head_version_id

    # Apply mutation: Add shortcut edge
    action = MutationAction.add_edge("mod_db", "mod_auth", weight=0.9, edge_type=EdgeType.SHORTCUT)
    success = backend.apply_mutation(action)
    assert success is True

    # Commit as v1
    v1 = backend.commit(
        reason_card="Added reverse shortcut edge based on co-access pattern",
        metrics={"Recall@5": 0.88, "MRR": 0.77, "latency_ms": 9.4, "cost": 1.05},
    )

    assert v1.parent_version == root_id
    assert backend.head_version_id == v1.version_id

    history = backend.list_versions()
    assert len(history) == 2
    assert history[0]["version_id"] == root_id
    assert history[1]["version_id"] == v1.version_id
    assert history[1]["is_head"] is True


def test_rollback(initial_backend):
    backend = initial_backend
    root_id = backend.head_version_id
    assert backend.graph.number_of_edges() == 1

    # Mutate and commit v1 (adds edge)
    backend.apply_mutation(MutationAction.add_edge("mod_db", "mod_auth", weight=0.9))
    v1 = backend.commit("Added second edge", metrics={"Recall@5": 0.85})
    assert backend.graph.number_of_edges() == 2

    # Mutate and commit v2 (adds module C)
    backend.apply_mutation(MutationAction(
        mutation_type=MutationType.ADD_MODULE,
        new_module_id="mod_cache",
        record_ids=("rec_5",),
        charter="Cache module",
    ))
    v2 = backend.commit("Added cache module", metrics={"Recall@5": 0.90})
    assert backend.graph.number_of_nodes() == 3

    # Rollback to root_id
    rollback_ok = backend.rollback(root_id)
    assert rollback_ok is True
    assert backend.head_version_id == root_id
    assert backend.graph.number_of_nodes() == 2
    assert backend.graph.number_of_edges() == 1
    assert "mod_cache" not in backend.graph.nodes


def test_copy_on_write_fork_isolation(initial_backend):
    backend = initial_backend
    root_head = backend.head_version_id

    # Create sandbox fork
    sandbox = backend.fork()
    assert sandbox.head_version_id == root_head
    assert sandbox.graph.number_of_nodes() == 2

    # Mutate the sandbox
    sandbox.apply_mutation(MutationAction.add_edge("mod_db", "mod_auth", weight=0.88))
    sandbox_commit = sandbox.commit("Trial mutation in sandbox", metrics={"Recall@5": 0.92})

    # Invariant: Sandbox has 2 edges, but original backend remains UNTOUCHED with 1 edge!
    assert sandbox.graph.number_of_edges() == 2
    assert backend.graph.number_of_edges() == 1
    assert backend.head_version_id == root_head
    assert len(backend.list_versions()) == 1


def test_diff_between_versions(initial_backend):
    backend = initial_backend
    root_id = backend.head_version_id

    # Mutate:
    # 1. Split mod_auth: rec_2 goes into new module mod_tokens
    backend.apply_mutation(MutationAction.split_module(
        module_id="mod_auth",
        new_module_id="mod_tokens",
        migrated_record_ids=["rec_2"],
        new_module_charter="Token management",
    ))
    # 2. Update weight of edge mod_auth -> mod_db
    backend.apply_mutation(MutationAction.update_edge_weight("mod_auth", "mod_db", weight=0.95))

    v1 = backend.commit(
        reason_card="Split auth and boosted edge weight",
        metrics={"Recall@5": 0.91, "MRR": 0.81},
    )

    diff_result = backend.diff(root_id, v1.version_id)
    assert diff_result.is_empty is False
    assert "mod_tokens" in diff_result.added_modules
    assert "mod_auth" in diff_result.module_changes
    assert "rec_2" in diff_result.module_changes["mod_auth"].removed_records

    # Edge weight change detected
    assert len(diff_result.modified_edges) == 1
    assert diff_result.modified_edges[0].source == "mod_auth"
    assert diff_result.modified_edges[0].target == "mod_db"
    assert pytest.approx(diff_result.modified_edges[0].old_weight) == 0.5
    assert pytest.approx(diff_result.modified_edges[0].new_weight) == 0.95

