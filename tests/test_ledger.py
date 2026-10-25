"""
Tests for SQLite VersionLedger with per-version metrics, reason cards, and Pareto frontier.
"""

import pytest
from morphe.storage.ledger import VersionLedger
from morphe.topology.models import ModuleManifest, TopologySnapshot, VersionCommit


def test_version_ledger_record_and_get():
    ledger = VersionLedger(":memory:")
    m = ModuleManifest.create("mod_init", ["r1", "r2"], charter="Init")
    snap = TopologySnapshot.create(modules={"mod_init": m}, edges=[])

    commit = VersionCommit.create(
        snapshot=snap,
        reason_card="Initial bootstrap graph",
        metrics={"Recall@5": 0.85, "MRR": 0.75, "latency_ms": 12.0, "cost": 1.0},
    )

    ledger.record_commit(commit, set_as_head=True)
    assert ledger.count() == 1

    fetched = ledger.get_version(commit.version_id)
    assert fetched is not None
    assert fetched.version_id == commit.version_id
    assert fetched.reason_card == "Initial bootstrap graph"
    assert fetched.metrics["Recall@5"] == 0.85
    assert fetched.metrics["latency_ms"] == 12.0

    head = ledger.get_head()
    assert head is not None
    assert head.version_id == commit.version_id


def test_version_ledger_history_and_head_switch():
    ledger = VersionLedger(":memory:")
    m1 = ModuleManifest.create("m1", ["r1"])
    snap = TopologySnapshot.create(modules={"m1": m1}, edges=[])

    v1 = VersionCommit.create(snapshot=snap, reason_card="Commit 1", metrics={"Recall@5": 0.80})
    v2 = VersionCommit.create(snapshot=snap, parent_version=v1.version_id, reason_card="Commit 2", metrics={"Recall@5": 0.88})

    ledger.record_commit(v1, set_as_head=True)
    ledger.record_commit(v2, set_as_head=True)

    assert ledger.get_head().version_id == v2.version_id
    history = ledger.list_history()
    assert len(history) == 2
    assert history[0]["version_id"] == v1.version_id
    assert history[1]["version_id"] == v2.version_id

    # Switch HEAD (rollback)
    success = ledger.set_head(v1.version_id)
    assert success is True
    assert ledger.get_head().version_id == v1.version_id


def test_version_ledger_pareto_frontier():
    ledger = VersionLedger(":memory:")
    m = ModuleManifest.create("m1", ["r1"])
    snap = TopologySnapshot.create(modules={"m1": m}, edges=[])

    # vA: low latency, moderate recall
    vA = VersionCommit.create(
        snapshot=snap,
        reason_card="Fast profile",
        metrics={"Recall@5": 0.80, "latency_ms": 5.0, "cost": 1.0},
    )
    # vB: high latency, higher recall (non-dominated trade-off)
    vB = VersionCommit.create(
        snapshot=snap,
        reason_card="Accurate profile",
        metrics={"Recall@5": 0.95, "latency_ms": 20.0, "cost": 1.5},
    )
    # vC: strictly worse than vA (lower recall, worse latency, higher cost -> DOMINATED)
    vC = VersionCommit.create(
        snapshot=snap,
        reason_card="Worse profile",
        metrics={"Recall@5": 0.70, "latency_ms": 10.0, "cost": 2.0},
    )

    ledger.record_commit(vA)
    ledger.record_commit(vB)
    ledger.record_commit(vC)

    frontier = ledger.get_pareto_frontier()
    frontier_vids = {f["version_id"] for f in frontier}

    assert vA.version_id in frontier_vids
    assert vB.version_id in frontier_vids
    assert vC.version_id not in frontier_vids  # Dominated candidate eliminated


def test_version_ledger_disk_persistence(tmp_path):
    db_file = tmp_path / "ledger_test.db"
    ledger1 = VersionLedger(db_file)
    m = ModuleManifest.create("m1", ["r1"])
    snap = TopologySnapshot.create(modules={"m1": m}, edges=[])
    v = VersionCommit.create(snapshot=snap, reason_card="Persisted version", metrics={"MRR": 0.9})
    ledger1.record_commit(v)
    ledger1.close()

    # Re-open from disk
    ledger2 = VersionLedger(db_file)
    assert ledger2.count() == 1
    loaded_v = ledger2.get_version(v.version_id)
    assert loaded_v is not None
    assert loaded_v.reason_card == "Persisted version"
    assert loaded_v.metrics["MRR"] == 0.9
    ledger2.close()

