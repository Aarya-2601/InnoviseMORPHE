"""
Tests for MORPHE Object Model:
- Module Manifest (immutable, replaced never mutated in-place)
- Edges with validity windows and types (regular, shortcut, supersedes, hierarchical)
- Topology Snapshots
- Version Commits with metrics and reason cards
"""

import numpy as np
import pytest

from morphe.topology.models import (
    Edge,
    EdgeType,
    ModuleManifest,
    TopologySnapshot,
    VersionCommit,
)


def test_edge_validity_and_temporal_window():
    # Regular active edge
    e1 = Edge(source="mod_a", target="mod_b", weight=0.8, edge_type=EdgeType.REGULAR)
    assert e1.is_active() is True
    assert e1.edge_type == EdgeType.REGULAR

    # Supersedes edge with temporal validity window
    t_start = 1000.0
    t_end = 2000.0
    e2 = Edge(
        source="mod_new",
        target="mod_old",
        weight=1.0,
        edge_type=EdgeType.SUPERSEDES,
        valid_from=t_start,
        valid_to=t_end,
    )
    assert e2.is_active(query_time=500.0) is False   # before window
    assert e2.is_active(query_time=1500.0) is True   # inside window
    assert e2.is_active(query_time=2500.0) is False  # expired window


def test_module_manifest_immutability_and_replacement():
    cent = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    m1 = ModuleManifest.create(
        module_id="mod_auth",
        record_ids=["hash_1", "hash_2", "hash_3"],
        charter="Authentication protocols and tokens",
        centroid=cent,
        tags=["auth", "security"],
    )

    assert m1.record_count == 3
    assert "hash_1" in m1.record_ids
    assert m1.charter == "Authentication protocols and tokens"

    # Invariant: Frozen dataclass must reject direct modification
    with pytest.raises(Exception):
        m1.charter = "Modified charter"  # type: ignore

    # Test replacement copy with new records
    m2 = m1.with_records(new_record_ids=["hash_1", "hash_2", "hash_3", "hash_4"])
    assert m2.record_count == 4
    assert m1.record_count == 3  # original untouched


def test_topology_snapshot_integrity_and_edges():
    m_auth = ModuleManifest.create(
        module_id="mod_auth",
        record_ids=["h1", "h2"],
        charter="Auth Module",
    )
    m_db = ModuleManifest.create(
        module_id="mod_db",
        record_ids=["h3", "h4"],
        charter="Database Module",
    )
    edge = Edge(source="mod_auth", target="mod_db", weight=0.75, edge_type=EdgeType.SHORTCUT)

    snapshot = TopologySnapshot.create(
        modules={"mod_auth": m_auth, "mod_db": m_db},
        edges=[edge],
    )

    assert len(snapshot.snapshot_id) == 16
    assert snapshot.get_module("mod_auth") is not None
    assert len(snapshot.get_outgoing_edges("mod_auth")) == 1
    assert snapshot.get_outgoing_edges("mod_auth")[0].target == "mod_db"
    assert len(snapshot.get_incoming_edges("mod_db")) == 1


def test_version_commit_lineage_and_metrics():
    m_auth = ModuleManifest.create("mod_auth", ["h1"])
    snap = TopologySnapshot.create(modules={"mod_auth": m_auth}, edges=[])

    # Initial version commit (v0)
    v0 = VersionCommit.create(
        snapshot=snap,
        parent_version=None,
        reason_card="Initial bootstrap architecture",
        metrics={"Recall@5": 0.82, "MRR": 0.70, "latency_ms": 12.5, "cost": 1.0},
    )

    assert v0.version_id.startswith("v_")
    assert v0.parent_version is None
    assert v0.metrics["Recall@5"] == 0.82

    # Second version commit (v1) derived from v0
    v1 = VersionCommit.create(
        snapshot=snap,
        parent_version=v0.version_id,
        reason_card="Bandit split overloaded cluster into mod_auth and mod_tokens",
        metrics={"Recall@5": 0.89, "MRR": 0.78, "latency_ms": 10.2, "cost": 1.1},
    )

    assert v1.parent_version == v0.version_id
    assert v1.metrics["Recall@5"] > v0.metrics["Recall@5"]
    assert "split overloaded cluster" in v1.reason_card

    # Immutability check
    with pytest.raises(Exception):
        v1.reason_card = "Tampered reason"  # type: ignore

