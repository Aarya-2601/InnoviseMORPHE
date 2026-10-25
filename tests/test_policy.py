"""
Unit tests for Diagnostician and ArchitectBandit:
- Diagnostician symptom severity calculation (overload, redundancy, shortcut)
- LinUCB arm selection gated by eligibility rules
- Two-level candidate selection
- Regret tracking and online updates
"""

import numpy as np
import pytest

from morphe.backend.morphe_backend import MorpheBackend
from morphe.policy.architect_bandit import ArchitectBandit
from morphe.policy.diagnostician import Diagnostician
from morphe.storage.record_store import RecordStore
from morphe.telemetry.logger import TelemetryLogger


def test_diagnostician_detects_overload_and_redundancy():
    store = RecordStore(":memory:")
    backend = MorpheBackend(record_store=store)
    
    # Create overloaded module with 160 dummy record IDs
    recs = [f"rec_hash_{i}" for i in range(160)]
    backend.add_module("mod_bloated", charter="Bloated Module", record_ids=recs)

    # Create 2 redundant modules with identical centroids
    c = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    m1 = backend.add_module("mod_dup1", charter="Duplicate 1")
    m2 = backend.add_module("mod_dup2", charter="Duplicate 2")
    m1.centroid = c
    m2.centroid = c

    diagnostician = Diagnostician(overload_record_threshold=150, redundancy_cosine_threshold=0.90)
    report = diagnostician.diagnose(backend)

    symptom_types = [s.symptom_type for s in report.symptoms]
    assert "overloaded" in symptom_types
    assert "redundant" in symptom_types
    assert report.primary_symptom is not None


def test_architect_bandit_eligibility_and_ucb_update():
    store = RecordStore(":memory:")
    backend = MorpheBackend(record_store=store)
    
    # Module with 160 records -> Split is eligible
    recs = [f"rec_hash_{i}" for i in range(160)]
    backend.add_module("mod_bloated", record_ids=recs)

    diagnostician = Diagnostician(overload_record_threshold=150)
    bandit = ArchitectBandit(alpha=1.0)

    # Propose mutation
    mutation, context = bandit.propose_mutation(backend, diagnostician)
    assert mutation.action_type in ["split", "do_nothing"]
    
    # Repeatedly reward Split with high positive value
    for _ in range(5):
        bandit.update(arm="split", context=context, reward=2.0, was_accepted=True)

    # Now split should be strongly preferred
    mut2, ctx2 = bandit.propose_mutation(backend, diagnostician)
    assert mut2.action_type == "split"
    assert mut2.target_modules == ["mod_bloated"]


def test_architect_bandit_regret_tracking():
    bandit = ArchitectBandit()
    context = np.array([0.8, 0.5, 0.0, 0.0, 0.0])

    bandit.update("split", context, reward=0.5)
    assert bandit.cumulative_regret > 0.0
    assert len(bandit.history) == 1
