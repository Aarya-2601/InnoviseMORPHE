"""
Unit tests for TrialScheduler, SelfPlayAuditor, and SleepCycleConsolidator.
"""

import pytest

from morphe.audit.self_play import SelfPlayAuditor
from morphe.audit.sleep_cycle import SleepCycleConsolidator
from morphe.backend.morphe_backend import MorpheBackend
from morphe.models.topology import Mutation
from morphe.policy.trial_scheduler import TrialScheduler
from morphe.router.learned_router import LearnedRouter
from morphe.storage.record_store import RecordStore
from morphe.telemetry.logger import TelemetryLogger


@pytest.fixture
def setup_environment():
    store = RecordStore(":memory:")
    backend = MorpheBackend(record_store=store)

    rec1, _ = store.put("OAuth2 PKCE authorization code security", metadata={"project": "Auth"})
    rec2, _ = store.put("PostgreSQL Patroni high-availability replication", metadata={"project": "DB"})
    rec3, _ = store.put("Prometheus scraping interval 15s latency", metadata={"project": "Telemetry"})

    backend.add_module("mod_auth", charter="Auth Module", record_ids=[rec1.id])
    backend.add_module("mod_db", charter="DB Module", record_ids=[rec2.id])
    backend.add_module("mod_telemetry", charter="Telemetry Module", record_ids=[rec3.id])
    backend.add_edge("mod_auth", "mod_db", weight=0.6)

    telemetry = TelemetryLogger(":memory:")
    router = LearnedRouter(backend=backend, telemetry=telemetry)

    return backend, telemetry, router, [rec1, rec2, rec3]


def test_trial_scheduler_tier0_and_tier3(setup_environment):
    backend, telemetry, router, recs = setup_environment
    scheduler = TrialScheduler(telemetry=telemetry)

    heldout = [
        {"query": "OAuth2 PKCE authorization", "gold_id": recs[0].id, "gold_project": "Auth"},
        {"query": "PostgreSQL replication Patroni", "gold_id": recs[1].id, "gold_project": "DB"},
    ]

    # Tier 0 test: do_nothing should be skipped immediately with 1.0 compute saved!
    mut_skip = Mutation(action_type="do_nothing", target_modules=[])
    res_skip = scheduler.evaluate_mutation(backend, mut_skip, heldout)
    assert res_skip.tier_reached == 0
    assert res_skip.compute_saved_score == 1.0

    # Test shortcut mutation evaluated in sandbox
    mut_sc = Mutation(action_type="shortcut", target_modules=["mod_db", "mod_auth"], parameters={"weight": 1.2})
    res_sc = scheduler.evaluate_mutation(backend, mut_sc, heldout)
    assert res_sc.tier_reached in [2, 3]


def test_self_play_auditor(setup_environment):
    backend, telemetry, router, recs = setup_environment
    auditor = SelfPlayAuditor(backend=backend, router=router)

    stats = auditor.run_audit_cycle(num_probes=2)
    assert stats["probes_tested"] == 2
    assert "pass_rate" in stats


def test_sleep_cycle_consolidation(setup_environment):
    backend, telemetry, router, recs = setup_environment
    consolidator = SleepCycleConsolidator(backend=backend, telemetry=telemetry)

    summary = consolidator.consolidate()
    assert "plasticity" in summary
    assert "timestamp" in summary
