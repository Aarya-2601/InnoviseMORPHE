"""
Unit tests for LearnedRouter (optimal stopping, spreading activation, BM25+Dense RRF)
and PlasticityEngine (Hebbian three-factor reinforcement and decay).
"""

import numpy as np
import pytest

from morphe.backend.morphe_backend import MorpheBackend
from morphe.router.learned_router import LearnedRouter, SimpleBM25
from morphe.router.plasticity import PlasticityEngine
from morphe.storage.record_store import RecordStore
from morphe.telemetry.logger import TelemetryLogger


def test_simple_bm25_ranking():
    corpus = {
        "doc1": "OAuth2 with PKCE verification and security policy",
        "doc2": "PostgreSQL indexing with B-tree and BRIN",
        "doc3": "Prometheus metric scraping interval 15 seconds",
    }
    bm25 = SimpleBM25(corpus)
    scores = bm25.score("OAuth2 PKCE")
    assert scores["doc1"] > scores["doc2"]
    assert scores["doc1"] > scores["doc3"]


def test_learned_router_optimal_stopping():
    store = RecordStore(":memory:")
    backend = MorpheBackend(record_store=store)

    rec1, _ = store.put("OAuth2 code challenge PKCE verification", metadata={"project": "Auth"})
    rec2, _ = store.put("PostgreSQL multi-node Patroni replication", metadata={"project": "DB"})

    backend.add_module("mod_auth", charter="Authentication & OAuth2 PKCE", record_ids=[rec1.id])
    backend.add_module("mod_db", charter="Database & PostgreSQL", record_ids=[rec2.id])
    backend.add_edge("mod_auth", "mod_db", weight=0.5)

    telemetry = TelemetryLogger(":memory:")
    router = LearnedRouter(backend=backend, telemetry=telemetry, margin_threshold=0.20)

    # Specific clear query -> should activate only mod_auth and stop immediately!
    results, trace = router.route_and_retrieve("OAuth2 PKCE verification", gold_id=rec1.id)
    assert len(trace.modules_activated) == 1
    assert trace.modules_activated[0] == "mod_auth"
    assert len(results) > 0
    assert results[0]["id"] == rec1.id
    assert trace.gold_hit is True


def test_plasticity_engine_reinforcement_and_pruning():
    store = RecordStore(":memory:")
    backend = MorpheBackend(record_store=store)

    backend.add_module("mod_1", charter="Module 1")
    backend.add_module("mod_2", charter="Module 2")
    backend.add_module("mod_3", charter="Module 3")

    # Add active edge and cold edge
    backend.add_edge("mod_1", "mod_2", weight=0.5)
    backend.add_edge("mod_1", "mod_3", weight=0.03)  # near epsilon (0.02)

    engine = PlasticityEngine(eta=0.2, decay_lambda=0.5, epsilon=0.02)

    # Simulate cycle where (mod_1, mod_2) co-accessed with positive reward
    co_rewards = {("mod_1", "mod_2"): 1.5}
    summary = engine.apply_cycle(backend, co_rewards)

    # mod_1 -> mod_2 should be strengthened
    new_w = backend.graph["mod_1"]["mod_2"]["weight"]
    assert new_w > 0.5

    # mod_1 -> mod_3 had no reward and was decayed: 0.03 * (1 - 0.5) = 0.015 < epsilon -> PRUNED!
    assert not backend.graph.has_edge("mod_1", "mod_3")
    assert summary["pruned_count"] >= 1
