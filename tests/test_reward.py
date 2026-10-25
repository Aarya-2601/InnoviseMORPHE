"""
Unit tests for reward computation, statistical acceptance gate, and watchdog.
"""

import numpy as np
import pytest

from morphe.eval.reward import (
    AcceptanceResult,
    PostCommitWatchdog,
    RewardWeights,
    acceptance_gate,
    compute_reward,
)


def test_compute_reward_accurate_vs_inaccurate():
    gold_id = "target_hash_123"
    accurate_results = [{"id": gold_id}, {"id": "other_1"}]
    inaccurate_results = [{"id": "wrong_1"}, {"id": "wrong_2"}]

    r_good = compute_reward(accurate_results, gold_id, latency_ms=10.0, modules_touched=1, structural_cost=2.0)
    r_bad = compute_reward(inaccurate_results, gold_id, latency_ms=150.0, modules_touched=4, structural_cost=10.0)

    assert r_good > 0.0
    assert r_good > r_bad
    assert r_bad < 0.0 or r_bad < r_good - 1.0


def test_acceptance_gate_accepts_solid_improvement():
    np.random.seed(42)
    # Rewards before: ~1.0, after: ~1.4
    before = list(np.random.normal(1.0, 0.1, 50))
    after = list(np.random.normal(1.4, 0.1, 50))

    res = acceptance_gate(before, after, tol=0.02)
    assert res.accepted is True
    assert res.mean_improvement > 0.2
    assert res.lower_bound > 0.1


def test_acceptance_gate_rejects_regression_even_with_lucky_average():
    # Mean slightly positive (+0.014) but high tail risk drops lower bound below -0.02
    np.random.seed(42)
    before = [1.0] * 50
    # 38 queries improve by 0.08, but 12 drop by -0.25
    delta = [0.08] * 38 + [-0.25] * 12
    after = [b + d for b, d in zip(before, delta)]

    res = acceptance_gate(before, after, tol=0.02)
    # The lower bound falls below -0.02, triggering rejection despite average being near zero!
    assert res.accepted is False
    assert res.lower_bound < -0.02


def test_post_commit_watchdog():
    baseline = 1.5
    watchdog = PostCommitWatchdog(baseline_reward=baseline, window_size=10, max_allowed_drop=0.20)

    # Initial stable queries
    for _ in range(5):
        rollback, _ = watchdog.record_query(1.5)
        assert rollback is False

    # Simulate catastrophic degradation
    for _ in range(10):
        rollback, mean = watchdog.record_query(1.0)  # drop of 0.5 > 0.20

    assert rollback is True
    assert mean < 1.3
