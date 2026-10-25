"""
Unit and integration tests for Innovation features:
- LLM vs. Verifiable Reward Ablation (Section 12, Chart 3)
- Bandit Policy Transfer Benchmark (Section 10A.4)
- Autonomous 7-Step Presentation Demo Runner (Section 15)
- Pareto Frontier Weight Reconfiguration (Section 10A.3)
"""

import pytest
from fastapi.testclient import TestClient

from morphe.api.app import app
from morphe.demo import run_demo
from morphe.eval.ablation_llm_vs_reward import LLMvsRewardAblation
from morphe.eval.policy_transfer import PolicyTransferBenchmark

client = TestClient(app)


def test_llm_vs_reward_ablation():
    ablation = LLMvsRewardAblation(random_seed=42)
    results = ablation.run_ablation(num_trials=4)

    assert "total_mutations_tested" in results
    assert "llm_approval_rate" in results
    assert "factual_ground_truth_approval_rate" in results
    assert len(results["samples"]) == 4


def test_policy_transfer_benchmark():
    pt = PolicyTransferBenchmark(random_seed=42)
    results = pt.evaluate_transfer(test_cycles=3)

    assert "cold_start" in results
    assert "warm_start" in results
    assert "transfer_advantage_confirmed" in results["metrics"]


def test_pareto_frontier_api():
    res = client.post("/api/config/pareto", json={"speed_weight": 0.8})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "updated"
    assert "active_weights" in data
    assert "pareto_frontier" in data


def test_autonomous_7_step_demo():
    res = run_demo()
    assert res["status"] == "success"
    assert res["baseline_version"] == "v1"
    assert res["rollback_verified"] is True
    assert res["chaos_prevented"] is True
