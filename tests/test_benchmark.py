"""
Unit tests for BenchmarkRunner comparing all 4 baselines.
"""

from morphe.eval.benchmark import BenchmarkRunner


def test_benchmark_runner_all_baselines():
    runner = BenchmarkRunner(random_seed=42)
    results = runner.run_all_baselines(num_queries=6)

    assert "Flat_Baseline" in results
    assert "Fixed_Hierarchy" in results
    assert "MORPHE_Heuristic" in results
    assert "MORPHE_RL_Architect" in results

    for name, metrics in results.items():
        assert "recall_at_5" in metrics
        assert "mrr" in metrics
        assert "p50_latency_ms" in metrics
