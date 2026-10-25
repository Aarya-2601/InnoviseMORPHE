"""
Unit and integration tests for BEIR SciFact Benchmark Evaluator.
"""

from morphe.eval.beir_evaluator import BEIRSciFactEvaluator


def test_beir_evaluator_loader_and_benchmark():
    evaluator = BEIRSciFactEvaluator(
        dataset_dir="data/beir_scifact/scifact",
        sample_doc_limit=40,
        sample_query_limit=5,
        random_seed=42,
    )
    corpus, queries, qrels = evaluator.load_dataset()
    assert len(corpus) >= 10
    assert len(queries) >= 1

    backend, router, telemetry, doc_to_rec = evaluator.populate_system(corpus)
    assert len(backend.manifests) == 5

    metrics = evaluator.evaluate_retrieval(router, queries, qrels, doc_to_rec, top_k=5)
    assert "recall_at_5" in metrics
    assert "mrr" in metrics
    assert "p50_latency_ms" in metrics
