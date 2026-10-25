"""
Unit tests for DatasetBuilder:
- Ingestion of multi-project corpus
- Disjoint splits: golden_set, selfplay_set, heldout_set
- Lexical overlap Jaccard filtering
- Serialization to JSON
"""

import tempfile
from pathlib import Path

from morphe.data.generator import DatasetBuilder, compute_jaccard_similarity
from morphe.storage.record_store import RecordStore


def test_jaccard_similarity():
    text_a = "OAuth2 authorization code flow with PKCE"
    text_b = "OAuth2 code flow"
    sim = compute_jaccard_similarity(text_a, text_b)
    assert 0.0 < sim < 1.0

    # Disjoint text
    text_c = "PostgreSQL B-Tree index optimization"
    assert compute_jaccard_similarity(text_a, text_c) == 0.0


def test_dataset_builder_generation_and_splits():
    store = RecordStore(":memory:")
    builder = DatasetBuilder(record_store=store, random_seed=42)

    records = builder.ingest_corpus()
    assert len(records) > 0
    assert store.count() == len(records)

    golden, selfplay, heldout = builder.generate_benchmark_splits(records)
    assert len(golden) > 0
    assert len(selfplay) > 0
    assert len(heldout) > 0

    # Ensure strictly disjoint gold_id chunk membership across splits
    gold_ids_g = {q["gold_id"] for q in golden}
    gold_ids_s = {q["gold_id"] for q in selfplay}
    gold_ids_h = {q["gold_id"] for q in heldout}

    assert len(gold_ids_g & gold_ids_s) == 0, "Golden and Self-play share chunk IDs!"
    assert len(gold_ids_g & gold_ids_h) == 0, "Golden and Held-out share chunk IDs!"
    assert len(gold_ids_s & gold_ids_h) == 0, "Self-play and Held-out share chunk IDs!"


def test_save_splits_to_json():
    store = RecordStore(":memory:")
    builder = DatasetBuilder(record_store=store, random_seed=42)
    records = builder.ingest_corpus()
    golden, selfplay, heldout = builder.generate_benchmark_splits(records)

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = builder.save_splits_to_json(tmpdir, golden, selfplay, heldout)
        for name, p in paths.items():
            assert Path(p).exists()
            assert Path(p).stat().st_size > 10
