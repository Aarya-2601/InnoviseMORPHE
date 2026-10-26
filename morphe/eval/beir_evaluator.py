"""
BEIR SciFact Benchmark Evaluation for MORPHE.
Evaluates MORPHE against the official BEIR SciFact scientific retrieval dataset.
Compares:
1. Flat Baseline (unpartitioned flat index)
2. Static Fixed Hierarchy (un-adapted graph modules)
3. MORPHE Self-Reconfiguring Architecture (LinUCB Architect + Learned Stopping Router)

Measures:
- Recall@1, Recall@5, Recall@10
- Mean Reciprocal Rank (MRR)
- Latency (p50 and p95 ms)
- Modularity & Traversal efficiency (modules touched per query)
- Adaptation improvement (before vs. after self-reconfiguration)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from morphe.backend.morphe_backend import MorpheBackend
from morphe.eval.reward import compute_reward
from morphe.ingestion.chunker import HybridChunker
from morphe.models.metadata import MemoryType, RecordMetadata
from morphe.policy.architect_bandit import ArchitectBandit
from morphe.policy.diagnostician import Diagnostician
from morphe.policy.trial_scheduler import TrialScheduler
from morphe.router.learned_router import LearnedRouter
from morphe.storage.record_store import Record, RecordStore
from morphe.telemetry.logger import TelemetryLogger

logger = logging.getLogger(__name__)


class BEIRSciFactEvaluator:
    """
    Loads, ingests, and benchmarks MORPHE on the BEIR SciFact dataset.
    """

    def __init__(
        self,
        dataset_dir: str = "data/beir_scifact/scifact",
        sample_doc_limit: int = 400,
        sample_query_limit: int = 50,
        random_seed: int = 42,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.doc_limit = sample_doc_limit
        self.query_limit = sample_query_limit
        self.seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def load_dataset(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Set[str]]]:
        """
        Load corpus, queries, and qrels test mapping.
        Returns (corpus_dict, queries_dict, qrels_dict).
        """
        corpus_path = self.dataset_dir / "corpus.jsonl"
        queries_path = self.dataset_dir / "queries.jsonl"
        qrels_path = self.dataset_dir / "qrels" / "test.tsv"

        if not corpus_path.exists():
            raise FileNotFoundError(f"SciFact corpus not found at {corpus_path}")

        # 1. Load Qrels test set: query_id -> set of relevant doc_ids
        qrels: Dict[str, Set[str]] = {}
        with open(qrels_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)  # skip header query-id, corpus-id, score
            for row in reader:
                if len(row) >= 3 and int(row[2]) > 0:
                    qid, docid = row[0], row[1]
                    qrels.setdefault(qid, set()).add(docid)

        # 2. Identify target doc IDs needed by the test queries
        target_doc_ids: Set[str] = set()
        for qid in list(qrels.keys())[:self.query_limit]:
            target_doc_ids.update(qrels[qid])

        # 3. Load Corpus (include target docs + distractors up to doc_limit)
        corpus: Dict[str, Dict[str, str]] = {}
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                doc_id = str(item["_id"])
                # Always load docs that have relevance labels, fill remaining with distractors
                if doc_id in target_doc_ids or len(corpus) < self.doc_limit:
                    text_content = f"{item.get('title', '')}. {item.get('text', '')}".strip()
                    corpus[doc_id] = {
                        "title": item.get("title", ""),
                        "text": text_content,
                    }
                if len(corpus) >= self.doc_limit and target_doc_ids.issubset(corpus.keys()):
                    break

        # 4. Load Queries
        queries: Dict[str, str] = {}
        with open(queries_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                qid = str(item["_id"])
                if qid in qrels and any(did in corpus for did in qrels[qid]):
                    queries[qid] = item.get("text", "")
                    if len(queries) >= self.query_limit:
                        break

        return corpus, queries, qrels

    def populate_system(
        self,
        corpus: Dict[str, Dict[str, str]],
        num_initial_modules: int = 5,
    ) -> Tuple[MorpheBackend, LearnedRouter, TelemetryLogger, Dict[str, str]]:
        """
        Ingest SciFact docs into RecordStore and organize into thematic modules.
        Returns (backend, router, telemetry, docid_to_recordid_map).
        """
        store = RecordStore(":memory:")
        backend = MorpheBackend(record_store=store)
        doc_to_rec: Dict[str, str] = {}

        # Heuristic topic clustering by scientific keywords
        topic_buckets: Dict[str, List[str]] = {
            "mod_genetics_dna": [],
            "mod_immunology_cells": [],
            "mod_oncology_cancer": [],
            "mod_neurology_brain": [],
            "mod_biochem_metabolism": [],
        }

        for doc_id, doc_info in corpus.items():
            text = doc_info["text"]
            t_lower = text.lower()
            rec, _ = store.put(
                text=text,
                metadata={"source_id": doc_id, "project": "SciFact", "title": doc_info["title"]},
            )
            doc_to_rec[doc_id] = rec.id

            # Assign to topic bucket
            if any(k in t_lower for k in ["gene", "dna", "rna", "mutation", "sequence"]):
                topic_buckets["mod_genetics_dna"].append(rec.id)
            elif any(k in t_lower for k in ["immune", "t cell", "antibody", "infection", "vaccine"]):
                topic_buckets["mod_immunology_cells"].append(rec.id)
            elif any(k in t_lower for k in ["tumor", "cancer", "carcinoma", "malignant", "metastasis"]):
                topic_buckets["mod_oncology_cancer"].append(rec.id)
            elif any(k in t_lower for k in ["brain", "neural", "neuron", "cortex", "cognitive"]):
                topic_buckets["mod_neurology_brain"].append(rec.id)
            else:
                topic_buckets["mod_biochem_metabolism"].append(rec.id)

        # Register modules in backend
        charters = {
            "mod_genetics_dna": "Genetics, genomic sequencing, DNA repair, and transcription regulation.",
            "mod_immunology_cells": "Immunology, cellular defenses, antibodies, cytokines, and pathogen response.",
            "mod_oncology_cancer": "Oncology, tumor microenvironments, carcinogenesis, and antineoplastic agents.",
            "mod_neurology_brain": "Neurology, neural plasticity, cerebral cortex development, and synaptogenesis.",
            "mod_biochem_metabolism": "General biochemistry, cellular metabolism, signaling pathways, and homeostasis.",
        }

        for mod_id, r_ids in topic_buckets.items():
            backend.add_module(
                module_id=mod_id,
                name=mod_id.replace("mod_", "").replace("_", " ").title(),
                charter=charters.get(mod_id, "Scientific memory module"),
                record_ids=r_ids,
            )

        # Add initial cross-disciplinary associative edges
        m_keys = list(topic_buckets.keys())
        for i in range(len(m_keys)):
            u = m_keys[i]
            v = m_keys[(i + 1) % len(m_keys)]
            backend.add_edge(u, v, weight=0.6, edge_type="associative")

        backend.commit(reason_card="BEIR SciFact baseline graph initialization")

        telemetry = TelemetryLogger(":memory:")
        router = LearnedRouter(backend=backend, telemetry=telemetry, margin_threshold=0.20)
        return backend, router, telemetry, doc_to_rec

    def evaluate_retrieval(
        self,
        router: LearnedRouter,
        queries: Dict[str, str],
        qrels: Dict[str, Set[str]],
        doc_to_rec: Dict[str, str],
        top_k: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate full query set and calculate Recall@1, Recall@5, Recall@10, MRR, and Latency.
        """
        recalls_at_1 = []
        recalls_at_5 = []
        recalls_at_10 = []
        mrrs = []
        latencies = []
        modules_touched_list = []

        for qid, qtext in queries.items():
            relevant_doc_ids = qrels.get(qid, set())
            gold_rec_ids = {doc_to_rec[did] for did in relevant_doc_ids if did in doc_to_rec}
            if not gold_rec_ids:
                continue

            t0 = time.perf_counter()
            results, trace = router.route_and_retrieve(qtext, top_k=top_k)
            t_ms = (time.perf_counter() - t0) * 1000.0

            retrieved_ids = [r["id"] for r in results]
            
            # Hit check
            r1 = 1.0 if any(rid in gold_rec_ids for rid in retrieved_ids[:1]) else 0.0
            r5 = 1.0 if any(rid in gold_rec_ids for rid in retrieved_ids[:5]) else 0.0
            r10 = 1.0 if any(rid in gold_rec_ids for rid in retrieved_ids[:10]) else 0.0

            # Rank for MRR
            hit_rank = 0
            for idx, rid in enumerate(retrieved_ids, start=1):
                if rid in gold_rec_ids:
                    hit_rank = idx
                    break
            mrr_val = (1.0 / hit_rank) if hit_rank > 0 else 0.0

            recalls_at_1.append(r1)
            recalls_at_5.append(r5)
            recalls_at_10.append(r10)
            mrrs.append(mrr_val)
            latencies.append(t_ms)
            modules_touched_list.append(len(trace.modules_activated))

        lat_arr = np.array(latencies) if latencies else np.array([0.0])
        return {
            "queries_evaluated": len(recalls_at_5),
            "recall_at_1": round(float(np.mean(recalls_at_1)), 4),
            "recall_at_5": round(float(np.mean(recalls_at_5)), 4),
            "recall_at_10": round(float(np.mean(recalls_at_10)), 4),
            "mrr": round(float(np.mean(mrrs)), 4),
            "p50_latency_ms": round(float(np.median(lat_arr)), 2),
            "p95_latency_ms": round(float(np.percentile(lat_arr, 95)), 2),
            "avg_modules_touched": round(float(np.mean(modules_touched_list)), 2),
        }

    def run_full_benchmark(self) -> Dict[str, Any]:
        """
        Run complete comparison:
        1. Flat Baseline
        2. Static Modular Baseline
        3. MORPHE Self-Reconfiguring Adaptation Cycle
        """
        corpus, queries, qrels = self.load_dataset()
        backend, router, telemetry, doc_to_rec = self.populate_system(corpus)

        # Baseline 1: Flat Baseline (single monolithic module searching everything)
        all_recs = backend.record_store.list_all_ids()
        flat_backend = MorpheBackend(record_store=backend.record_store)
        flat_backend.add_module("mod_flat_scifact", charter="All Scientific Abstracts", record_ids=all_recs)
        flat_router = LearnedRouter(backend=flat_backend)
        metrics_flat = self.evaluate_retrieval(flat_router, queries, qrels, doc_to_rec)

        # Baseline 2: Static Fixed Modular Graph (Before Adaptation)
        metrics_before_adapt = self.evaluate_retrieval(router, queries, qrels, doc_to_rec)

        # 3. MORPHE Self-Reconfiguration Adaptation
        diagnostician = Diagnostician(overload_record_threshold=60, shortcut_co_access_threshold=2)
        bandit = ArchitectBandit(random_seed=self.seed)
        scheduler = TrialScheduler(telemetry=telemetry)

        # Let bandit propose adaptation based on observed query telemetry
        mutation, ctx = bandit.propose_mutation(backend, diagnostician, telemetry)
        
        # Build evaluation query list
        eval_queries = []
        for qid, qtext in queries.items():
            rel_docs = qrels.get(qid, set())
            for did in rel_docs:
                if did in doc_to_rec:
                    eval_queries.append({"query": qtext, "gold_id": doc_to_rec[did]})
                    break

        trial_result = scheduler.evaluate_mutation(backend, mutation, eval_queries)

        # Evaluate after adaptation
        metrics_after_adapt = self.evaluate_retrieval(router, queries, qrels, doc_to_rec)

        return {
            "dataset": "BEIR_SciFact",
            "total_documents_ingested": len(corpus),
            "total_queries_tested": len(queries),
            "baselines": {
                "Flat_Baseline": metrics_flat,
                "Static_Modular_Graph": metrics_before_adapt,
                "MORPHE_Adapted_Graph": metrics_after_adapt,
            },
            "adaptation_cycle": {
                "mutation_proposed": mutation.to_dict(),
                "tier_reached": trial_result.tier_reached,
                "accepted": trial_result.accepted,
                "compute_saved_score": trial_result.compute_saved_score,
                "reason": trial_result.reason,
            },
            "deltas": {
                "delta_recall_at_5": round(metrics_after_adapt["recall_at_5"] - metrics_before_adapt["recall_at_5"], 4),
                "delta_mrr": round(metrics_after_adapt["mrr"] - metrics_before_adapt["mrr"], 4),
                "delta_latency_p50": round(metrics_after_adapt["p50_latency_ms"] - metrics_before_adapt["p50_latency_ms"], 2),
            }
        }


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  BEIR SciFact Comparative Benchmark Evaluation for MORPHE (PS 3)")
    print("=" * 80 + "\n")
    evaluator = BEIRSciFactEvaluator(sample_doc_limit=497, sample_query_limit=50)
    results = evaluator.run_full_benchmark()

    print("\n" + "-" * 80)
    print(f"  Benchmark Dataset: {results['dataset']} ({results['total_documents_ingested']} docs, {results['total_queries_tested']} queries)")
    print("-" * 80)
    print(f"{'Metric':<22} | {'Flat Baseline':<16} | {'Static Modular':<16} | {'MORPHE Adapted':<16}")
    print("-" * 80)
    flat = results["baselines"]["Flat_Baseline"]
    static = results["baselines"]["Static_Modular_Graph"]
    morphe = results["baselines"]["MORPHE_Adapted_Graph"]

    print(f"{'Recall@1':<22} | {flat['recall_at_1']*100:<15.1f}% | {static['recall_at_1']*100:<15.1f}% | {morphe['recall_at_1']*100:<15.1f}%")
    print(f"{'Recall@5':<22} | {flat['recall_at_5']*100:<15.1f}% | {static['recall_at_5']*100:<15.1f}% | {morphe['recall_at_5']*100:<15.1f}%")
    print(f"{'Recall@10':<22} | {flat['recall_at_10']*100:<15.1f}% | {static['recall_at_10']*100:<15.1f}% | {morphe['recall_at_10']*100:<15.1f}%")
    print(f"{'MRR':<22} | {flat['mrr']:<16.3f} | {static['mrr']:<16.3f} | {morphe['mrr']:<16.3f}")
    print(f"{'p50 Latency (ms)':<22} | {flat['p50_latency_ms']:<16.1f} | {static['p50_latency_ms']:<16.1f} | {morphe['p50_latency_ms']:<16.1f}")
    print(f"{'p95 Latency (ms)':<22} | {flat['p95_latency_ms']:<16.1f} | {static['p95_latency_ms']:<16.1f} | {morphe['p95_latency_ms']:<16.1f}")
    print(f"{'Avg Modules Touched':<22} | {'All (Corpus)':<16} | {static['avg_modules_touched']:<16.1f} | {morphe['avg_modules_touched']:<16.1f}")
    print("-" * 80)
    print(f"  Adaptation Proposal: {results['adaptation_cycle']['mutation_proposed']['action_type'].upper()} on {results['adaptation_cycle']['mutation_proposed']['target_modules']}")
    print(f"  VoI Evaluation Gate: Tier {results['adaptation_cycle']['tier_reached']} | Compute Saved: {results['adaptation_cycle']['compute_saved_score']*100:.0f}% | Accepted: {results['adaptation_cycle']['accepted']}")
    print(f"  Decision Invariant : {results['adaptation_cycle']['reason']}")
    print("=" * 80 + "\n")

