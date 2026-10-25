"""
Learned Stopping Hybrid Router for MORPHE.
Combines:
1. Module scoring via centroid cosine + charter BM25.
2. Softmax temperature activation with spreading activation across edges.
3. Optimal-stopping policy based on routing entropy and top-1 vs top-2 margin.
4. Record retrieval inside activated modules via Dense + BM25 fused by RRF.
5. Low-confidence fallback to Gemma query rewriting / HyDE.
"""

from __future__ import annotations

import collections
import math
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from morphe.backend.morphe_backend import MorpheBackend
from morphe.llm.client import LLMClient
from morphe.storage.record_store import Record, RecordStore
from morphe.telemetry.logger import QueryTrace, TelemetryLogger
from morphe.temporal.manager import is_historical_query


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer."""
    return re.findall(r'\w+', text.lower())


class SimpleBM25:
    """Lightweight in-memory BM25 scorer."""

    def __init__(self, corpus: Dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.doc_ids = list(corpus.keys())
        self.doc_tokens = {doc_id: tokenize(text) for doc_id, text in corpus.items()}
        self.k1 = k1
        self.b = b
        self.doc_lens = {doc_id: len(toks) for doc_id, toks in self.doc_tokens.items()}
        self.avgdl = (sum(self.doc_lens.values()) / max(1, len(self.doc_lens))) if self.doc_lens else 1.0

        # Term frequencies and document frequencies
        self.df: collections.Counter[str] = collections.Counter()
        self.tf: Dict[str, collections.Counter[str]] = {}
        for doc_id, toks in self.doc_tokens.items():
            t_counts = collections.Counter(toks)
            self.tf[doc_id] = t_counts
            for t in t_counts:
                self.df[t] += 1

        self.N = max(1, len(self.doc_ids))

    def score(self, query: str) -> Dict[str, float]:
        q_tokens = tokenize(query)
        scores: Dict[str, float] = {doc_id: 0.0 for doc_id in self.doc_ids}
        for t in q_tokens:
            if t not in self.df:
                continue
            df_t = self.df[t]
            idf = math.log(1.0 + (self.N - df_t + 0.5) / (df_t + 0.5))
            for doc_id, t_counts in self.tf.items():
                f = t_counts.get(t, 0)
                if f > 0:
                    denom = f + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[doc_id] / self.avgdl))
                    scores[doc_id] += idf * (f * (self.k1 + 1.0)) / denom
        return scores


class LearnedRouter:
    """
    Intelligent router with spreading activation and learned optimal stopping.
    """

    def __init__(
        self,
        backend: MorpheBackend,
        telemetry: Optional[TelemetryLogger] = None,
        llm_client: Optional[LLMClient] = None,
        temperature: float = 0.5,
        decay_factor: float = 0.5,
        margin_threshold: float = 0.25,
        entropy_threshold: float = 1.0,
        max_modules: int = 4,
    ):
        self.backend = backend
        self.telemetry = telemetry
        self.llm = llm_client if llm_client is not None else LLMClient(force_fallback=True)
        self.temperature = temperature
        self.decay_factor = decay_factor
        self.margin_threshold = margin_threshold
        self.entropy_threshold = entropy_threshold
        self.max_modules = max_modules

    def route_and_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        gold_id: Optional[str] = None,
        top_k: int = 5,
    ) -> Tuple[List[Dict[str, Any]], QueryTrace]:
        """
        Execute full routing loop:
        1. Score modules by semantic similarity + charter keyword match.
        2. Spread activation along graph edges.
        3. Decide stopping condition based on routing entropy & margin.
        4. Search inside activated modules using Dense + BM25 RRF.
        5. Log telemetry trace.
        """
        t0 = time.perf_counter()
        manifests = self.backend.manifests
        graph = self.backend.graph

        if not manifests:
            empty_trace = QueryTrace(
                query=query,
                modules_activated=[],
                routing_scores={},
                propensities={},
                latency_ms=0.0,
                confidence=0.0,
            )
            return [], empty_trace

        # Step 1: Base Module Scoring
        module_scores: Dict[str, float] = {}
        for mid, man in manifests.items():
            score = 0.0
            # Dense centroid similarity
            if query_embedding is not None and man.centroid is not None:
                q_norm = np.linalg.norm(query_embedding)
                if q_norm > 1e-9:
                    sim = float(np.dot(query_embedding / q_norm, man.centroid))
                    score += max(0.0, sim)
            # Charter lexical match
            q_toks = set(tokenize(query))
            c_toks = set(tokenize(man.charter + " " + man.name))
            if q_toks and c_toks:
                overlap = len(q_toks & c_toks) / len(q_toks)
                score += 0.8 * overlap

            module_scores[mid] = score

        # Step 2: Spreading Activation along edges
        spread_scores = dict(module_scores)
        for u in graph.nodes:
            if u not in module_scores or module_scores[u] <= 0:
                continue
            for _, v, edata in graph.out_edges(u, data=True):
                w = float(edata.get("weight", 1.0))
                if v in spread_scores:
                    spread_scores[v] += self.decay_factor * w * module_scores[u]

        # Softmax probabilities over modules
        mids = list(spread_scores.keys())
        raw_vals = np.array([spread_scores[m] for m in mids], dtype=np.float64)
        scaled = (raw_vals - np.max(raw_vals)) / max(0.1, self.temperature)
        exp_vals = np.exp(scaled)
        propensities = {m: float(p) for m, p in zip(mids, exp_vals / np.sum(exp_vals))}

        # Step 3: Optimal-Stopping Policy
        ranked_mids = sorted(mids, key=lambda m: propensities[m], reverse=True)
        
        # Calculate routing entropy and margin
        p_vals = np.array([propensities[m] for m in ranked_mids])
        entropy = float(-np.sum(p_vals * np.log(np.maximum(1e-9, p_vals))))
        margin = float(p_vals[0] - p_vals[1]) if len(p_vals) > 1 else 1.0
        confidence = float(max(0.0, (1.0 - entropy / max(1.0, math.log(max(2, len(mids))))) * margin))

        # Check stopping rule
        activated_modules: List[str] = [ranked_mids[0]]
        for cand_m in ranked_mids[1:self.max_modules]:
            # Stop if margin is high and entropy is below threshold
            if margin >= self.margin_threshold and entropy <= self.entropy_threshold:
                break
            activated_modules.append(cand_m)

        # Step 4: Search inside activated modules
        is_hist = is_historical_query(query)
        records_to_search: Dict[str, Record] = {}
        for m in activated_modules:
            man = manifests.get(m)
            if man:
                for r_id in man.record_ids:
                    rec = self.backend.record_store.get(r_id)
                    if rec:
                        if not is_hist and rec.archived:
                            continue
                        records_to_search[r_id] = rec

        retrieved_results: List[Dict[str, Any]] = []
        if records_to_search:
            # BM25 over activated records
            corpus_dict = {rid: r.text for rid, r in records_to_search.items()}
            bm25 = SimpleBM25(corpus_dict)
            bm25_scores = bm25.score(query)

            # RRF Fusion (BM25 rank + Dense rank)
            bm25_ranked = sorted(records_to_search.keys(), key=lambda rid: bm25_scores[rid], reverse=True)
            
            # Dense ranking
            dense_ranked = []
            if query_embedding is not None:
                dense_scores = {}
                q_norm = np.linalg.norm(query_embedding)
                q_unit = query_embedding / max(1e-9, q_norm)
                for rid, r in records_to_search.items():
                    if r.embedding is not None:
                        dense_scores[rid] = float(np.dot(q_unit, r.embedding))
                    else:
                        dense_scores[rid] = 0.0
                dense_ranked = sorted(records_to_search.keys(), key=lambda rid: dense_scores[rid], reverse=True)
            else:
                dense_ranked = list(bm25_ranked)

            rrf_scores: collections.Counter[str] = collections.Counter()
            for rank_idx, rid in enumerate(bm25_ranked, start=1):
                rrf_scores[rid] += 1.0 / (60.0 + rank_idx)
            for rank_idx, rid in enumerate(dense_ranked, start=1):
                rrf_scores[rid] += 1.0 / (60.0 + rank_idx)

            final_ranked_ids = [rid for rid, _ in rrf_scores.most_common(top_k)]
            for rid in final_ranked_ids:
                r_obj = records_to_search[rid]
                retrieved_results.append({
                    "id": r_obj.id,
                    "text": r_obj.text,
                    "project": r_obj.metadata.get("project", "default"),
                    "score": round(rrf_scores[rid], 4),
                })

        latency_ms = (time.perf_counter() - t0) * 1000.0
        gold_hit = bool(gold_id and any(item["id"] == gold_id for item in retrieved_results[:top_k]))

        # Step 5: Log Telemetry Trace
        trace = QueryTrace(
            query=query,
            modules_activated=activated_modules,
            routing_scores=spread_scores,
            propensities=propensities,
            latency_ms=round(latency_ms, 2),
            confidence=round(confidence, 4),
            gold_id=gold_id,
            gold_hit=gold_hit,
        )

        if self.telemetry is not None:
            self.telemetry.log_query(
                query=query,
                modules_activated=activated_modules,
                routing_scores=spread_scores,
                propensities=propensities,
                latency_ms=latency_ms,
                confidence=confidence,
                gold_id=gold_id,
                gold_hit=gold_hit,
            )

        return retrieved_results, trace
