"""
Query Telemetry Logger and Co-Access Tracker for MORPHE.
Records routing decisions, activated modules, latencies, confidence,
and maintains a real-time co-access frequency matrix between modules.
"""

from __future__ import annotations

import collections
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


@dataclass
class QueryTrace:
    """Trace of a single query through the memory architecture."""
    query: str
    modules_activated: List[str]
    routing_scores: Dict[str, float]
    propensities: Dict[str, float]
    latency_ms: float
    confidence: float
    gold_id: Optional[str] = None
    gold_hit: bool = False
    co_accessed_pairs: List[Tuple[str, str]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class TelemetryLogger:
    """
    SQLite-backed telemetry recorder for query routing and module usage.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = ":memory:"):
        self.db_path = str(db_path) if db_path is not None else ":memory:"
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        self._co_access_counter: collections.Counter[Tuple[str, str]] = collections.Counter()
        self._module_access_counter: collections.Counter[str] = collections.Counter()

    def _init_db(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    modules_json TEXT NOT NULL,
                    scores_json TEXT NOT NULL,
                    propensities_json TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    confidence REAL NOT NULL,
                    gold_id TEXT,
                    gold_hit INTEGER NOT NULL,
                    co_accessed_pairs_json TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
                """
            )

    def log_query(
        self,
        query: str,
        modules_activated: List[str],
        routing_scores: Dict[str, float],
        propensities: Dict[str, float],
        latency_ms: float,
        confidence: float,
        gold_id: Optional[str] = None,
        gold_hit: bool = False,
    ) -> QueryTrace:
        """Log a routing trace and update live co-access frequency counts."""
        # Calculate co-access pairs
        pairs: List[Tuple[str, str]] = []
        if len(modules_activated) > 1:
            sorted_mods = sorted(modules_activated)
            for i in range(len(sorted_mods)):
                for j in range(i + 1, len(sorted_mods)):
                    p = (sorted_mods[i], sorted_mods[j])
                    pairs.append(p)
                    self._co_access_counter[p] += 1

        for m in modules_activated:
            self._module_access_counter[m] += 1

        trace = QueryTrace(
            query=query,
            modules_activated=modules_activated,
            routing_scores=routing_scores,
            propensities=propensities,
            latency_ms=latency_ms,
            confidence=confidence,
            gold_id=gold_id,
            gold_hit=gold_hit,
            co_accessed_pairs=pairs,
            timestamp=time.time(),
        )

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO query_traces (
                    query, modules_json, scores_json, propensities_json,
                    latency_ms, confidence, gold_id, gold_hit,
                    co_accessed_pairs_json, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.query,
                    json.dumps(trace.modules_activated),
                    json.dumps(trace.routing_scores),
                    json.dumps(trace.propensities),
                    trace.latency_ms,
                    trace.confidence,
                    trace.gold_id,
                    1 if trace.gold_hit else 0,
                    json.dumps(trace.co_accessed_pairs),
                    trace.timestamp,
                ),
            )
        return trace

    def get_co_access_count(self, m1: str, m2: str) -> int:
        """Get total times module m1 and m2 were co-activated."""
        pair = tuple(sorted([m1, m2]))
        return self._co_access_counter.get(pair, 0)

    def get_module_access_count(self, module_id: str) -> int:
        """Get total activations for a specific module."""
        return self._module_access_counter.get(module_id, 0)

    def get_top_co_accessed_pairs(self, limit: int = 10) -> List[Tuple[Tuple[str, str], int]]:
        """Return the most frequent co-access module pairs."""
        return self._co_access_counter.most_common(limit)

    def recent_traces(self, limit: int = 50) -> List[QueryTrace]:
        """Fetch recent query traces from SQLite."""
        cursor = self.conn.execute(
            "SELECT * FROM query_traces ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        traces = []
        for row in cursor.fetchall():
            traces.append(QueryTrace(
                query=row["query"],
                modules_activated=json.loads(row["modules_json"]),
                routing_scores=json.loads(row["scores_json"]),
                propensities=json.loads(row["propensities_json"]),
                latency_ms=float(row["latency_ms"]),
                confidence=float(row["confidence"]),
                gold_id=row["gold_id"],
                gold_hit=bool(row["gold_hit"]),
                co_accessed_pairs=[tuple(p) for p in json.loads(row["co_accessed_pairs_json"])],
                timestamp=float(row["timestamp"]),
            ))
        return traces
