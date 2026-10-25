"""
Persistent SQLite Ledger for MORPHE.
Specification from Section 2A, 2B (Q3), 10A.3, and 13 of MORPHE Final Plan:
- Stores every committed version:
  * version_id (PRIMARY KEY)
  * parent_version
  * snapshot_json (full module manifests, edges, validity windows)
  * reason_card (plain-English justification)
  * metrics: Recall@5, MRR, latency_ms, cost, modules_touched, structural_cost
  * is_head flag
  * timestamp
- Provides time-travel query methods:
  * get_version(version_id)
  * get_head()
  * list_history()
  * get_pareto_frontier() -> returns versions on accuracy vs. latency vs. cost frontier
  * log_commit(...)
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from morphe.topology.models import Edge, ModuleManifest, TopologySnapshot, VersionCommit


class VersionLedger:
    """
    SQLite-backed immutable Version Ledger.
    Records topology snapshots, reason cards, and metrics per version.
    """

    def __init__(self, db_path: Union[str, Path] = ":memory:"):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Create version ledger table and indexes."""
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ledger_versions (
                    version_id TEXT PRIMARY KEY,
                    parent_version TEXT,
                    snapshot_json TEXT NOT NULL,
                    reason_card TEXT NOT NULL,
                    recall_at_5 REAL,
                    mrr REAL,
                    latency_ms REAL,
                    cost REAL,
                    modules_touched REAL,
                    structural_cost REAL,
                    metrics_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    is_head INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                )
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ledger_created ON ledger_versions(created_at)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ledger_head ON ledger_versions(is_head)"
            )

    def record_commit(
        self,
        commit: VersionCommit,
        set_as_head: bool = True,
    ) -> None:
        """Record an immutable version commit into SQLite ledger."""
        metrics = commit.metrics or {}
        snapshot_dict = commit.snapshot.to_dict(include_vectors=False)

        with self.conn:
            if set_as_head:
                self.conn.execute("UPDATE ledger_versions SET is_head = 0 WHERE is_head = 1")

            self.conn.execute(
                """
                INSERT OR REPLACE INTO ledger_versions (
                    version_id, parent_version, snapshot_json, reason_card,
                    recall_at_5, mrr, latency_ms, cost, modules_touched, structural_cost,
                    metrics_json, metadata_json, is_head, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    commit.version_id,
                    commit.parent_version,
                    json.dumps(snapshot_dict),
                    commit.reason_card,
                    metrics.get("Recall@5", metrics.get("recall_at_5")),
                    metrics.get("MRR", metrics.get("mrr")),
                    metrics.get("latency_ms", metrics.get("latency")),
                    metrics.get("cost"),
                    metrics.get("modules_touched"),
                    metrics.get("structural_cost"),
                    json.dumps(metrics),
                    json.dumps(commit.metadata),
                    1 if set_as_head else 0,
                    commit.created_at,
                ),
            )

    def set_head(self, version_id: str) -> bool:
        """Move HEAD pointer to specified version (e.g., during rollback)."""
        with self.conn:
            cur = self.conn.execute(
                "SELECT 1 FROM ledger_versions WHERE version_id = ?", (version_id,)
            )
            if not cur.fetchone():
                return False
            self.conn.execute("UPDATE ledger_versions SET is_head = 0")
            self.conn.execute(
                "UPDATE ledger_versions SET is_head = 1 WHERE version_id = ?",
                (version_id,),
            )
            return True

    def get_head(self) -> Optional[VersionCommit]:
        """Fetch current HEAD version commit."""
        cur = self.conn.execute(
            "SELECT * FROM ledger_versions WHERE is_head = 1 ORDER BY created_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            return None
        return self._row_to_commit(row)

    def get_version(self, version_id: str) -> Optional[VersionCommit]:
        """Retrieve a specific version commit by version ID."""
        cur = self.conn.execute(
            "SELECT * FROM ledger_versions WHERE version_id = ?", (version_id,)
        )
        row = cur.fetchone()
        if not row:
            return None
        return self._row_to_commit(row)

    def list_history(self) -> List[Dict[str, Any]]:
        """List entire commit history ordered chronologically."""
        cur = self.conn.execute(
            """
            SELECT version_id, parent_version, reason_card, metrics_json,
                   is_head, created_at
            FROM ledger_versions
            ORDER BY created_at ASC
            """
        )
        rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append({
                "version_id": r["version_id"],
                "parent_version": r["parent_version"],
                "reason_card": r["reason_card"],
                "metrics": json.loads(r["metrics_json"]),
                "is_head": bool(r["is_head"]),
                "created_at": r["created_at"],
            })
        return out

    def get_pareto_frontier(self) -> List[Dict[str, Any]]:
        """
        Returns non-dominated versions across (accuracy vs latency vs structural_cost).
        A version A dominates version B if:
        Recall(A) >= Recall(B), Latency(A) <= Latency(B), Cost(A) <= Cost(B)
        with at least one strict inequality.
        """
        history = self.list_history()
        candidates: List[Dict[str, Any]] = []
        for item in history:
            m = item["metrics"]
            recall = m.get("Recall@5", m.get("recall_at_5", 0.0))
            latency = m.get("latency_ms", m.get("latency", 9999.0))
            cost = m.get("cost", m.get("structural_cost", 1.0))
            candidates.append({
                "version_id": item["version_id"],
                "reason_card": item["reason_card"],
                "recall": float(recall),
                "latency": float(latency),
                "cost": float(cost),
                "metrics": m,
            })

        frontier: List[Dict[str, Any]] = []
        for i, a in enumerate(candidates):
            dominated = False
            for j, b in enumerate(candidates):
                if i == j:
                    continue
                # b dominates a if b is at least as good in all, and strictly better in at least one
                b_as_good = (
                    b["recall"] >= a["recall"]
                    and b["latency"] <= a["latency"]
                    and b["cost"] <= a["cost"]
                )
                b_strictly_better = (
                    b["recall"] > a["recall"]
                    or b["latency"] < a["latency"]
                    or b["cost"] < a["cost"]
                )
                if b_as_good and b_strictly_better:
                    dominated = True
                    break
            if not dominated:
                frontier.append(a)

        return frontier

    def _row_to_commit(self, row: sqlite3.Row) -> VersionCommit:
        snapshot_dict = json.loads(row["snapshot_json"])
        snapshot = TopologySnapshot.from_dict(snapshot_dict)
        metrics = json.loads(row["metrics_json"])
        metadata = json.loads(row["metadata_json"])

        return VersionCommit(
            version_id=row["version_id"],
            snapshot=snapshot,
            parent_version=row["parent_version"],
            reason_card=row["reason_card"],
            metrics=metrics,
            created_at=row["created_at"],
            metadata=metadata,
        )

    def count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM ledger_versions")
        return cur.fetchone()[0]

    def close(self) -> None:
        self.conn.close()

