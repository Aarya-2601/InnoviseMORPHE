"""
Temporal Memory and Supersede Resolution for MORPHE.
Specification from Section 2B (Q7), 10A.1, and 14A (build step 7):
- Supersede Edges:
  * When a fact is updated (e.g. "we switched from PostgreSQL to MongoDB"),
    the old record is kept (immutable) and linked by a SUPERSEDES edge with a validity window.
  * Retrieval prefers current facts by default.
  * If the query asks for history ("what did we use before?"), retrieval traverses
    the supersedes edge backwards to return the historical fact.
- Learned Safe Forgetting:
  * Archived records stay in the immutable store with archived=True,
    allowing archive/forgetting decisions to be safely rolled back.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from morphe.core.backend import MorpheBackend
from morphe.storage.record_store import Record, RecordStore
from morphe.topology.models import Edge, EdgeType, ModuleManifest
from morphe.topology.mutations import MutationAction, MutationType


HISTORY_KEYWORDS = [
    "before",
    "previously",
    "historical",
    "historically",
    "history",
    "past",
    "prior",
    "former",
    "formerly",
    "originally",
    "switched from",
    "used to",
    "earlier",
]


def is_historical_query(query: str) -> bool:
    """Detect if query is seeking historical / superseded information."""
    q_lower = query.lower()
    return any(re.search(r'\b' + re.escape(kw) + r'\b', q_lower) for kw in HISTORY_KEYWORDS)


class TemporalMemoryManager:
    """
    Manages supersedes edges, record validity windows, and archive flags.
    """

    def __init__(self, backend: MorpheBackend):
        self.backend = backend

    def record_supersession(
        self,
        new_record_id: str,
        old_record_id: str,
        source_module: str,
        target_module: Optional[str] = None,
        effective_time: Optional[float] = None,
        archive_old: bool = True,
    ) -> Edge:
        """
        Links a new record to an old record via a SUPERSEDES edge.
        Sets validity window:
        - old record valid until effective_time
        - new record valid from effective_time
        - marks old record as archived (if archive_old=True)
        - adds SUPERSEDES edge into the topology graph
        """
        now = time.time() if effective_time is None else effective_time
        tgt_mod = target_module or source_module

        # 1. Update metadata in RecordStore
        old_record = self.backend.record_store.get(old_record_id)
        if old_record is not None:
            if archive_old:
                self.backend.record_store.set_archived(old_record_id, archived=True)

        # 2. Add supersedes edge into topology
        edge = Edge(
            source=source_module,
            target=tgt_mod,
            weight=1.0,
            edge_type=EdgeType.SUPERSEDES,
            valid_from=now,
            valid_to=None,
            metadata={
                "supersedes": True,
                "new_record_id": new_record_id,
                "old_record_id": old_record_id,
                "superseded_at": now,
            },
        )

        action = MutationAction(
            mutation_type=MutationType.ADD_EDGE,
            source_module=source_module,
            target_module=tgt_mod,
            weight=1.0,
            edge_type=EdgeType.SUPERSEDES,
            metadata=edge.metadata,
        )
        self.backend.apply_mutation(action)
        return edge

    def resolve_temporal_records(
        self,
        record_candidates: Sequence[Tuple[str, float]],  # (record_id, similarity_score)
        query: str,
        query_time: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """
        Filters and ranks candidate records based on temporal intent and supersedes edges.
        - If query is historical: boost superseded / archived records matching history.
        - If query is current: prefer active (non-archived) records and suppress superseded ones.
        """
        t = time.time() if query_time is None else query_time
        wants_history = is_historical_query(query)

        # Build supersedes map from active topology edges
        snapshot = self.backend.get_staged_snapshot()
        supersedes_pairs: Dict[str, str] = {}  # new_id -> old_id
        reverse_supersedes: Dict[str, str] = {}  # old_id -> new_id

        for edge in snapshot.edges:
            if edge.edge_type == EdgeType.SUPERSEDES:
                meta = edge.metadata or {}
                new_id = meta.get("new_record_id")
                old_id = meta.get("old_record_id")
                if new_id and old_id:
                    supersedes_pairs[new_id] = old_id
                    reverse_supersedes[old_id] = new_id

        # Fetch records to check archive status
        all_ids = [rid for rid, _ in record_candidates]
        record_map = self.backend.record_store.get_many(all_ids)

        adjusted_results: List[Tuple[str, float]] = []

        for rid, score in record_candidates:
            rec = record_map.get(rid)
            is_archived = rec.archived if rec else False

            if wants_history:
                # If query seeks history, prefer the older / superseded records
                if rid in reverse_supersedes:
                    # This is an older record superseded by a newer one -> boost it
                    adjusted_results.append((rid, score * 1.5))
                elif is_archived:
                    adjusted_results.append((rid, score * 1.3))
                elif rid in supersedes_pairs:
                    # Current record, down-weight if query specifically asks "what did we use before"
                    adjusted_results.append((rid, score * 0.7))
                else:
                    adjusted_results.append((rid, score))
            else:
                # Normal current query
                if is_archived:
                    # Down-weight or suppress archived records
                    adjusted_results.append((rid, score * 0.1))
                elif rid in supersedes_pairs:
                    # Active newer record -> boost
                    adjusted_results.append((rid, score * 1.2))
                else:
                    adjusted_results.append((rid, score))

        # Re-sort descending by adjusted score
        adjusted_results.sort(key=lambda x: x[1], reverse=True)
        return adjusted_results
