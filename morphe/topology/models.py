"""
MORPHE Object Model: Module Manifests, Topology Snapshots, and Version Commits.
Specification from Section 2C & 10A of Final Plan:
1. Module Manifest:
   - List of immutable record hashes in a module.
   - Module centroid vector (float32 array).
   - Charter (natural-language description / scope) & charter embedding.
   - Replaced, never mutated in-place.
2. Edge & Edge Types:
   - Source module, Target module, Weight.
   - EdgeType: REGULAR, SHORTCUT, SUPERSEDES, HIERARCHICAL.
   - Validity window: valid_from, valid_to (temporal memory support).
3. Topology Snapshot:
   - Mapping of module_id -> ModuleManifest.
   - List/Dict of Edges with weights and validity windows.
   - Replaced, never mutated in-place.
4. Version Commit:
   - Unique commit SHA or version ID.
   - Pointer to snapshot.
   - Parent version ID(s).
   - Timestamp and Reason card (plain-English justification).
   - Metrics dictionary: Recall@5, MRR, latency, cost.
   - Immutable once committed.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np


class EdgeType(str, Enum):
    """Types of edges between memory modules."""
    REGULAR = "regular"
    SHORTCUT = "shortcut"
    SUPERSEDES = "supersedes"       # Links newer knowledge to older superseded knowledge
    HIERARCHICAL = "hierarchical"   # Links parent summary nodes to child cluster modules


@dataclass(frozen=True)
class Edge:
    """
    Directed edge between two memory modules in the MORPHE topology graph.
    """
    source: str
    target: str
    weight: float = 1.0
    edge_type: EdgeType = EdgeType.REGULAR
    valid_from: Optional[float] = None
    valid_to: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def is_active(self, query_time: Optional[float] = None) -> bool:
        """Check if edge is currently valid in the temporal window."""
        t = time.time() if query_time is None else query_time
        if self.valid_from is not None and t < self.valid_from:
            return False
        if self.valid_to is not None and t > self.valid_to:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "edge_type": self.edge_type.value,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Edge:
        etype = data.get("edge_type", EdgeType.REGULAR)
        if isinstance(etype, str):
            etype = EdgeType(etype)
        return cls(
            source=data["source"],
            target=data["target"],
            weight=float(data.get("weight", 1.0)),
            edge_type=etype,
            valid_from=data.get("valid_from"),
            valid_to=data.get("valid_to"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class ModuleManifest:
    """
    Manifest of a MORPHE Memory Module.
    Immutable: replaced, never edited in-place.
    """
    module_id: str
    record_ids: Tuple[str, ...]
    charter: str = ""
    centroid: Optional[np.ndarray] = field(default=None, hash=False, compare=False)
    charter_embedding: Optional[np.ndarray] = field(default=None, hash=False, compare=False)
    created_at: float = field(default_factory=time.time)
    parent_module_id: Optional[str] = None
    tags: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    @classmethod
    def create(
        cls,
        module_id: str,
        record_ids: Sequence[str],
        charter: str = "",
        centroid: Optional[Union[np.ndarray, Sequence[float]]] = None,
        charter_embedding: Optional[Union[np.ndarray, Sequence[float]]] = None,
        created_at: Optional[float] = None,
        parent_module_id: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModuleManifest:
        cent = np.asarray(centroid, dtype=np.float32) if centroid is not None else None
        c_emb = np.asarray(charter_embedding, dtype=np.float32) if charter_embedding is not None else None
        
        # Deduplicate and sort record IDs for deterministic identity
        unique_records = tuple(sorted(set(record_ids)))

        return cls(
            module_id=module_id,
            record_ids=unique_records,
            charter=charter,
            centroid=cent,
            charter_embedding=c_emb,
            created_at=time.time() if created_at is None else created_at,
            parent_module_id=parent_module_id,
            tags=tuple(sorted(set(tags or []))),
            metadata=dict(metadata or {}),
        )

    def with_records(
        self,
        new_record_ids: Sequence[str],
        new_centroid: Optional[np.ndarray] = None,
    ) -> ModuleManifest:
        """Create a new replacement manifest with updated records (immutable object model)."""
        return ModuleManifest.create(
            module_id=self.module_id,
            record_ids=new_record_ids,
            charter=self.charter,
            centroid=new_centroid if new_centroid is not None else self.centroid,
            charter_embedding=self.charter_embedding,
            created_at=self.created_at,
            parent_module_id=self.parent_module_id,
            tags=self.tags,
            metadata=self.metadata,
        )

    @property
    def record_count(self) -> int:
        return len(self.record_ids)

    def to_dict(self, include_vectors: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "module_id": self.module_id,
            "record_ids": list(self.record_ids),
            "record_count": len(self.record_ids),
            "charter": self.charter,
            "created_at": self.created_at,
            "parent_module_id": self.parent_module_id,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }
        if include_vectors:
            if self.centroid is not None:
                data["centroid"] = self.centroid.tolist()
            if self.charter_embedding is not None:
                data["charter_embedding"] = self.charter_embedding.tolist()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModuleManifest:
        cent = np.asarray(data["centroid"], dtype=np.float32) if "centroid" in data and data["centroid"] is not None else None
        c_emb = np.asarray(data["charter_embedding"], dtype=np.float32) if "charter_embedding" in data and data["charter_embedding"] is not None else None
        return cls.create(
            module_id=data["module_id"],
            record_ids=data.get("record_ids", []),
            charter=data.get("charter", ""),
            centroid=cent,
            charter_embedding=c_emb,
            created_at=data.get("created_at"),
            parent_module_id=data.get("parent_module_id"),
            tags=data.get("tags"),
            metadata=data.get("metadata"),
        )


@dataclass(frozen=True)
class TopologySnapshot:
    """
    Complete immutable topology snapshot:
    Contains all module manifests and inter-module edges.
    """
    snapshot_id: str
    modules: Dict[str, ModuleManifest]
    edges: Tuple[Edge, ...]
    created_at: float = field(default_factory=time.time)

    @classmethod
    def create(
        cls,
        modules: Dict[str, ModuleManifest],
        edges: Sequence[Edge],
        created_at: Optional[float] = None,
    ) -> TopologySnapshot:
        # Deterministic snapshot hash based on module IDs, record sets, and edges
        summary_payload = {
            "modules": {m_id: sorted(m.record_ids) for m_id, m in sorted(modules.items())},
            "edges": sorted([(e.source, e.target, round(e.weight, 4), e.edge_type.value) for e in edges]),
        }
        raw_bytes = json.dumps(summary_payload, sort_keys=True).encode("utf-8")
        snapshot_id = hashlib.sha256(raw_bytes).hexdigest()[:16]

        return cls(
            snapshot_id=snapshot_id,
            modules=dict(modules),
            edges=tuple(edges),
            created_at=time.time() if created_at is None else created_at,
        )

    def get_module(self, module_id: str) -> Optional[ModuleManifest]:
        return self.modules.get(module_id)

    def get_outgoing_edges(self, source_module_id: str) -> List[Edge]:
        return [e for e in self.edges if e.source == source_module_id]

    def get_incoming_edges(self, target_module_id: str) -> List[Edge]:
        return [e for e in self.edges if e.target == target_module_id]

    def to_dict(self, include_vectors: bool = False) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "modules": {m_id: m.to_dict(include_vectors=include_vectors) for m_id, m in self.modules.items()},
            "edges": [e.to_dict() for e in self.edges],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TopologySnapshot:
        modules = {m_id: ModuleManifest.from_dict(m_data) for m_id, m_data in data.get("modules", {}).items()}
        edges = [Edge.from_dict(e_data) for e_data in data.get("edges", [])]
        return cls.create(modules=modules, edges=edges, created_at=data.get("created_at"))


@dataclass(frozen=True)
class VersionCommit:
    """
    Immutable Version Commit representing an architectural state of MORPHE.
    """
    version_id: str
    snapshot: TopologySnapshot
    parent_version: Optional[str] = None
    reason_card: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)  # Recall@5, MRR, latency, cost
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        snapshot: TopologySnapshot,
        parent_version: Optional[str] = None,
        reason_card: str = "",
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        version_id: Optional[str] = None,
    ) -> VersionCommit:
        t = time.time()
        if version_id is None:
            # Deterministic commit hash from parent, snapshot, and timestamp
            commit_payload = f"{parent_version}_{snapshot.snapshot_id}_{t}_{reason_card}"
            version_id = "v_" + hashlib.sha256(commit_payload.encode("utf-8")).hexdigest()[:12]

        return cls(
            version_id=version_id,
            snapshot=snapshot,
            parent_version=parent_version,
            reason_card=reason_card,
            metrics=dict(metrics or {}),
            created_at=t,
            metadata=dict(metadata or {}),
        )

    def to_dict(self, include_vectors: bool = False) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "parent_version": self.parent_version,
            "snapshot": self.snapshot.to_dict(include_vectors=include_vectors),
            "reason_card": self.reason_card,
            "metrics": dict(self.metrics),
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

