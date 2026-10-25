"""
Topology data models for MORPHE.
Represents memory modules (manifests), graph edges, mutations, version commits, and diffs.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np


@dataclass
class ModuleManifest:
    """
    Manifest of a specialized memory module (graph node).
    Immutable within a version snapshot; replaced rather than mutated in-place.
    """
    module_id: str
    name: str = ""
    charter: str = ""
    record_ids: Set[str] = field(default_factory=set)
    centroid: Optional[np.ndarray] = field(default=None, repr=False)
    level: int = 0  # 0: leaf base module, 1+: hierarchical summary hub
    created_at: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_centroid: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "module_id": self.module_id,
            "name": self.name,
            "charter": self.charter,
            "record_ids": sorted(list(self.record_ids)),
            "level": self.level,
            "record_count": len(self.record_ids),
            "created_at": self.created_at,
            "extra": dict(self.extra),
        }
        if include_centroid and self.centroid is not None:
            d["centroid"] = self.centroid.tolist()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModuleManifest:
        rec_ids = set(data.get("record_ids", []))
        centroid = None
        if "centroid" in data and data["centroid"] is not None:
            centroid = np.asarray(data["centroid"], dtype=np.float32)
        return cls(
            module_id=data["module_id"],
            name=data.get("name", data["module_id"]),
            charter=data.get("charter", ""),
            record_ids=rec_ids,
            centroid=centroid,
            level=int(data.get("level", 0)),
            created_at=float(data.get("created_at", time.time())),
            extra=dict(data.get("extra", {})),
        )

    def clone(self) -> ModuleManifest:
        """Deep copy manifest for copy-on-write branching."""
        return ModuleManifest(
            module_id=self.module_id,
            name=self.name,
            charter=self.charter,
            record_ids=set(self.record_ids),
            centroid=self.centroid.copy() if self.centroid is not None else None,
            level=self.level,
            created_at=self.created_at,
            extra=copy.deepcopy(self.extra),
        )


@dataclass
class EdgeData:
    """
    Edge representing associative, hierarchical, or temporal pathways between modules.
    """
    source: str
    target: str
    weight: float = 1.0
    edge_type: str = "associative"  # "associative", "hierarchical", "shortcut", "supersede"
    valid_from: Optional[float] = None
    valid_to: Optional[float] = None
    co_access_count: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "weight": float(self.weight),
            "edge_type": self.edge_type,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "co_access_count": self.co_access_count,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EdgeData:
        return cls(
            source=data["source"],
            target=data["target"],
            weight=float(data.get("weight", 1.0)),
            edge_type=data.get("edge_type", "associative"),
            valid_from=data.get("valid_from"),
            valid_to=data.get("valid_to"),
            co_access_count=int(data.get("co_access_count", 0)),
            extra=dict(data.get("extra", {})),
        )


@dataclass
class Mutation:
    """
    A proposed or applied structural reconfiguration action.
    """
    action_type: str  # "split", "merge", "shortcut", "prune", "promote", "do_nothing"
    target_modules: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target_modules": list(self.target_modules),
            "parameters": copy.deepcopy(self.parameters),
            "reason": self.reason,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Mutation:
        return cls(
            action_type=data["action_type"],
            target_modules=list(data.get("target_modules", [])),
            parameters=dict(data.get("parameters", {})),
            reason=data.get("reason", ""),
            timestamp=float(data.get("timestamp", time.time())),
        )


@dataclass
class VersionCommit:
    """
    Immutable ledger commit snapshot representing an architecture version.
    """
    version_id: str
    parent_version_id: Optional[str]
    timestamp: float = field(default_factory=time.time)
    mutation_applied: Optional[Dict[str, Any]] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    reason_card: str = ""
    module_count: int = 0
    edge_count: int = 0
    total_records: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "parent_version_id": self.parent_version_id,
            "timestamp": self.timestamp,
            "mutation_applied": copy.deepcopy(self.mutation_applied),
            "metrics": copy.deepcopy(self.metrics),
            "reason_card": self.reason_card,
            "module_count": self.module_count,
            "edge_count": self.edge_count,
            "total_records": self.total_records,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VersionCommit:
        return cls(
            version_id=data["version_id"],
            parent_version_id=data.get("parent_version_id"),
            timestamp=float(data.get("timestamp", time.time())),
            mutation_applied=data.get("mutation_applied"),
            metrics=dict(data.get("metrics", {})),
            reason_card=data.get("reason_card", ""),
            module_count=int(data.get("module_count", 0)),
            edge_count=int(data.get("edge_count", 0)),
            total_records=int(data.get("total_records", 0)),
        )


@dataclass
class TopologyDiff:
    """
    Diff between two architecture versions v1 and v2.
    """
    v1: str
    v2: str
    added_modules: List[str] = field(default_factory=list)
    removed_modules: List[str] = field(default_factory=list)
    modified_modules: List[str] = field(default_factory=list)
    added_edges: List[Dict[str, Any]] = field(default_factory=list)
    removed_edges: List[Tuple[str, str]] = field(default_factory=list)
    modified_edges: List[Dict[str, Any]] = field(default_factory=list)
    metric_deltas: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "v1": self.v1,
            "v2": self.v2,
            "added_modules": self.added_modules,
            "removed_modules": self.removed_modules,
            "modified_modules": self.modified_modules,
            "added_edges": self.added_edges,
            "removed_edges": self.removed_edges,
            "modified_edges": self.modified_edges,
            "metric_deltas": self.metric_deltas,
        }
