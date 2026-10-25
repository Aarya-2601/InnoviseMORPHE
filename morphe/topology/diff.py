"""
Topology Diff representation for MORPHE.
Tracks:
- Added modules
- Removed modules
- Modified modules (records added or removed, centroid/charter changes)
- Added edges
- Removed edges
- Modified edge weights
- Metric deltas (Recall, MRR, latency, cost)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from morphe.topology.models import Edge, TopologySnapshot, VersionCommit


@dataclass
class ModuleDiff:
    module_id: str
    added_records: List[str] = field(default_factory=list)
    removed_records: List[str] = field(default_factory=list)
    charter_changed: bool = False
    old_charter: str = ""
    new_charter: str = ""

    @property
    def has_changes(self) -> bool:
        return bool(self.added_records or self.removed_records or self.charter_changed)


@dataclass
class EdgeDiff:
    source: str
    target: str
    old_weight: Optional[float] = None
    new_weight: Optional[float] = None
    old_type: Optional[str] = None
    new_type: Optional[str] = None


@dataclass
class TopologyDiff:
    """
    Diff between two snapshots or two version commits.
    """
    v1_id: str
    v2_id: str
    added_modules: List[str] = field(default_factory=list)
    removed_modules: List[str] = field(default_factory=list)
    module_changes: Dict[str, ModuleDiff] = field(default_factory=dict)
    added_edges: List[Edge] = field(default_factory=list)
    removed_edges: List[Edge] = field(default_factory=list)
    modified_edges: List[EdgeDiff] = field(default_factory=list)
    metric_deltas: Dict[str, float] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (
            self.added_modules
            or self.removed_modules
            or self.module_changes
            or self.added_edges
            or self.removed_edges
            or self.modified_edges
        )

    def summary(self) -> str:
        parts: List[str] = []
        if self.added_modules:
            parts.append(f"+modules: {self.added_modules}")
        if self.removed_modules:
            parts.append(f"-modules: {self.removed_modules}")
        if self.module_changes:
            parts.append(f"~modules: {list(self.module_changes.keys())}")
        if self.added_edges:
            parts.append(f"+edges: {len(self.added_edges)}")
        if self.removed_edges:
            parts.append(f"-edges: {len(self.removed_edges)}")
        if self.modified_edges:
            parts.append(f"~edges: {len(self.modified_edges)}")
        return "; ".join(parts) if parts else "No structural changes"

