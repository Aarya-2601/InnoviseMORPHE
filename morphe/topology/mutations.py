"""
Action / Mutation model for MORPHE graph topology.
Operations supported:
- ADD_EDGE: Add edge between modules (regular or shortcut)
- REMOVE_EDGE / PRUNE_EDGE: Remove edge
- UPDATE_EDGE_WEIGHT: Update edge weight (plasticity)
- ADD_MODULE: Add a new module manifest
- REMOVE_MODULE: Remove a module
- SPLIT_MODULE: Split source module into two modules
- MERGE_MODULES: Merge two modules into a combined module
- REASSIGN_RECORDS: Move records between modules
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from morphe.topology.models import Edge, EdgeType, ModuleManifest


class MutationType(str, Enum):
    ADD_EDGE = "add_edge"
    REMOVE_EDGE = "remove_edge"
    UPDATE_EDGE_WEIGHT = "update_edge_weight"
    ADD_MODULE = "add_module"
    REMOVE_MODULE = "remove_module"
    SPLIT_MODULE = "split_module"
    MERGE_MODULES = "merge_modules"
    REASSIGN_RECORDS = "reassign_records"


@dataclass(frozen=True)
class MutationAction:
    """
    Structured atomic mutation proposed on the MORPHE topology graph.
    """
    mutation_type: MutationType
    source_module: Optional[str] = None
    target_module: Optional[str] = None
    new_module_id: Optional[str] = None
    weight: Optional[float] = None
    edge_type: EdgeType = EdgeType.REGULAR
    record_ids: Tuple[str, ...] = field(default_factory=tuple)
    charter: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    @classmethod
    def add_edge(
        cls,
        source: str,
        target: str,
        weight: float = 1.0,
        edge_type: EdgeType = EdgeType.REGULAR,
    ) -> MutationAction:
        return cls(
            mutation_type=MutationType.ADD_EDGE,
            source_module=source,
            target_module=target,
            weight=weight,
            edge_type=edge_type,
        )

    @classmethod
    def remove_edge(cls, source: str, target: str) -> MutationAction:
        return cls(
            mutation_type=MutationType.REMOVE_EDGE,
            source_module=source,
            target_module=target,
        )

    @classmethod
    def update_edge_weight(cls, source: str, target: str, weight: float) -> MutationAction:
        return cls(
            mutation_type=MutationType.UPDATE_EDGE_WEIGHT,
            source_module=source,
            target_module=target,
            weight=weight,
        )

    @classmethod
    def split_module(
        cls,
        module_id: str,
        new_module_id: str,
        migrated_record_ids: Sequence[str],
        new_module_charter: str = "",
    ) -> MutationAction:
        return cls(
            mutation_type=MutationType.SPLIT_MODULE,
            source_module=module_id,
            new_module_id=new_module_id,
            record_ids=tuple(migrated_record_ids),
            charter=new_module_charter,
        )

    @classmethod
    def merge_modules(
        cls,
        source_module: str,
        target_module: str,
        merged_charter: str = "",
    ) -> MutationAction:
        return cls(
            mutation_type=MutationType.MERGE_MODULES,
            source_module=source_module,
            target_module=target_module,
            charter=merged_charter,
        )
