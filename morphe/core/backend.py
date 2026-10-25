"""
MORPHE Backend implementing the systems contract:
- fork(): Lightweight copy-on-write sandbox sharing immutable records
- apply_mutation(action): Applies atomic topological change to sandbox
- commit(reason_card, metrics): Commits current staging topology as a new version
- rollback(version_id): Moves HEAD pointer to target version
- list_versions(): Returns ordered version history with reason cards & metrics
- diff(v1, v2): Computes structural differences between any two versions
- NetworkX graph synchronization for graph algorithms and visualization
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import networkx as nx
import numpy as np

from morphe.storage.record_store import RecordStore
from morphe.topology.diff import EdgeDiff, ModuleDiff, TopologyDiff
from morphe.topology.models import Edge, EdgeType, ModuleManifest, TopologySnapshot, VersionCommit
from morphe.topology.mutations import MutationAction, MutationType


class MorpheBackend:
    """
    MORPHE Graph & Versioning Backend.
    Maintains:
    - Immutable RecordStore reference
    - Version history tree (ledger)
    - Active HEAD version pointer
    - Working / Staging Topology (modules, edges, and NetworkX DiGraph)
    """

    def __init__(
        self,
        record_store: Optional[RecordStore] = None,
        initial_snapshot: Optional[TopologySnapshot] = None,
    ):
        self.record_store = record_store if record_store is not None else RecordStore(":memory:")
        self.ledger: Dict[str, VersionCommit] = {}
        self.version_history: List[str] = []
        self.head_version_id: Optional[str] = None

        # Staging / Working state
        self._staged_modules: Dict[str, ModuleManifest] = {}
        self._staged_edges: Dict[Tuple[str, str], Edge] = {}
        self._graph = nx.DiGraph()

        if initial_snapshot is not None:
            self._load_snapshot_into_staging(initial_snapshot)
            # Create root commit
            root_commit = VersionCommit.create(
                snapshot=initial_snapshot,
                parent_version=None,
                reason_card="Root initialization",
                metrics={},
            )
            self._record_commit(root_commit)

    def _load_snapshot_into_staging(self, snapshot: TopologySnapshot) -> None:
        """Load snapshot into current staging structures and sync NetworkX graph."""
        self._staged_modules = dict(snapshot.modules)
        self._staged_edges = {(e.source, e.target): e for e in snapshot.edges}
        self._sync_nx_graph()

    def _sync_nx_graph(self) -> None:
        """Re-synchronize internal NetworkX DiGraph from staged modules and edges."""
        self._graph.clear()
        for mod_id, manifest in self._staged_modules.items():
            self._graph.add_node(
                mod_id,
                record_count=manifest.record_count,
                charter=manifest.charter,
                tags=list(manifest.tags),
                parent=manifest.parent_module_id,
            )
        for (src, tgt), edge in self._staged_edges.items():
            self._graph.add_edge(
                src,
                tgt,
                weight=edge.weight,
                edge_type=edge.edge_type.value,
                valid_from=edge.valid_from,
                valid_to=edge.valid_to,
            )

    @property
    def graph(self) -> nx.DiGraph:
        """Current NetworkX graph view."""
        return self._graph

    def get_staged_snapshot(self) -> TopologySnapshot:
        """Produce an immutable snapshot of the current staging state."""
        return TopologySnapshot.create(
            modules=self._staged_modules,
            edges=list(self._staged_edges.values()),
        )

    # -------------------------------------------------------------
    # 1. Copy-on-Write Sandbox Fork
    # -------------------------------------------------------------
    def fork(self) -> MorpheBackend:
        """
        Creates a lightweight Copy-on-Write (CoW) sandbox fork.
        Shares the underlying immutable record store, but clones staged manifests
        and edge topology so mutations can be trialed in isolation without touching HEAD.
        """
        sandbox = MorpheBackend(record_store=self.record_store)
        sandbox.ledger = dict(self.ledger)
        sandbox.version_history = list(self.version_history)
        sandbox.head_version_id = self.head_version_id

        # Copy manifests and edges map (manifests themselves are immutable)
        sandbox._staged_modules = dict(self._staged_modules)
        sandbox._staged_edges = dict(self._staged_edges)
        sandbox._sync_nx_graph()
        return sandbox

    # -------------------------------------------------------------
    # 2. Apply Mutations
    # -------------------------------------------------------------
    def apply_mutation(self, action: MutationAction) -> bool:
        """
        Apply an atomic topological mutation to the staging area.
        Returns True if successful, False if invalid or rejected.
        """
        mtype = action.mutation_type

        if mtype == MutationType.ADD_EDGE:
            if not action.source_module or not action.target_module:
                return False
            weight = action.weight if action.weight is not None else 1.0
            edge = Edge(
                source=action.source_module,
                target=action.target_module,
                weight=weight,
                edge_type=action.edge_type,
            )
            self._staged_edges[(edge.source, edge.target)] = edge
            self._sync_nx_graph()
            return True

        elif mtype == MutationType.REMOVE_EDGE:
            key = (action.source_module, action.target_module)
            if key in self._staged_edges:
                del self._staged_edges[key]
                self._sync_nx_graph()
                return True
            return False

        elif mtype == MutationType.UPDATE_EDGE_WEIGHT:
            key = (action.source_module, action.target_module)
            if key in self._staged_edges and action.weight is not None:
                old = self._staged_edges[key]
                self._staged_edges[key] = Edge(
                    source=old.source,
                    target=old.target,
                    weight=action.weight,
                    edge_type=old.edge_type,
                    valid_from=old.valid_from,
                    valid_to=old.valid_to,
                    metadata=old.metadata,
                )
                self._sync_nx_graph()
                return True
            return False

        elif mtype == MutationType.ADD_MODULE:
            if not action.new_module_id:
                return False
            mod_id = action.new_module_id
            manifest = ModuleManifest.create(
                module_id=mod_id,
                record_ids=action.record_ids,
                charter=action.charter or "",
            )
            self._staged_modules[mod_id] = manifest
            self._sync_nx_graph()
            return True

        elif mtype == MutationType.REMOVE_MODULE:
            mod_id = action.source_module
            if mod_id in self._staged_modules:
                del self._staged_modules[mod_id]
                # Prune incident edges
                self._staged_edges = {
                    (s, t): e for (s, t), e in self._staged_edges.items()
                    if s != mod_id and t != mod_id
                }
                self._sync_nx_graph()
                return True
            return False

        elif mtype == MutationType.SPLIT_MODULE:
            src_id = action.source_module
            new_id = action.new_module_id
            if not src_id or not new_id or src_id not in self._staged_modules:
                return False

            old_manifest = self._staged_modules[src_id]
            split_set = set(action.record_ids)
            remaining_records = [r for r in old_manifest.record_ids if r not in split_set]

            # Replace original module manifest
            self._staged_modules[src_id] = old_manifest.with_records(remaining_records)
            # Create new module manifest
            self._staged_modules[new_id] = ModuleManifest.create(
                module_id=new_id,
                record_ids=list(split_set),
                charter=action.charter or f"Split cluster from {src_id}",
            )
            # Automatically connect split sibling modules
            self._staged_edges[(src_id, new_id)] = Edge(src_id, new_id, weight=0.8, edge_type=EdgeType.REGULAR)
            self._staged_edges[(new_id, src_id)] = Edge(new_id, src_id, weight=0.8, edge_type=EdgeType.REGULAR)
            self._sync_nx_graph()
            return True

        elif mtype == MutationType.MERGE_MODULES:
            src_id = action.source_module
            tgt_id = action.target_module
            if not src_id or not tgt_id or src_id not in self._staged_modules or tgt_id not in self._staged_modules:
                return False

            src_manifest = self._staged_modules[src_id]
            tgt_manifest = self._staged_modules[tgt_id]
            combined_records = list(set(src_manifest.record_ids) | set(tgt_manifest.record_ids))

            # Update target module with merged records
            merged_charter = action.charter or f"{tgt_manifest.charter}; {src_manifest.charter}".strip("; ")
            self._staged_modules[tgt_id] = ModuleManifest.create(
                module_id=tgt_id,
                record_ids=combined_records,
                charter=merged_charter,
            )
            # Remove source module
            del self._staged_modules[src_id]

            # Rewire edges: redirect edges connected to src_id over to tgt_id
            new_edges: Dict[Tuple[str, str], Edge] = {}
            for (s, t), e in self._staged_edges.items():
                ns = tgt_id if s == src_id else s
                nt = tgt_id if t == src_id else t
                if ns != nt:  # no self-loops
                    new_edges[(ns, nt)] = Edge(ns, nt, weight=e.weight, edge_type=e.edge_type)
            self._staged_edges = new_edges
            self._sync_nx_graph()
            return True

        return False

    # -------------------------------------------------------------
    # 3. Commit
    # -------------------------------------------------------------
    def commit(
        self,
        reason_card: str = "",
        metrics: Optional[Dict[str, float]] = None,
        version_id: Optional[str] = None,
    ) -> VersionCommit:
        """
        Commits staged topology as a new immutable version in the ledger.
        Updates the HEAD pointer.
        """
        snapshot = self.get_staged_snapshot()
        commit_obj = VersionCommit.create(
            snapshot=snapshot,
            parent_version=self.head_version_id,
            reason_card=reason_card,
            metrics=metrics or {},
            version_id=version_id,
        )
        self._record_commit(commit_obj)
        return commit_obj

    def _record_commit(self, commit_obj: VersionCommit) -> None:
        self.ledger[commit_obj.version_id] = commit_obj
        self.version_history.append(commit_obj.version_id)
        self.head_version_id = commit_obj.version_id

    # -------------------------------------------------------------
    # 4. Rollback
    # -------------------------------------------------------------
    def rollback(self, target_version_id: str) -> bool:
        """
        Rollback staging topology to match the snapshot at target_version_id.
        Moves HEAD pointer to target_version_id.
        """
        if target_version_id not in self.ledger:
            return False

        target_commit = self.ledger[target_version_id]
        self._load_snapshot_into_staging(target_commit.snapshot)
        self.head_version_id = target_version_id
        return True

    # -------------------------------------------------------------
    # 5. List Versions
    # -------------------------------------------------------------
    def list_versions(self) -> List[Dict[str, Any]]:
        """
        Returns list of all committed versions with reason card, parent, and metrics.
        """
        out: List[Dict[str, Any]] = []
        for vid in self.version_history:
            commit_obj = self.ledger[vid]
            out.append({
                "version_id": commit_obj.version_id,
                "parent_version": commit_obj.parent_version,
                "reason_card": commit_obj.reason_card,
                "metrics": commit_obj.metrics,
                "module_count": len(commit_obj.snapshot.modules),
                "edge_count": len(commit_obj.snapshot.edges),
                "created_at": commit_obj.created_at,
                "is_head": (commit_obj.version_id == self.head_version_id),
            })
        return out

    # -------------------------------------------------------------
    # 6. Diff(v1, v2)
    # -------------------------------------------------------------
    def diff(self, v1_id: str, v2_id: str) -> TopologyDiff:
        """
        Computes detailed diff between two version commits.
        """
        if v1_id not in self.ledger or v2_id not in self.ledger:
            raise KeyError(f"Invalid version IDs: {v1_id} or {v2_id}")

        c1 = self.ledger[v1_id]
        c2 = self.ledger[v2_id]
        snap1 = c1.snapshot
        snap2 = c2.snapshot

        diff_res = TopologyDiff(v1_id=v1_id, v2_id=v2_id)

        # 1. Module additions & removals
        mods1 = set(snap1.modules.keys())
        mods2 = set(snap2.modules.keys())

        diff_res.added_modules = sorted(list(mods2 - mods1))
        diff_res.removed_modules = sorted(list(mods1 - mods2))

        # 2. Module internal changes (records and charters)
        for m_id in mods1 & mods2:
            m1 = snap1.modules[m_id]
            m2 = snap2.modules[m_id]
            recs1 = set(m1.record_ids)
            recs2 = set(m2.record_ids)

            added_recs = sorted(list(recs2 - recs1))
            removed_recs = sorted(list(recs1 - recs2))
            charter_changed = (m1.charter != m2.charter)

            if added_recs or removed_recs or charter_changed:
                diff_res.module_changes[m_id] = ModuleDiff(
                    module_id=m_id,
                    added_records=added_recs,
                    removed_records=removed_recs,
                    charter_changed=charter_changed,
                    old_charter=m1.charter,
                    new_charter=m2.charter,
                )

        # 3. Edge changes
        edges1_map = {(e.source, e.target): e for e in snap1.edges}
        edges2_map = {(e.source, e.target): e for e in snap2.edges}

        keys1 = set(edges1_map.keys())
        keys2 = set(edges2_map.keys())

        diff_res.added_edges = [edges2_map[k] for k in sorted(list(keys2 - keys1))]
        diff_res.removed_edges = [edges1_map[k] for k in sorted(list(keys1 - keys2))]

        for k in sorted(list(keys1 & keys2)):
            e1 = edges1_map[k]
            e2 = edges2_map[k]
            if round(e1.weight, 4) != round(e2.weight, 4) or e1.edge_type != e2.edge_type:
                diff_res.modified_edges.append(
                    EdgeDiff(
                        source=k[0],
                        target=k[1],
                        old_weight=e1.weight,
                        new_weight=e2.weight,
                        old_type=e1.edge_type.value,
                        new_type=e2.edge_type.value,
                    )
                )

        # 4. Metric deltas
        for metric_name, val2 in c2.metrics.items():
            if metric_name in c1.metrics:
                diff_res.metric_deltas[metric_name] = round(val2 - c1.metrics[metric_name], 4)

        return diff_res

