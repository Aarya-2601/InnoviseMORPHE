"""
MORPHE Systems Backend Contract Implementation.
Provides:
- In-memory & SQLite-backed graph topology using NetworkX.
- Copy-on-write fork() for candidate sandbox evaluations.
- Atomic apply_mutation(action).
- Standardized evaluate(queryset).
- Version ledger commit() and rollback(version_id).
- Detailed diff(v1, v2).
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import networkx as nx
import numpy as np

from morphe.models.metadata import RecordMetadata
from morphe.models.topology import (
    EdgeData,
    ModuleManifest,
    Mutation,
    TopologyDiff,
    VersionCommit,
)
from morphe.storage.record_store import Record, RecordStore

logger = logging.getLogger(__name__)


class MorpheBackend:
    """
    Core graph memory backend for MORPHE.
    Manages topology, module manifests, ledger versioning, mutations, and sandbox forks.
    """

    def __init__(
        self,
        record_store: Optional[RecordStore] = None,
        graph: Optional[nx.DiGraph] = None,
        manifests: Optional[Dict[str, ModuleManifest]] = None,
        head_version: Optional[str] = None,
        ledger: Optional[Dict[str, VersionCommit]] = None,
        snapshots: Optional[Dict[str, Tuple[nx.DiGraph, Dict[str, ModuleManifest]]]] = None,
    ):
        self.record_store: RecordStore = record_store if record_store is not None else RecordStore(":memory:")
        self.graph: nx.DiGraph = graph.copy() if graph is not None else nx.DiGraph()
        
        # Dictionary of module_id -> ModuleManifest
        if manifests is not None:
            self.manifests: Dict[str, ModuleManifest] = {k: v.clone() for k, v in manifests.items()}
        else:
            self.manifests = {}

        self.ledger: Dict[str, VersionCommit] = dict(ledger) if ledger is not None else {}
        self.snapshots: Dict[str, Tuple[nx.DiGraph, Dict[str, ModuleManifest]]] = (
            {k: (g.copy(), {mid: m.clone() for mid, m in mfs.items()}) for k, (g, mfs) in snapshots.items()}
            if snapshots is not None
            else {}
        )
        self.head_version: Optional[str] = head_version
        self._version_counter: int = len(self.ledger)
        self._pending_mutation: Optional[Mutation] = None

        # If starting fresh with an empty ledger and empty graph, initialize baseline v0 if graph has nodes
        if not self.ledger and len(self.graph.nodes) > 0:
            self.commit(reason_card="Initial architecture baseline (v0)")

    # -------------------------------------------------------------------------
    # Core Topology Accessors & Management
    # -------------------------------------------------------------------------

    def add_module(
        self,
        module_id: str,
        name: str = "",
        charter: str = "",
        record_ids: Optional[Sequence[str]] = None,
        level: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ModuleManifest:
        """Register a new memory module."""
        rec_set = set(record_ids) if record_ids else set()
        manifest = ModuleManifest(
            module_id=module_id,
            name=name or module_id,
            charter=charter,
            record_ids=rec_set,
            level=level,
            extra=extra or {},
        )
        manifest.centroid = self._compute_centroid(rec_set)
        self.manifests[module_id] = manifest
        self.graph.add_node(module_id, charter=charter, level=level, count=len(rec_set))
        return manifest

    def remove_module(self, module_id: str) -> None:
        """Remove a module and all associated incident edges."""
        if module_id in self.manifests:
            del self.manifests[module_id]
        if module_id in self.graph:
            self.graph.remove_node(module_id)

    def add_edge(
        self,
        source: str,
        target: str,
        weight: float = 1.0,
        edge_type: str = "associative",
        valid_from: Optional[float] = None,
        valid_to: Optional[float] = None,
        co_access_count: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create or update a directed edge between two modules."""
        if source not in self.graph or target not in self.graph:
            raise KeyError(f"Both nodes must exist in graph. Missing source={source in self.graph}, target={target in self.graph}")
        
        edge_obj = EdgeData(
            source=source,
            target=target,
            weight=weight,
            edge_type=edge_type,
            valid_from=valid_from,
            valid_to=valid_to,
            co_access_count=co_access_count,
            extra=extra or {},
        )
        self.graph.add_edge(source, target, **edge_obj.to_dict())

    def get_edge(self, source: str, target: str) -> Optional[EdgeData]:
        """Retrieve edge metadata if edge exists."""
        if self.graph.has_edge(source, target):
            return EdgeData.from_dict(dict(self.graph[source][target]))
        return None

    def add_record_to_module(self, module_id: str, record_id: str) -> None:
        """Assign an immutable record ID to an existing module."""
        if module_id not in self.manifests:
            raise KeyError(f"Module '{module_id}' not found.")
        self.manifests[module_id].record_ids.add(record_id)
        self.manifests[module_id].centroid = self._compute_centroid(self.manifests[module_id].record_ids)
        if module_id in self.graph:
            self.graph.nodes[module_id]["count"] = len(self.manifests[module_id].record_ids)

    def _compute_centroid(self, record_ids: Set[str]) -> Optional[np.ndarray]:
        """Compute average embedding centroid for all records in this module."""
        if not record_ids:
            return None
        embeddings: List[np.ndarray] = []
        for r_id in record_ids:
            rec = self.record_store.get(r_id)
            if rec and rec.embedding is not None:
                embeddings.append(rec.embedding)
        if not embeddings:
            return None
        centroid = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-9:
            centroid = centroid / norm
        return centroid.astype(np.float32)

    # -------------------------------------------------------------------------
    # Copy-on-Write Sandbox Fork
    # -------------------------------------------------------------------------

    def fork(self) -> MorpheBackend:
        """
        Creates an isolated, lightweight sandbox copy of the current state.
        The underlying RecordStore is shared (as records are immutable), while
        the NetworkX graph and ModuleManifests are branched cleanly.
        """
        forked = MorpheBackend(
            record_store=self.record_store,
            graph=self.graph,
            manifests=self.manifests,
            head_version=self.head_version,
            ledger=self.ledger,
            snapshots=self.snapshots,
        )
        forked._version_counter = self._version_counter
        return forked

    # -------------------------------------------------------------------------
    # Mutation Engine
    # -------------------------------------------------------------------------

    def apply_mutation(self, action: Union[Mutation, Dict[str, Any]]) -> bool:
        """
        Apply a structural reconfiguration mutation to the current graph.
        Supported actions:
        - split: subdivide an overloaded module into 2 or more sub-modules
        - merge: consolidate 2 redundant modules into 1
        - shortcut: add/reinforce direct edge between frequently co-accessed modules
        - prune: remove cold edge or empty module
        - promote: create a parent summary hub (level=1) for a cluster of modules
        - do_nothing: no operation
        """
        if isinstance(action, dict):
            action = Mutation.from_dict(action)

        act_type = action.action_type.lower()
        targets = action.target_modules
        params = action.parameters

        if act_type == "do_nothing":
            self._pending_mutation = action
            return True

        elif act_type == "split":
            if not targets or targets[0] not in self.manifests:
                raise ValueError(f"Split requires a valid target module: {targets}")
            src_mid = targets[0]
            src_manifest = self.manifests[src_mid]
            records = list(src_manifest.record_ids)

            # Sub-module IDs
            sub1_id = params.get("sub_module_1", f"{src_mid}_a")
            sub2_id = params.get("sub_module_2", f"{src_mid}_b")
            charter1 = params.get("charter_1", f"{src_manifest.charter} (Part A)")
            charter2 = params.get("charter_2", f"{src_manifest.charter} (Part B)")

            # Determine split partition
            partition1: List[str] = []
            partition2: List[str] = []
            if "split_records" in params:
                partition1 = params["split_records"].get(sub1_id, [])
                partition2 = params["split_records"].get(sub2_id, [])
            elif len(records) > 1:
                # Default half-split if no explicit partition provided
                midpoint = len(records) // 2
                partition1 = records[:midpoint]
                partition2 = records[midpoint:]
            else:
                partition1 = records
                partition2 = []

            # Create new sub-modules
            self.add_module(sub1_id, name=sub1_id, charter=charter1, record_ids=partition1, level=src_manifest.level)
            self.add_module(sub2_id, name=sub2_id, charter=charter2, record_ids=partition2, level=src_manifest.level)

            # Re-link existing neighbors of src_mid to the new submodules
            in_edges = list(self.graph.in_edges(src_mid, data=True))
            out_edges = list(self.graph.out_edges(src_mid, data=True))

            for u, _, edata in in_edges:
                w = edata.get("weight", 1.0)
                etype = edata.get("edge_type", "associative")
                self.add_edge(u, sub1_id, weight=w, edge_type=etype)
                self.add_edge(u, sub2_id, weight=w, edge_type=etype)

            for _, v, edata in out_edges:
                w = edata.get("weight", 1.0)
                etype = edata.get("edge_type", "associative")
                self.add_edge(sub1_id, v, weight=w, edge_type=etype)
                self.add_edge(sub2_id, v, weight=w, edge_type=etype)

            # Link sub-modules to each other
            self.add_edge(sub1_id, sub2_id, weight=0.8, edge_type="associative")
            self.add_edge(sub2_id, sub1_id, weight=0.8, edge_type="associative")

            # Remove original split parent
            self.remove_module(src_mid)
            self._pending_mutation = action
            return True

        elif act_type == "merge":
            if len(targets) < 2 or targets[0] not in self.manifests or targets[1] not in self.manifests:
                raise ValueError(f"Merge requires two valid target modules: {targets}")
            m1, m2 = targets[0], targets[1]
            man1, man2 = self.manifests[m1], self.manifests[m2]

            merged_id = params.get("merged_id", f"{m1}_{m2}_merged")
            merged_charter = params.get("charter", f"{man1.charter} + {man2.charter}")
            merged_records = set(man1.record_ids) | set(man2.record_ids)

            # Add merged module
            self.add_module(
                merged_id,
                name=params.get("name", merged_id),
                charter=merged_charter,
                record_ids=list(merged_records),
                level=max(man1.level, man2.level),
            )

            # Inherit and deduplicate edges from both modules
            for old_m in (m1, m2):
                for u, _, edata in list(self.graph.in_edges(old_m, data=True)):
                    if u not in (m1, m2):
                        w = edata.get("weight", 1.0)
                        self.add_edge(u, merged_id, weight=w, edge_type=edata.get("edge_type", "associative"))
                for _, v, edata in list(self.graph.out_edges(old_m, data=True)):
                    if v not in (m1, m2):
                        w = edata.get("weight", 1.0)
                        self.add_edge(merged_id, v, weight=w, edge_type=edata.get("edge_type", "associative"))

            # Remove original modules
            self.remove_module(m1)
            self.remove_module(m2)
            self._pending_mutation = action
            return True

        elif act_type == "shortcut":
            if len(targets) < 2 or targets[0] not in self.manifests or targets[1] not in self.manifests:
                raise ValueError(f"Shortcut requires two existing module targets: {targets}")
            src, dst = targets[0], targets[1]
            weight = float(params.get("weight", 1.0))
            bidirectional = bool(params.get("bidirectional", True))

            self.add_edge(src, dst, weight=weight, edge_type="shortcut")
            if bidirectional:
                self.add_edge(dst, src, weight=weight, edge_type="shortcut")
            self._pending_mutation = action
            return True

        elif act_type == "prune":
            target_edge = params.get("edge")
            if target_edge and len(target_edge) == 2:
                u, v = target_edge[0], target_edge[1]
                if self.graph.has_edge(u, v):
                    self.graph.remove_edge(u, v)
            elif targets:
                for t in targets:
                    self.remove_module(t)
            self._pending_mutation = action
            return True

        elif act_type == "promote":
            # Hierarchical evolution: summary parent hub created over a cluster of modules
            if not targets:
                raise ValueError("Promote requires target child modules to summarize.")
            hub_id = params.get("hub_id", f"hub_{int(time.time())}")
            hub_charter = params.get("charter", f"Summary Hub for {targets}")
            self.add_module(hub_id, name=hub_id, charter=hub_charter, record_ids=[], level=1)
            for child in targets:
                if child in self.manifests:
                    self.add_edge(hub_id, child, weight=1.0, edge_type="hierarchical")
                    self.add_edge(child, hub_id, weight=0.5, edge_type="hierarchical")
            self._pending_mutation = action
            return True

        else:
            raise ValueError(f"Unknown mutation action_type: '{act_type}'")

    # -------------------------------------------------------------------------
    # Evaluation Against Queryset
    # -------------------------------------------------------------------------

    def evaluate(
        self,
        queryset: Sequence[Dict[str, Any]],
        evaluator_fn: Optional[Callable[[MorpheBackend, Sequence[Dict[str, Any]]], Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate current graph retrieval performance against a benchmark queryset.
        Returns standardized metrics: Recall@5, MRR, latency_ms, modules_touched, structural_cost.
        """
        if evaluator_fn is not None:
            return evaluator_fn(self, queryset)

        # Built-in lightweight evaluator
        if not queryset:
            return {
                "recall_at_5": 1.0,
                "mrr": 1.0,
                "latency_ms": 1.0,
                "modules_touched": 1.0,
                "structural_cost": float(len(self.graph.nodes) + 0.2 * len(self.graph.edges)),
                "fitness": 1.0,
            }

        recalls: List[float] = []
        mrrs: List[float] = []
        latencies: List[float] = []
        modules_touched_list: List[int] = []

        total_nodes = len(self.graph.nodes)
        if total_nodes == 0:
            return {
                "recall_at_5": 0.0,
                "mrr": 0.0,
                "latency_ms": 0.0,
                "modules_touched": 0.0,
                "structural_cost": 0.0,
                "fitness": 0.0,
            }

        for q in queryset:
            t0 = time.perf_counter()
            gold_id = q.get("gold_id")
            gold_module = q.get("gold_module")
            q_emb = q.get("embedding")

            touched_count = 1
            hit_rank = 0

            # If embedding present, rank modules by centroid cosine similarity
            if q_emb is not None:
                q_vec = np.asarray(q_emb, dtype=np.float32)
                q_norm = np.linalg.norm(q_vec)
                if q_norm > 1e-9:
                    q_vec = q_vec / q_norm

                mod_scores: List[Tuple[str, float]] = []
                for mid, man in self.manifests.items():
                    if man.centroid is not None:
                        sim = float(np.dot(q_vec, man.centroid))
                    else:
                        sim = 0.0
                    mod_scores.append((mid, sim))
                mod_scores.sort(key=lambda x: x[1], reverse=True)
                top_mids = [m for m, _ in mod_scores[:5]]
                touched_count = min(3, len(top_mids))

                # Check if gold record is in any of top retrieved modules
                found = False
                for r_idx, mid in enumerate(top_mids, start=1):
                    man = self.manifests.get(mid)
                    if man and ((gold_id and gold_id in man.record_ids) or (gold_module and gold_module == mid)):
                        hit_rank = r_idx
                        found = True
                        break
                recalls.append(1.0 if found else 0.0)
                mrrs.append(1.0 / hit_rank if hit_rank > 0 else 0.0)
            else:
                # Text/keyword heuristic lookup fallback
                q_text = str(q.get("query", "")).lower()
                found = False
                for r_idx, (mid, man) in enumerate(self.manifests.items(), start=1):
                    if (gold_id and gold_id in man.record_ids) or (gold_module and gold_module == mid) or (q_text in man.charter.lower()):
                        hit_rank = r_idx
                        found = True
                        break
                recalls.append(1.0 if found else 0.0)
                mrrs.append(1.0 / hit_rank if hit_rank > 0 else 0.0)

            t_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(t_elapsed_ms)
            modules_touched_list.append(touched_count)

        avg_recall = float(np.mean(recalls)) if recalls else 0.0
        avg_mrr = float(np.mean(mrrs)) if mrrs else 0.0
        avg_latency = float(np.mean(latencies)) if latencies else 0.0
        avg_touched = float(np.mean(modules_touched_list)) if modules_touched_list else 1.0
        structural_cost = float(len(self.graph.nodes) + 0.2 * len(self.graph.edges))

        # Multi-objective fitness
        fitness = float(0.4 * avg_recall + 0.3 * avg_mrr - 0.01 * avg_latency - 0.05 * avg_touched - 0.01 * structural_cost)

        return {
            "recall_at_5": round(avg_recall, 4),
            "mrr": round(avg_mrr, 4),
            "latency_ms": round(avg_latency, 3),
            "modules_touched": round(avg_touched, 2),
            "structural_cost": round(structural_cost, 2),
            "fitness": round(fitness, 4),
        }

    # -------------------------------------------------------------------------
    # Ledger Versioning, Commit & Rollback
    # -------------------------------------------------------------------------

    def commit(
        self,
        reason_card: str = "",
        metrics: Optional[Dict[str, float]] = None,
    ) -> VersionCommit:
        """
        Record the current graph state as an immutable commit in the ledger.
        """
        self._version_counter += 1
        version_id = f"v{self._version_counter}"
        
        # Snapshot current topology graph and manifests
        snap_graph = self.graph.copy()
        snap_manifests = {mid: m.clone() for mid, m in self.manifests.items()}
        self.snapshots[version_id] = (snap_graph, snap_manifests)

        total_recs = sum(len(m.record_ids) for m in self.manifests.values())
        commit_obj = VersionCommit(
            version_id=version_id,
            parent_version_id=self.head_version,
            timestamp=time.time(),
            mutation_applied=self._pending_mutation.to_dict() if self._pending_mutation else None,
            metrics=dict(metrics or {}),
            reason_card=reason_card or ("Applied mutation" if self._pending_mutation else "Baseline commit"),
            module_count=len(self.graph.nodes),
            edge_count=len(self.graph.edges),
            total_records=total_recs,
        )
        self.ledger[version_id] = commit_obj
        self.head_version = version_id
        self._pending_mutation = None
        return commit_obj

    def rollback(self, version_id: str) -> bool:
        """
        Revert the graph topology and manifests to the exact snapshot at version_id.
        """
        if version_id not in self.snapshots:
            raise KeyError(f"Version '{version_id}' not found in ledger snapshots.")

        snap_graph, snap_manifests = self.snapshots[version_id]
        self.graph = snap_graph.copy()
        self.manifests = {mid: m.clone() for mid, m in snap_manifests.items()}
        self.head_version = version_id
        self._pending_mutation = None
        return True

    def list_versions(self) -> List[VersionCommit]:
        """Return list of all commits in chronological order."""
        return list(self.ledger.values())

    def get_commit(self, version_id: str) -> Optional[VersionCommit]:
        """Fetch metadata for a specific commit version."""
        return self.ledger.get(version_id)

    # -------------------------------------------------------------------------
    # Topology Diffing
    # -------------------------------------------------------------------------

    def diff(self, v1: str, v2: str) -> TopologyDiff:
        """
        Compute precise structural and metric difference between two versions.
        """
        if v1 not in self.snapshots or v2 not in self.snapshots:
            raise KeyError(f"Both versions must exist to diff: v1 in snapshots={v1 in self.snapshots}, v2 in snapshots={v2 in self.snapshots}")

        g1, m1 = self.snapshots[v1]
        g2, m2 = self.snapshots[v2]

        mods1 = set(m1.keys())
        mods2 = set(m2.keys())

        added_mods = sorted(list(mods2 - mods1))
        removed_mods = sorted(list(mods1 - mods2))
        
        modified_mods: List[str] = []
        for mid in mods1 & mods2:
            if m1[mid].record_ids != m2[mid].record_ids or m1[mid].charter != m2[mid].charter:
                modified_mods.append(mid)

        edges1 = set(g1.edges())
        edges2 = set(g2.edges())

        added_edges: List[Dict[str, Any]] = [
            {"source": u, "target": v, **dict(g2[u][v])}
            for u, v in (edges2 - edges1)
        ]
        removed_edges: List[Tuple[str, str]] = list(edges1 - edges2)

        modified_edges: List[Dict[str, Any]] = []
        for u, v in edges1 & edges2:
            d1 = dict(g1[u][v])
            d2 = dict(g2[u][v])
            if d1 != d2:
                modified_edges.append({"source": u, "target": v, "before": d1, "after": d2})

        # Metric deltas
        c1 = self.ledger.get(v1)
        c2 = self.ledger.get(v2)
        metric_deltas: Dict[str, float] = {}
        if c1 and c2:
            for k in set(c1.metrics.keys()) | set(c2.metrics.keys()):
                val1 = c1.metrics.get(k, 0.0)
                val2 = c2.metrics.get(k, 0.0)
                metric_deltas[k] = round(val2 - val1, 4)

        return TopologyDiff(
            v1=v1,
            v2=v2,
            added_modules=added_mods,
            removed_modules=removed_mods,
            modified_modules=modified_mods,
            added_edges=added_edges,
            removed_edges=removed_edges,
            modified_edges=modified_edges,
            metric_deltas=metric_deltas,
        )
