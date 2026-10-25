"""
MORPHE Mutation Engine: split, merge, shortcut, prune, promote.
Implements Section 5.1 & 5.1a of the MORPHE Final Plan.

Eligibility & Execution Rules:
1. Split:
   - Eligible when module is overloaded or internally mixed (record_count > split_threshold
     or 2-way clustering silhouette / intra-module variance above threshold).
   - Operation: 2-means clustering on embeddings splits records into two coherent clusters.
2. Merge:
   - Eligible when two modules are redundant (centroid cosine similarity > merge_cosine_threshold
     and combined size under split limit).
   - Operation: Combine records into target module and rewire edges.
3. Shortcut:
   - Eligible when modules are repeatedly co-accessed without a direct edge
     (co_access_count >= shortcut_coaccess_threshold).
   - Operation: Creates a SHORTCUT edge between the co-accessed pair.
4. Prune:
   - Eligible when an edge or module is cold (edge weight < prune_epsilon for M consecutive cycles).
   - Operation: Removes stale or low-utility edge.
5. Promote:
   - Eligible when a stable cluster of modules (>= min_cluster_size) has persistent co-access across cycles.
   - Operation: Creates a parent summary module and HIERARCHICAL edges down to member modules.
6. Do Nothing:
   - Always eligible fallback.
"""

from __future__ import annotations

import collections
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from morphe.adaptation.candidates import MutationCandidate
from morphe.core.backend import MorpheBackend
from morphe.storage.record_store import RecordStore
from morphe.topology.models import Edge, EdgeType, ModuleManifest
from morphe.topology.mutations import MutationAction, MutationType


class MutationEngine:
    """
    Evaluates topology state and telemetry to discover and execute eligible mutations:
    Split, Merge, Shortcut, Prune, Promote, Do Nothing.
    """

    def __init__(
        self,
        split_size_threshold: int = 150,
        split_silhouette_threshold: float = 0.25,
        merge_cosine_threshold: float = 0.90,
        shortcut_coaccess_threshold: int = 10,
        prune_epsilon: float = 0.10,
        promote_min_cluster_size: int = 3,
    ):
        self.split_size_threshold = split_size_threshold
        self.split_silhouette_threshold = split_silhouette_threshold
        self.merge_cosine_threshold = merge_cosine_threshold
        self.shortcut_coaccess_threshold = shortcut_coaccess_threshold
        self.prune_epsilon = prune_epsilon
        self.promote_min_cluster_size = promote_min_cluster_size

    # ------------------------------------------------------------------
    # 1. Eligibility Discovery
    # ------------------------------------------------------------------

    def find_eligible_splits(
        self,
        backend: MorpheBackend,
    ) -> List[MutationCandidate]:
        """
        Identify modules eligible for splitting:
        Overloaded by record count (> split_size_threshold) or internally bimodal.
        """
        candidates: List[MutationCandidate] = []
        record_store = backend.record_store
        snapshot = backend.get_staged_snapshot()

        for mod_id, manifest in snapshot.modules.items():
            count = manifest.record_count
            if count < 4:  # minimum records required to split into two non-trivial clusters
                continue

            # Fetch embeddings for records in this module
            records = record_store.get_many(manifest.record_ids)
            embs = [r.embedding for r in records.values() if r.embedding is not None]

            # Condition 1: Size overload
            if count >= self.split_size_threshold:
                # Propose split
                split_action = self._propose_split_clustering(mod_id, manifest, records, embs)
                if split_action is not None:
                    candidates.append(
                        MutationCandidate(
                            action=split_action,
                            priority_score=float(count) / float(self.split_size_threshold),
                            symptom="overloaded",
                            evidence={"record_count": count, "threshold": self.split_size_threshold},
                        )
                    )
                continue

            # Condition 2: 2-way clustering silhouette check if enough embeddings exist
            if len(embs) >= 6:
                emb_matrix = np.stack(embs, axis=0)
                try:
                    km = KMeans(n_clusters=2, random_state=42, n_init="auto")
                    labels = km.fit_predict(emb_matrix)
                    if len(set(labels)) == 2:
                        sil = float(silhouette_score(emb_matrix, labels))
                        if sil >= self.split_silhouette_threshold:
                            split_action = self._propose_split_clustering(
                                mod_id, manifest, records, embs, labels=labels
                            )
                            if split_action is not None:
                                candidates.append(
                                    MutationCandidate(
                                        action=split_action,
                                        priority_score=1.0 + sil,
                                        symptom="fragmented_topic",
                                        evidence={"silhouette": sil, "record_count": count},
                                    )
                                )
                except Exception:
                    pass

        return candidates

    def _propose_split_clustering(
        self,
        mod_id: str,
        manifest: ModuleManifest,
        records: Dict[str, Any],
        embs: List[np.ndarray],
        labels: Optional[np.ndarray] = None,
    ) -> Optional[MutationAction]:
        """Split records into two clusters via 2-means or half-split."""
        all_ids = list(manifest.record_ids)
        if labels is None:
            if len(embs) == len(all_ids):
                matrix = np.stack(embs, axis=0)
                # Check for zero variance / identical vectors before invoking KMeans
                # to avoid ConvergenceWarning: Number of distinct clusters (1) < n_clusters (2)
                var = np.var(matrix, axis=0).sum()
                if var < 1e-6:
                    # All vectors are effectively identical: split by half deterministically
                    mid = len(all_ids) // 2
                    cluster1_ids = all_ids[mid:]
                    new_id = f"{mod_id}_sub1"
                    return MutationAction.split_module(
                        module_id=mod_id,
                        new_module_id=new_id,
                        migrated_record_ids=cluster1_ids,
                        new_module_charter=f"Sub-cluster split from {mod_id}",
                    )

                km = KMeans(n_clusters=2, random_state=42, n_init="auto")
                labels = km.fit_predict(matrix)
            else:
                # Deterministic partition
                mid = len(all_ids) // 2
                migrated = all_ids[mid:]
                new_id = f"{mod_id}_sub1"
                return MutationAction.split_module(
                    module_id=mod_id,
                    new_module_id=new_id,
                    migrated_record_ids=migrated,
                    new_module_charter=f"Sub-cluster split from {mod_id}",
                )

        cluster1_ids = [all_ids[i] for i, lbl in enumerate(labels) if lbl == 1]
        if not cluster1_ids or len(cluster1_ids) == len(all_ids):
            # Fallback to half split
            cluster1_ids = all_ids[len(all_ids) // 2:]

        new_id = f"{mod_id}_sub1"
        return MutationAction.split_module(
            module_id=mod_id,
            new_module_id=new_id,
            migrated_record_ids=cluster1_ids,
            new_module_charter=f"Sub-cluster split from {mod_id}",
        )

    def find_eligible_merges(
        self,
        backend: MorpheBackend,
    ) -> List[MutationCandidate]:
        """
        Identify redundant modules eligible for merging:
        Centroid cosine similarity >= merge_cosine_threshold and combined size <= split_size_threshold.
        """
        candidates: List[MutationCandidate] = []
        snapshot = backend.get_staged_snapshot()
        module_items = list(snapshot.modules.items())

        for i in range(len(module_items)):
            id1, m1 = module_items[i]
            if m1.centroid is None:
                continue
            for j in range(i + 1, len(module_items)):
                id2, m2 = module_items[j]
                if m2.centroid is None:
                    continue

                combined_size = m1.record_count + m2.record_count
                if combined_size > self.split_size_threshold:
                    continue

                # Cosine similarity between L2-normalized centroids
                cosine_sim = float(np.dot(m1.centroid, m2.centroid))
                if cosine_sim >= self.merge_cosine_threshold:
                    action = MutationAction.merge_modules(
                        source_module=id2,
                        target_module=id1,
                        merged_charter=f"{m1.charter}; {m2.charter}".strip("; "),
                    )
                    candidates.append(
                        MutationCandidate(
                            action=action,
                            priority_score=cosine_sim,
                            symptom="redundant",
                            evidence={
                                "cosine_similarity": round(cosine_sim, 4),
                                "combined_size": combined_size,
                            },
                        )
                    )

        return candidates

    def find_eligible_shortcuts(
        self,
        backend: MorpheBackend,
        coaccess_counts: Dict[Tuple[str, str], int],
    ) -> List[MutationCandidate]:
        """
        Identify candidate shortcut edges:
        Modules frequently co-accessed without an existing direct edge.
        """
        candidates: List[MutationCandidate] = []
        snapshot = backend.get_staged_snapshot()
        existing_edges = {(e.source, e.target) for e in snapshot.edges}

        for (m1, m2), count in coaccess_counts.items():
            if m1 == m2:
                continue
            if m1 not in snapshot.modules or m2 not in snapshot.modules:
                continue

            if count >= self.shortcut_coaccess_threshold:
                # If no direct edge exists from m1 -> m2
                if (m1, m2) not in existing_edges:
                    action = MutationAction.add_edge(
                        source=m1,
                        target=m2,
                        weight=min(1.0, 0.5 + 0.05 * count),
                        edge_type=EdgeType.SHORTCUT,
                    )
                    candidates.append(
                        MutationCandidate(
                            action=action,
                            priority_score=float(count),
                            symptom="missing_shortcut",
                            evidence={"coaccess_count": count},
                        )
                    )

        return candidates

    def find_eligible_prunes(
        self,
        backend: MorpheBackend,
        cold_edge_cycles: Optional[Dict[Tuple[str, str], int]] = None,
        min_cold_cycles: int = 3,
    ) -> List[MutationCandidate]:
        """
        Identify candidate edges for pruning:
        Edge weight below epsilon or cold across consecutive cycles.
        """
        candidates: List[MutationCandidate] = []
        snapshot = backend.get_staged_snapshot()
        cycles = cold_edge_cycles or {}

        for edge in snapshot.edges:
            pair = (edge.source, edge.target)
            consecutive_cold = cycles.get(pair, 0)

            if edge.weight < self.prune_epsilon or consecutive_cold >= min_cold_cycles:
                action = MutationAction.remove_edge(edge.source, edge.target)
                score = 1.0 - edge.weight
                candidates.append(
                    MutationCandidate(
                        action=action,
                        priority_score=score,
                        symptom="cold_edge",
                        evidence={
                            "weight": edge.weight,
                            "consecutive_cold_cycles": consecutive_cold,
                        },
                    )
                )

        return candidates

    def find_eligible_promotions(
        self,
        backend: MorpheBackend,
        coaccess_counts: Dict[Tuple[str, str], int],
    ) -> List[MutationCandidate]:
        """
        Identify candidate promotions:
        A cluster of >= min_cluster_size modules with persistent high co-access
        that should receive a parent summary module.
        """
        candidates: List[MutationCandidate] = []
        snapshot = backend.get_staged_snapshot()

        # Build coaccess adjacency graph
        high_coaccess_pairs: List[Tuple[str, str]] = [
            pair for pair, c in coaccess_counts.items()
            if c >= self.shortcut_coaccess_threshold and pair[0] in snapshot.modules and pair[1] in snapshot.modules
        ]

        if not high_coaccess_pairs:
            return []

        # Find connected components of co-accessed modules
        adj: Dict[str, Set[str]] = collections.defaultdict(set)
        for u, v in high_coaccess_pairs:
            adj[u].add(v)
            adj[v].add(u)

        visited: Set[str] = set()
        for node in adj:
            if node not in visited:
                comp = set()
                queue = [node]
                while queue:
                    curr = queue.pop(0)
                    if curr not in comp:
                        comp.add(curr)
                        visited.add(curr)
                        for neighbor in adj[curr]:
                            if neighbor not in comp:
                                queue.append(neighbor)

                if len(comp) >= self.promote_min_cluster_size:
                    # Check if already promoted under the same parent
                    existing_parents = {
                        snapshot.modules[m].parent_module_id
                        for m in comp if snapshot.modules[m].parent_module_id is not None
                    }
                    if not existing_parents:
                        parent_id = f"parent_cluster_{len(candidates) + 1}"
                        # Propose adding parent module
                        action = MutationAction(
                            mutation_type=MutationType.ADD_MODULE,
                            new_module_id=parent_id,
                            charter=f"Parent hierarchical summary for modules: {sorted(comp)}",
                            metadata={"is_hierarchical_parent": True, "children": list(comp)},
                        )
                        candidates.append(
                            MutationCandidate(
                                action=action,
                                priority_score=float(len(comp)),
                                symptom="cluster_convergence",
                                evidence={"cluster_size": len(comp), "members": list(comp)},
                            )
                        )

        return candidates

    # ------------------------------------------------------------------
    # 2. Comprehensive Candidate Scan
    # ------------------------------------------------------------------

    def get_all_eligible_candidates(
        self,
        backend: MorpheBackend,
        coaccess_counts: Optional[Dict[Tuple[str, str], int]] = None,
        cold_edge_cycles: Optional[Dict[Tuple[str, str], int]] = None,
    ) -> List[MutationCandidate]:
        """
        Scan all 5 mutation operations plus DO_NOTHING.
        Returns all eligible candidates ranked by priority score.
        """
        coaccess = coaccess_counts or {}
        cold_cycles = cold_edge_cycles or {}

        candidates: List[MutationCandidate] = []
        candidates.extend(self.find_eligible_splits(backend))
        candidates.extend(self.find_eligible_merges(backend))
        candidates.extend(self.find_eligible_shortcuts(backend, coaccess))
        candidates.extend(self.find_eligible_prunes(backend, cold_cycles))
        candidates.extend(self.find_eligible_promotions(backend, coaccess))

        # Always include 'Do Nothing' fallback
        candidates.append(
            MutationCandidate(
                action=MutationAction(mutation_type=MutationType.UPDATE_EDGE_WEIGHT, metadata={"do_nothing": True}),
                priority_score=0.0,
                symptom="optimal_steady_state",
                evidence={"info": "Graph requires no structural change"},
            )
        )

        # Sort descending by priority score
        candidates.sort(key=lambda c: c.priority_score, reverse=True)
        return candidates

