"""
Sleep Cycle Consolidation for MORPHE.
Specification from Section 10 and 10A.5 of the MORPHE Final Plan:
- Scheduled offline cycle (replay telemetry & structural maintenance)
- Near-duplicate record merging:
  * Detects record pairs in a module with cosine similarity >= near_duplicate_threshold (e.g. 0.95)
  * Retains primary canonical record, links near-duplicate via SUPERSEDES edge and soft-archives it
- Stale-memory down-weighting:
  * Identifies records unaccessed / cold over staleness_window_seconds
  * Decays their retrieval weight or soft-archives them (learned safe forgetting)
- Proposes an atomic consolidation mutation that can be evaluated in sandbox and committed/rolled back
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from morphe.core.backend import MorpheBackend
from morphe.storage.record_store import Record, RecordStore
from morphe.temporal.manager import TemporalMemoryManager
from morphe.topology.models import ModuleManifest
from morphe.topology.mutations import MutationAction, MutationType


@dataclass
class NearDuplicatePair:
    module_id: str
    primary_record_id: str
    duplicate_record_id: str
    similarity: float


@dataclass
class StaleRecord:
    module_id: str
    record_id: str
    last_accessed_at: float
    age_seconds: float


@dataclass
class SleepCycleReport:
    """Consolidated summary of changes proposed during the sleep cycle."""
    merged_near_duplicates: List[NearDuplicatePair] = field(default_factory=list)
    downweighted_stale_records: List[StaleRecord] = field(default_factory=list)
    records_archived_count: int = 0
    modules_compacted_count: int = 0


class SleepCycleConsolidator:
    """
    Offline sleep-cycle consolidator:
    - Merges near-duplicate records inside modules
    - Down-weights or archives stale, unaccessed memories
    """

    def __init__(
        self,
        backend: MorpheBackend,
        near_duplicate_cosine_threshold: float = 0.95,
        staleness_threshold_seconds: float = 7 * 86400,  # 7 days default
    ):
        self.backend = backend
        self.record_store = backend.record_store
        self.temporal_mgr = TemporalMemoryManager(backend)
        self.near_duplicate_cosine_threshold = near_duplicate_cosine_threshold
        self.staleness_threshold_seconds = staleness_threshold_seconds

    def find_near_duplicates_in_module(
        self,
        module_id: str,
    ) -> List[NearDuplicatePair]:
        """
        Scan a module for near-duplicate records using embedding cosine similarity.
        """
        manifest = self.backend.get_staged_snapshot().get_module(module_id)
        if not manifest or manifest.record_count < 2:
            return []

        records = self.record_store.get_many(manifest.record_ids)
        active_items = [(rid, rec) for rid, rec in records.items() if not rec.archived and rec.embedding is not None]
        if len(active_items) < 2:
            return []

        # Stack embeddings (already L2-normalized)
        r_ids = [item[0] for item in active_items]
        matrix = np.stack([item[1].embedding for item in active_items], axis=0)

        # Compute pairwise cosine similarity
        sim_matrix = np.dot(matrix, matrix.T)

        duplicates: List[NearDuplicatePair] = []
        visited: Set[int] = set()

        for i in range(len(r_ids)):
            if i in visited:
                continue
            for j in range(i + 1, len(r_ids)):
                if j in visited:
                    continue
                sim = float(sim_matrix[i, j])
                if sim >= self.near_duplicate_cosine_threshold:
                    # Determine primary (older or longer text)
                    rec_i = active_items[i][1]
                    rec_j = active_items[j][1]
                    if rec_i.created_at <= rec_j.created_at:
                        primary_id, dup_id = r_ids[i], r_ids[j]
                    else:
                        primary_id, dup_id = r_ids[j], r_ids[i]

                    duplicates.append(
                        NearDuplicatePair(
                            module_id=module_id,
                            primary_record_id=primary_id,
                            duplicate_record_id=dup_id,
                            similarity=round(sim, 4),
                        )
                    )
                    visited.add(j)

        return duplicates

    def find_stale_records(
        self,
        last_accessed_map: Optional[Dict[str, float]] = None,
        now: Optional[float] = None,
    ) -> List[StaleRecord]:
        """
        Identify records across modules that haven't been accessed within staleness threshold.
        """
        current_time = time.time() if now is None else now
        access_map = last_accessed_map or {}
        snapshot = self.backend.get_staged_snapshot()
        stale: List[StaleRecord] = []

        for mod_id, manifest in snapshot.modules.items():
            records = self.record_store.get_many(manifest.record_ids)
            for rid, rec in records.items():
                if rec.archived:
                    continue
                last_time = access_map.get(rid, rec.created_at)
                age = current_time - last_time
                if age >= self.staleness_threshold_seconds:
                    stale.append(
                        StaleRecord(
                            module_id=mod_id,
                            record_id=rid,
                            last_accessed_at=last_time,
                            age_seconds=round(age, 2),
                        )
                    )

        return stale

    def run_sleep_cycle(
        self,
        last_accessed_map: Optional[Dict[str, float]] = None,
        now: Optional[float] = None,
        auto_commit: bool = False,
    ) -> SleepCycleReport:
        """
        Executes full sleep-cycle consolidation:
        1. Finds and merges near-duplicate records via supersedes edges & soft archive.
        2. Down-weights and soft-archives stale records.
        3. Updates module manifests without the duplicate/stale records.
        """
        report = SleepCycleReport()
        current_time = time.time() if now is None else now
        snapshot = self.backend.get_staged_snapshot()

        # Step 1: Near-duplicate merging per module
        records_to_archive: Set[str] = set()
        module_record_removals: Dict[str, Set[str]] = {}

        for mod_id in snapshot.modules:
            dup_pairs = self.find_near_duplicates_in_module(mod_id)
            for pair in dup_pairs:
                report.merged_near_duplicates.append(pair)
                # Link primary to duplicate via supersedes edge
                self.temporal_mgr.record_supersession(
                    new_record_id=pair.primary_record_id,
                    old_record_id=pair.duplicate_record_id,
                    source_module=mod_id,
                    effective_time=current_time,
                    archive_old=True,
                )
                records_to_archive.add(pair.duplicate_record_id)
                module_record_removals.setdefault(mod_id, set()).add(pair.duplicate_record_id)

        # Step 2: Stale-memory down-weighting
        stale_records = self.find_stale_records(last_accessed_map=last_accessed_map, now=current_time)
        for st in stale_records:
            report.downweighted_stale_records.append(st)
            self.record_store.set_archived(st.record_id, archived=True)
            records_to_archive.add(st.record_id)
            module_record_removals.setdefault(st.module_id, set()).add(st.record_id)

        # Step 3: Compact module manifests in staging
        for mod_id, removals in module_record_removals.items():
            manifest = self.backend.get_staged_snapshot().get_module(mod_id)
            if manifest:
                updated_recs = [r for r in manifest.record_ids if r not in removals]
                self.backend._staged_modules[mod_id] = manifest.with_records(updated_recs)
                report.modules_compacted_count += 1

        self.backend._sync_nx_graph()
        report.records_archived_count = len(records_to_archive)

        if auto_commit and report.records_archived_count > 0:
            self.backend.commit(
                reason_card=f"Sleep cycle: merged {len(report.merged_near_duplicates)} near-duplicates, downweighted {len(report.downweighted_stale_records)} stale records",
                metrics={"structural_cost_savings": report.records_archived_count * 0.05},
            )

        return report

