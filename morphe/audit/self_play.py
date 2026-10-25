"""
Self-Play Auditor for MORPHE.
Generates self-play probes from stored records during idle cycles.
Identifies weak retrieval pathways pre-emptively without requiring human supervision.
Failures become direct training reward signals for the Contextual Bandit.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from morphe.backend.morphe_backend import MorpheBackend
from morphe.llm.client import LLMClient
from morphe.router.learned_router import LearnedRouter
from morphe.storage.record_store import Record

logger = logging.getLogger(__name__)


@dataclass
class AuditPainPoint:
    """A detected retrieval failure during self-play."""
    probe_query: str
    target_record_id: str
    expected_module: str
    retrieved_module_ids: List[str]
    retrieved_record_ids: List[str]
    probe_type: str = "factual"  # factual, adversarial, paraphrase
    timestamp: float = field(default_factory=lambda: 0.0)


class SelfPlayAuditor:
    """
    Autonomous adversarial probe generator for continuous self-assessment.
    """

    def __init__(
        self,
        backend: MorpheBackend,
        router: LearnedRouter,
        llm_client: Optional[LLMClient] = None,
        random_seed: int = 42,
    ):
        self.backend = backend
        self.router = router
        self.llm = llm_client if llm_client is not None else LLMClient(force_fallback=True)
        self.rng = random.Random(random_seed)
        self.audit_history: List[Dict[str, Any]] = []
        self.pain_points: List[AuditPainPoint] = []

    def run_audit_cycle(
        self,
        num_probes: int = 5,
        adversarial_ratio: float = 0.4,
    ) -> Dict[str, Any]:
        """
        Execute an audit cycle over stored memory records.
        Returns audit statistics including pass rate and discovered pain points.
        """
        manifests = self.backend.manifests
        all_record_ids: List[Tuple[str, str]] = []  # (module_id, record_id)
        for mid, man in manifests.items():
            for rid in man.record_ids:
                all_record_ids.append((mid, rid))

        if not all_record_ids:
            return {"probes_tested": 0, "pass_rate": 1.0, "pain_points_found": 0}

        sample_size = min(num_probes, len(all_record_ids))
        sampled = self.rng.sample(all_record_ids, sample_size)

        passed = 0
        cycle_pain_points: List[AuditPainPoint] = []

        for mid, rid in sampled:
            rec = self.backend.record_store.get(rid)
            if not rec:
                continue

            # Generate probe
            probes = self.llm.generate_probes(rec.text, n=1)
            probe_query = probes[0] if probes else f"Lookup specifications for {rec.text[:30]}"

            # Test retrieval via router
            results, trace = self.router.route_and_retrieve(probe_query, gold_id=rid, top_k=5)
            hit = any(res["id"] == rid for res in results)

            if hit:
                passed += 1
            else:
                pp = AuditPainPoint(
                    probe_query=probe_query,
                    target_record_id=rid,
                    expected_module=mid,
                    retrieved_module_ids=trace.modules_activated,
                    retrieved_record_ids=[r["id"] for r in results],
                    probe_type="adversarial" if self.rng.random() < adversarial_ratio else "factual",
                )
                self.pain_points.append(pp)
                cycle_pain_points.append(pp)

        pass_rate = passed / max(1, sample_size)
        stats = {
            "probes_tested": sample_size,
            "passed": passed,
            "pass_rate": round(pass_rate, 4),
            "pain_points_found": len(cycle_pain_points),
            "cycle_pain_points": [pp.__dict__ for pp in cycle_pain_points],
        }
        self.audit_history.append(stats)
        return stats
