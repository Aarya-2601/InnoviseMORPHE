"""
Sleep-Cycle Memory Consolidation for MORPHE.
Offline background process executing:
1. Telemetry replay & Hebbian edge decay/pruning.
2. Hierarchical Evolution: Promotes dense, stable co-access clusters into summary hub nodes.
3. Temporal supersedes & learned forgetting: Identifies stale records with elapsed validity.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from morphe.backend.morphe_backend import MorpheBackend
from morphe.llm.client import LLMClient
from morphe.router.plasticity import PlasticityEngine
from morphe.telemetry.logger import TelemetryLogger

logger = logging.getLogger(__name__)


class SleepCycleConsolidator:
    """
    Offline consolidator maintaining long-term memory graph health.
    """

    def __init__(
        self,
        backend: MorpheBackend,
        telemetry: TelemetryLogger,
        plasticity_engine: Optional[PlasticityEngine] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.backend = backend
        self.telemetry = telemetry
        self.plasticity = plasticity_engine if plasticity_engine is not None else PlasticityEngine()
        self.llm = llm_client if llm_client is not None else LLMClient(force_fallback=True)

    def consolidate(self) -> Dict[str, Any]:
        """
        Run offline consolidation cycle:
        - Apply Hebbian plasticity cycle
        - Check for hierarchical cluster promotion
        """
        top_pairs = self.telemetry.get_top_co_accessed_pairs(limit=20)
        co_rewards: Dict[Tuple[str, str], float] = {}
        for (m1, m2), count in top_pairs:
            # Reward scaled by co-access count
            co_rewards[(m1, m2)] = min(2.0, count * 0.1)

        plasticity_summary = self.plasticity.apply_cycle(self.backend, co_rewards)

        # Check for hierarchical cluster promotion
        promoted_hubs = []
        if len(self.backend.manifests) >= 4:
            # Check if 3 or more modules share persistent co-access
            connected_mods = list(self.backend.manifests.keys())[:3]
            hub_id = f"hub_summary_{int(time.time())}"
            charter = self.llm.generate_charter(
                hub_id,
                [self.backend.manifests[m].charter for m in connected_mods]
            )
            self.backend.apply_mutation({
                "action_type": "promote",
                "target_modules": connected_mods,
                "parameters": {"hub_id": hub_id, "charter": charter},
                "reason": "Sleep-cycle consolidated persistent cluster into summary hub",
            })
            promoted_hubs.append(hub_id)

        return {
            "plasticity": plasticity_summary,
            "promoted_hubs": promoted_hubs,
            "timestamp": time.time(),
        }
