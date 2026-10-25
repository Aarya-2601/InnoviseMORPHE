"""
Hebbian Three-Factor Plasticity Rule for MORPHE.
Update rule:
Delta w_ij = eta * co_access_ij * max(0, verified_reward)
w_ij = (1 - lambda) * w_ij + Delta w_ij
Edges falling below epsilon are pruned.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

from morphe.backend.morphe_backend import MorpheBackend

logger = logging.getLogger(__name__)


class PlasticityEngine:
    """
    Applies biologically-inspired three-factor Hebbian plasticity to memory graph edges.
    """

    def __init__(
        self,
        eta: float = 0.15,      # Learning rate for co-activation reinforcement
        decay_lambda: float = 0.05,  # Passive decay factor per cycle
        epsilon: float = 0.02,  # Pruning threshold for cold edges
    ):
        self.eta = eta
        self.decay_lambda = decay_lambda
        self.epsilon = epsilon

    def apply_cycle(
        self,
        backend: MorpheBackend,
        co_access_rewards: Dict[Tuple[str, str], float],
    ) -> Dict[str, Any]:
        """
        Execute one plasticity cycle over all graph edges.
        co_access_rewards: map of (source_module, target_module) -> verified reward
        """
        reinforced_edges = 0
        decayed_edges = 0
        pruned_edges = []

        edges_to_process = list(backend.graph.edges(data=True))

        for u, v, data in edges_to_process:
            w = float(data.get("weight", 1.0))
            pair = (u, v) if (u, v) in co_access_rewards else (v, u)

            # Hebbian three-factor reinforcement
            reward = co_access_rewards.get(pair, 0.0)
            if reward > 0.0:
                delta_w = self.eta * reward
                w += delta_w
                reinforced_edges += 1
            else:
                # Passive decay
                w *= (1.0 - self.decay_lambda)
                decayed_edges += 1

            if w < self.epsilon:
                pruned_edges.append((u, v))
            else:
                backend.graph[u][v]["weight"] = round(w, 4)

        # Remove pruned edges
        for u, v in pruned_edges:
            backend.graph.remove_edge(u, v)

        return {
            "reinforced_count": reinforced_edges,
            "decayed_count": decayed_edges,
            "pruned_count": len(pruned_edges),
            "pruned_edges": pruned_edges,
        }
