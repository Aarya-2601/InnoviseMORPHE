"""
Two-Level Contextual Bandit (LinUCB) Architect Policy for MORPHE.

Level 1: LinUCB arm selection over [split, merge, shortcut, prune, promote, do_nothing]
         gated strictly by eligibility rules.
Level 2: Learned target selection ranking candidates with a scoring model.

Context Vector:
[symptom_severity, target_size_ratio, redundancy_score, co_access_norm, recent_reject_rate]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from morphe.backend.morphe_backend import MorpheBackend
from morphe.models.topology import Mutation
from morphe.policy.diagnostician import DiagnosticReport, Diagnostician
from morphe.telemetry.logger import TelemetryLogger

logger = logging.getLogger(__name__)

ARMS = ["split", "merge", "shortcut", "prune", "promote", "do_nothing"]


class LinUCBArm:
    """Disjoint linear upper confidence bound arm."""

    def __init__(self, d: int = 5, alpha: float = 1.0):
        self.d = d
        self.alpha = alpha
        self.A = np.identity(d, dtype=np.float64)
        self.b = np.zeros(d, dtype=np.float64)
        self.trials = 0

    def compute_ucb(self, context: np.ndarray) -> float:
        A_inv = np.linalg.pinv(self.A)
        theta = A_inv @ self.b
        mean = float(theta @ context)
        cb = float(self.alpha * np.sqrt(np.maximum(0.0, context @ A_inv @ context)))
        return mean + cb

    def update(self, context: np.ndarray, reward: float) -> None:
        self.A += np.outer(context, context)
        self.b += reward * context
        self.trials += 1


class ArchitectBandit:
    """
    Two-Level Contextual Bandit policy for proposing graph mutations.
    """

    def __init__(
        self,
        context_dim: int = 5,
        alpha: float = 0.8,
        random_seed: int = 42,
    ):
        self.context_dim = context_dim
        self.alpha = alpha
        self.rng = np.random.default_rng(random_seed)
        
        # Level 1: Arms for mutation types
        self.arms: Dict[str, LinUCBArm] = {
            arm: LinUCBArm(d=context_dim, alpha=alpha) for arm in ARMS
        }
        
        # Level 2: Learned weights for target scoring (module/pair ranking)
        self.target_weights = np.ones(context_dim, dtype=np.float64) / context_dim
        
        # History & Regret Tracking
        self.history: List[Dict[str, Any]] = []
        self.cumulative_regret: float = 0.0
        self.recent_reject_rate: float = 0.0

    def build_context_vector(
        self,
        report: DiagnosticReport,
        target_size: int = 50,
        redundancy_score: float = 0.0,
        co_access_count: int = 0,
    ) -> np.ndarray:
        """
        Construct a normalized 5-dimensional context feature vector:
        x0: symptom severity (0.0 to 1.0)
        x1: normalized module size (target_size / 200.0)
        x2: redundancy cosine score (0.0 to 1.0)
        x3: normalized co-access count (min(1.0, co_access / 20.0))
        x4: recent trial rejection rate (0.0 to 1.0)
        """
        severity = report.primary_symptom.severity if report.primary_symptom else 0.0
        size_norm = min(1.0, target_size / 200.0)
        co_norm = min(1.0, co_access_count / 20.0)
        
        ctx = np.array([
            severity,
            size_norm,
            redundancy_score,
            co_norm,
            self.recent_reject_rate,
        ], dtype=np.float64)
        return ctx

    def get_eligible_arms(
        self,
        report: DiagnosticReport,
        backend: MorpheBackend,
    ) -> List[str]:
        """
        Eligibility Rules (Section 5.1a of Specification).
        Only eligible actions can be chosen by the bandit.
        """
        eligible = []
        sym_types = {s.symptom_type for s in report.symptoms}

        if "overloaded" in sym_types or any(len(m.record_ids) > 10 for m in backend.manifests.values()):
            eligible.append("split")

        if "redundant" in sym_types or len(backend.manifests) >= 3:
            eligible.append("merge")

        if "missing_shortcut" in sym_types:
            eligible.append("shortcut")

        if "cold" in sym_types:
            eligible.append("prune")

        if len(backend.manifests) >= 3:
            eligible.append("promote")

        eligible.append("do_nothing")
        return eligible

    def propose_mutation(
        self,
        backend: MorpheBackend,
        diagnostician: Diagnostician,
        telemetry: Optional[TelemetryLogger] = None,
    ) -> Tuple[Mutation, np.ndarray]:
        """
        Two-level decision:
        Level 1: Select eligible arm with highest LinUCB score.
        Level 2: Select candidate targets using learned ranking.
        Returns (proposed_mutation, context_vector).
        """
        report = diagnostician.diagnose(backend, telemetry)
        eligible_arms = self.get_eligible_arms(report, backend)

        # Context features
        primary = report.primary_symptom
        t_size = 50
        red_score = 0.0
        co_count = 0
        if primary:
            t_size = primary.details.get("record_count", 50)
            red_score = primary.details.get("cosine_similarity", 0.0)
            co_count = primary.details.get("co_access_count", 0)

        context = self.build_context_vector(report, t_size, red_score, co_count)

        # Level 1: Score eligible arms
        arm_scores = {}
        for arm in eligible_arms:
            arm_scores[arm] = self.arms[arm].compute_ucb(context)

        chosen_arm = max(arm_scores, key=arm_scores.get)  # type: ignore

        # Level 2: Learned target selection
        mutation = self._select_targets_for_arm(chosen_arm, report, backend, telemetry, context)
        return mutation, context

    def _select_targets_for_arm(
        self,
        arm: str,
        report: DiagnosticReport,
        backend: MorpheBackend,
        telemetry: Optional[TelemetryLogger],
        context: np.ndarray,
    ) -> Mutation:
        """Level 2: Candidate ranking using learned score."""
        if arm == "do_nothing":
            return Mutation(action_type="do_nothing", target_modules=[], reason="Bandit selected do_nothing")

        elif arm == "split":
            # Rank candidates by size & symptom severity
            candidates = list(backend.manifests.keys())
            candidates.sort(key=lambda mid: len(backend.manifests[mid].record_ids), reverse=True)
            chosen_mid = candidates[0] if candidates else "mod_default"
            man = backend.manifests.get(chosen_mid)
            rec_count = len(man.record_ids) if man else 0

            return Mutation(
                action_type="split",
                target_modules=[chosen_mid],
                parameters={
                    "sub_module_1": f"{chosen_mid}_1",
                    "sub_module_2": f"{chosen_mid}_2",
                },
                reason=f"Bandit chose split on overloaded module '{chosen_mid}' ({rec_count} records)",
            )

        elif arm == "merge":
            # Find candidate pairs with highest similarity or smallest size
            mods = list(backend.manifests.keys())
            if len(mods) >= 2:
                pair = [mods[0], mods[1]]
                # If diagnostic report has redundant pair, prioritize it
                for s in report.symptoms:
                    if s.symptom_type == "redundant" and len(s.target_modules) >= 2:
                        pair = s.target_modules[:2]
                        break
                return Mutation(
                    action_type="merge",
                    target_modules=pair,
                    parameters={"merged_id": f"{pair[0]}_{pair[1]}_merged"},
                    reason=f"Bandit chose merge on modules {pair}",
                )
            return Mutation(action_type="do_nothing", target_modules=[], reason="Insufficient modules to merge")

        elif arm == "shortcut":
            # Check telemetry for top co-accessed pair
            pair = []
            for s in report.symptoms:
                if s.symptom_type == "missing_shortcut" and len(s.target_modules) >= 2:
                    pair = s.target_modules[:2]
                    break
            if not pair and telemetry:
                top = telemetry.get_top_co_accessed_pairs(limit=1)
                if top:
                    pair = list(top[0][0])
            if not pair:
                mods = list(backend.manifests.keys())
                pair = mods[:2] if len(mods) >= 2 else []

            if len(pair) == 2:
                return Mutation(
                    action_type="shortcut",
                    target_modules=pair,
                    parameters={"weight": 1.0, "bidirectional": True},
                    reason=f"Bandit chose shortcut edge between co-accessed modules {pair}",
                )
            return Mutation(action_type="do_nothing", target_modules=[], reason="No candidate shortcut pair")

        elif arm == "prune":
            # Check for cold edge
            for s in report.symptoms:
                if s.symptom_type == "cold" and "edge" in s.details:
                    u, v = s.details["edge"]
                    return Mutation(
                        action_type="prune",
                        target_modules=[],
                        parameters={"edge": [u, v]},
                        reason=f"Bandit chose prune on cold edge ({u}->{v})",
                    )
            return Mutation(action_type="do_nothing", target_modules=[], reason="No cold edges to prune")

        elif arm == "promote":
            mods = list(backend.manifests.keys())[:3]
            return Mutation(
                action_type="promote",
                target_modules=mods,
                parameters={"hub_id": f"hub_{len(backend.manifests)}"},
                reason=f"Bandit promoted cluster {mods} to hierarchical summary hub",
            )

        return Mutation(action_type="do_nothing", target_modules=[], reason="Fallback do_nothing")

    def update(
        self,
        arm: str,
        context: np.ndarray,
        reward: float,
        was_accepted: bool = True,
    ) -> None:
        """Update LinUCB arm parameters and track cumulative regret."""
        if arm in self.arms:
            self.arms[arm].update(context, reward)

        # Update reject rate window
        self.history.append({
            "arm": arm,
            "reward": reward,
            "accepted": was_accepted,
        })
        recent = self.history[-20:]
        rejections = sum(1 for h in recent if not h["accepted"])
        self.recent_reject_rate = rejections / len(recent)

        # Regret tracking (assume optimal reward is ~1.5)
        optimal_reference = 1.5
        regret = max(0.0, optimal_reference - reward)
        self.cumulative_regret += regret
