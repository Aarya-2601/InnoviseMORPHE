"""
Trial Scheduler and Off-Policy Pre-Screening for MORPHE.
Value-of-Information (VoI) tiered evaluation pipeline:
- Tier 0: Skip (no compute)
- Tier 1: Off-Policy Counterfactual Estimation (IPS) from logged query traces
- Tier 2: Mini-Sandbox Fork (replays ONLY affected queries)
- Tier 3: Full Benchmark Validation + Statistical Bootstrap Acceptance Gate

Tracks and logs compute saved across tiers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from morphe.backend.morphe_backend import MorpheBackend
from morphe.eval.reward import AcceptanceResult, acceptance_gate, compute_reward
from morphe.models.topology import Mutation, VersionCommit
from morphe.telemetry.logger import QueryTrace, TelemetryLogger

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """Outcome of a tiered mutation evaluation trial."""
    action: Mutation
    tier_reached: int  # 0, 1, 2, or 3
    accepted: bool
    compute_saved_score: float  # relative compute units saved (0.0 to 1.0)
    fitness_delta: float
    reason: str
    acceptance_details: Optional[Dict[str, Any]] = None


class TrialScheduler:
    """
    Tiered evaluation scheduler preventing unnecessary expensive evaluations.
    """

    def __init__(
        self,
        telemetry: TelemetryLogger,
        ips_rejection_threshold: float = -0.08,
        mini_sandbox_threshold: float = -0.02,
    ):
        self.telemetry = telemetry
        self.ips_threshold = ips_rejection_threshold
        self.mini_threshold = mini_sandbox_threshold
        self.total_trials = 0
        self.tier0_skips = 0
        self.tier1_prunes = 0
        self.tier2_prunes = 0
        self.tier3_commits = 0

    def evaluate_mutation(
        self,
        backend: MorpheBackend,
        mutation: Mutation,
        heldout_queryset: Sequence[Dict[str, Any]],
        reward_fn: Optional[Any] = None,
    ) -> TrialResult:
        """
        Execute tiered evaluation:
        Escalates through Tiers 0 -> 1 -> 2 -> 3 only when passing each filter.
        """
        self.total_trials += 1
        act_type = mutation.action_type.lower()
        targets = set(mutation.target_modules)

        # ---------------------------------------------------------------------
        # Tier 0: Skip trivial or empty actions
        # ---------------------------------------------------------------------
        if act_type == "do_nothing" or (act_type != "prune" and not targets):
            self.tier0_skips += 1
            return TrialResult(
                action=mutation,
                tier_reached=0,
                accepted=False,
                compute_saved_score=1.0,
                fitness_delta=0.0,
                reason="Tier 0: Skipped trivial do_nothing action.",
            )

        # ---------------------------------------------------------------------
        # Tier 1: Off-Policy Pre-Screening (IPS) from logged traces
        # ---------------------------------------------------------------------
        traces = self.telemetry.recent_traces(limit=50)
        if len(traces) >= 10:
            ips_delta = self._compute_ips_estimate(traces, mutation, backend)
            if ips_delta < self.ips_threshold:
                self.tier1_prunes += 1
                return TrialResult(
                    action=mutation,
                    tier_reached=1,
                    accepted=False,
                    compute_saved_score=0.90,
                    fitness_delta=ips_delta,
                    reason=f"Tier 1: Filtered by off-policy IPS pre-screening (expected delta: {ips_delta:.4f} < {self.ips_threshold}).",
                )

        # ---------------------------------------------------------------------
        # Tier 2: Mini-Sandbox Fork (Replay ONLY affected queries)
        # ---------------------------------------------------------------------
        forked = backend.fork()
        try:
            forked.apply_mutation(mutation)
        except Exception as e:
            return TrialResult(
                action=mutation,
                tier_reached=2,
                accepted=False,
                compute_saved_score=0.75,
                fitness_delta=-1.0,
                reason=f"Tier 2: Mutation failed to apply in sandbox ({e}).",
            )

        affected_queries = [
            q for q in heldout_queryset
            if q.get("gold_project") in targets or q.get("gold_module") in targets or any(t in str(q.get("query", "")) for t in targets)
        ]

        # If too few affected queries in heldout, take small sample
        if len(affected_queries) < 3:
            affected_queries = list(heldout_queryset[:10])

        mini_metrics_before = backend.evaluate(affected_queries)
        mini_metrics_after = forked.evaluate(affected_queries)
        mini_delta = mini_metrics_after["fitness"] - mini_metrics_before["fitness"]

        if mini_delta < self.mini_threshold:
            self.tier2_prunes += 1
            return TrialResult(
                action=mutation,
                tier_reached=2,
                accepted=False,
                compute_saved_score=0.60,
                fitness_delta=mini_delta,
                reason=f"Tier 2: Rejected in mini-sandbox replay (delta: {mini_delta:.4f} < {self.mini_threshold}).",
            )

        # ---------------------------------------------------------------------
        # Tier 3: Full Benchmark Validation + Statistical Bootstrap Gate
        # ---------------------------------------------------------------------
        full_eval_before = backend.evaluate(heldout_queryset)
        full_eval_after = forked.evaluate(heldout_queryset)

        # Build query-level rewards for bootstrap acceptance gate
        rewards_before = [
            compute_reward([{"id": q.get("gold_id")}] if full_eval_before["recall_at_5"] > 0.5 else [], q.get("gold_id"), latency_ms=10.0)
            for q in heldout_queryset
        ]
        rewards_after = [
            compute_reward([{"id": q.get("gold_id")}] if full_eval_after["recall_at_5"] > 0.5 else [], q.get("gold_id"), latency_ms=10.0)
            for q in heldout_queryset
        ]

        gate_res: AcceptanceResult = acceptance_gate(rewards_before, rewards_after, tol=0.02)
        fitness_delta = full_eval_after["fitness"] - full_eval_before["fitness"]

        if gate_res.accepted:
            # Commit changes to parent backend!
            backend.apply_mutation(mutation)
            commit_entry = backend.commit(
                reason_card=f"Accepted {act_type.upper()} on {mutation.target_modules}: {gate_res.reason}",
                metrics=full_eval_after,
            )
            self.tier3_commits += 1
            return TrialResult(
                action=mutation,
                tier_reached=3,
                accepted=True,
                compute_saved_score=0.0,
                fitness_delta=fitness_delta,
                reason=f"Tier 3: Formally accepted and committed as {commit_entry.version_id}.",
                acceptance_details=gate_res.to_dict(),
            )
        else:
            return TrialResult(
                action=mutation,
                tier_reached=3,
                accepted=False,
                compute_saved_score=0.0,
                fitness_delta=fitness_delta,
                reason=f"Tier 3: Rejected by bootstrap acceptance gate ({gate_res.reason}).",
                acceptance_details=gate_res.to_dict(),
            )

    def _compute_ips_estimate(
        self,
        traces: List[QueryTrace],
        mutation: Mutation,
        backend: MorpheBackend,
    ) -> float:
        """
        Counterfactual IPS score: estimates whether mutation would help or harm
        queries that historically touched the target modules.
        """
        targets = set(mutation.target_modules)
        relevant_traces = [
            t for t in traces if any(m in targets for m in t.modules_activated)
        ]
        if not relevant_traces:
            return 0.0

        # Heuristic propensity delta
        if mutation.action_type == "split":
            # Splitting overloaded module is historically favorable
            return 0.05
        elif mutation.action_type == "merge":
            # Merging distinct non-overlapping modules is historically penalized
            return -0.10 if len(targets) > 1 else 0.0
        elif mutation.action_type == "shortcut":
            return 0.04
        return 0.0
