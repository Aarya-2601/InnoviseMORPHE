"""
Reward Function, Pessimistic Acceptance Gate, and Post-Commit Watchdog for MORPHE.

Core Philosophy:
- Ground truth based reward (Recall, MRR, latency, modules touched, structural cost).
- Pessimistic Acceptance Gate: Bootstraps the lower confidence bound of improvement.
  Commits only if the 95% lower bound exceeds -tolerance (guard against regression).
- Post-commit Watchdog: Monitors live traffic reward, triggers automatic rollback if quality deteriorates.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass
class RewardWeights:
    """
    Tunable weights for the multi-objective reward function:
    R = a * Recall@k + b * MRR - c * latency_ms - d * modules_touched - e * structural_cost
    """
    a: float = 1.0     # Recall@5 weight
    b: float = 1.0     # MRR weight
    c: float = 0.005   # Latency penalty per millisecond
    d: float = 0.05    # Module traversal penalty per module touched
    e: float = 0.02    # Structural complexity penalty (nodes + edges)

    def to_dict(self) -> Dict[str, float]:
        return {"a": self.a, "b": self.b, "c": self.c, "d": self.d, "e": self.e}


@dataclass
class AcceptanceResult:
    """Outcome of the statistical bootstrap acceptance gate."""
    accepted: bool
    mean_improvement: float
    lower_bound: float
    upper_bound: float
    tolerance: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "mean_improvement": round(self.mean_improvement, 4),
            "lower_bound": round(self.lower_bound, 4),
            "upper_bound": round(self.upper_bound, 4),
            "tolerance": self.tolerance,
            "reason": self.reason,
        }


def compute_reward(
    results: Sequence[Dict[str, Any]],
    gold_id: Optional[str],
    latency_ms: float,
    modules_touched: int = 1,
    structural_cost: float = 1.0,
    weights: Optional[RewardWeights] = None,
    k: int = 5,
) -> float:
    """
    Computes verifiable reward from ground truth only.
    results: list of retrieved items, each having 'id' (or 'record_id')
    gold_id: known correct target record ID
    """
    if weights is None:
        weights = RewardWeights()

    # Calculate Recall@k and MRR
    hit_rank = 0
    if gold_id:
        for idx, item in enumerate(results[:k], start=1):
            item_id = item.get("id") or item.get("record_id")
            if item_id == gold_id:
                hit_rank = idx
                break

    recall_at_k = 1.0 if hit_rank > 0 else 0.0
    mrr = (1.0 / hit_rank) if hit_rank > 0 else 0.0

    reward = (
        weights.a * recall_at_k
        + weights.b * mrr
        - weights.c * max(0.0, latency_ms)
        - weights.d * max(1, modules_touched)
        - weights.e * max(0.0, structural_cost)
    )
    return float(reward)


def acceptance_gate(
    rewards_before: Sequence[float],
    rewards_after: Sequence[float],
    tol: float = 0.02,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_seed: int = 42,
) -> AcceptanceResult:
    """
    Pessimistic statistical acceptance gate.
    Bootstraps the distribution of mean improvement (after - before).
    Accepts ONLY if the (1 - alpha) lower confidence bound is >= -tol.
    """
    arr_before = np.asarray(rewards_before, dtype=np.float64)
    arr_after = np.asarray(rewards_after, dtype=np.float64)

    if len(arr_before) != len(arr_after) or len(arr_before) == 0:
        return AcceptanceResult(
            accepted=False,
            mean_improvement=0.0,
            lower_bound=0.0,
            upper_bound=0.0,
            tolerance=tol,
            reason="Empty or mismatched reward vectors passed to acceptance gate.",
        )

    delta = arr_after - arr_before
    mean_delta = float(np.mean(delta))

    # Bootstrap resampling
    rng = np.random.default_rng(random_seed)
    n = len(delta)
    boot_indices = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.mean(delta[boot_indices], axis=1)

    lower_bound = float(np.percentile(boot_means, alpha * 100))
    upper_bound = float(np.percentile(boot_means, (1.0 - alpha) * 100))

    # Guardrail condition: lower bound must not regress beyond tolerance
    if lower_bound >= -tol:
        accepted = True
        reason = (
            f"Accepted: Lower confidence bound {lower_bound:.4f} exceeds threshold -{tol:.4f} "
            f"(mean improvement: {mean_delta:+.4f})."
        )
    else:
        accepted = False
        reason = (
            f"Rejected: Lower confidence bound {lower_bound:.4f} violated tolerance -{tol:.4f} "
            f"(risk of regression under held-out evaluation)."
        )

    return AcceptanceResult(
        accepted=accepted,
        mean_improvement=mean_delta,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        tolerance=tol,
        reason=reason,
    )


class PostCommitWatchdog:
    """
    Monitors live traffic after commit.
    Triggers auto-rollback if rolling reward decays below safety baseline.
    """

    def __init__(
        self,
        baseline_reward: float,
        window_size: int = 20,
        max_allowed_drop: float = 0.10,
    ):
        self.baseline_reward = baseline_reward
        self.window_size = window_size
        self.max_allowed_drop = max_allowed_drop
        self.history: List[float] = []

    def record_query(self, reward: float) -> Tuple[bool, float]:
        """
        Record a live query reward.
        Returns (trigger_rollback: bool, rolling_mean: float).
        """
        self.history.append(reward)
        if len(self.history) < 5:
            # Warm-up period before enforcing watchdog
            return False, float(np.mean(self.history))

        recent = self.history[-self.window_size:]
        rolling_mean = float(np.mean(recent))

        # Check for degradation beyond allowed drop
        drop = self.baseline_reward - rolling_mean
        if drop > self.max_allowed_drop:
            return True, rolling_mean

        return False, rolling_mean
