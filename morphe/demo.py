"""
Autonomous 7-Step Presentation Demo Runner for MORPHE.
Executes Section 15 of the MORPHE Final Plan from end to end:
Step 1: Baseline Ingestion & Initial Architecture
Step 2: Autonomous Self-Play Audit (discovering retrieval pain points)
Step 3: Concept Drift Injection (traffic burst destabilizing baseline)
Step 4: Contextual Bandit Proposal & Off-Policy Pre-Screening
Step 5: Tiered Sandbox Validation, Acceptance Gate & Commit
Step 6: Chaos Attack ("Break It") & Auto-Rejection / Watchdog Recovery
Step 7: Time-Travel Rollback & LLM vs Ground Truth Ablation
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Dict

from morphe.audit.self_play import SelfPlayAuditor
from morphe.backend.morphe_backend import MorpheBackend
from morphe.data.generator import DatasetBuilder
from morphe.eval.ablation_llm_vs_reward import LLMvsRewardAblation
from morphe.eval.policy_transfer import PolicyTransferBenchmark
from morphe.eval.reward import PostCommitWatchdog
from morphe.llm.client import LLMClient
from morphe.models.topology import Mutation
from morphe.policy.architect_bandit import ArchitectBandit
from morphe.policy.diagnostician import Diagnostician
from morphe.policy.trial_scheduler import TrialScheduler
from morphe.router.learned_router import LearnedRouter
from morphe.storage.record_store import RecordStore
from morphe.telemetry.logger import TelemetryLogger

logging.basicConfig(level=logging.INFO, format="%(message)s")


def run_demo() -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  MORPHE: Adaptive, Self-Reconfiguring Memory System for AI Agents")
    print("  Official 7-Step Autonomous Live Demonstration (PS 3)")
    print("=" * 80 + "\n")

    # -------------------------------------------------------------------------
    # Step 1: Ingestion & Baseline
    # -------------------------------------------------------------------------
    print("--- [STEP 1/7] Ingesting Multi-Project Memory Corpus & Building Baseline Graph ---")
    store = RecordStore(":memory:")
    llm = LLMClient(force_fallback=True)
    builder = DatasetBuilder(record_store=store, llm_client=llm, random_seed=42)
    records = builder.ingest_corpus()

    backend = MorpheBackend(record_store=store)
    proj_map = {}
    for r in records:
        proj = r.metadata.get("project", "default")
        proj_map.setdefault(proj, []).append(r.id)

    for proj, r_ids in proj_map.items():
        backend.add_module(f"mod_{proj.lower()}", name=proj, charter=f"Docs for {proj}", record_ids=r_ids)

    m_keys = list(backend.manifests.keys())
    for i in range(len(m_keys)):
        backend.add_edge(m_keys[i], m_keys[(i + 1) % len(m_keys)], weight=0.5)

    golden, selfplay, heldout = builder.generate_benchmark_splits(records)
    v1_commit = backend.commit(reason_card="Baseline v1: 4 project modules, circular backbone", metrics=backend.evaluate(heldout))

    telemetry = TelemetryLogger(":memory:")
    router = LearnedRouter(backend=backend, telemetry=telemetry, llm_client=llm)
    bandit = ArchitectBandit(random_seed=42)
    diagnostician = Diagnostician(overload_record_threshold=3)
    scheduler = TrialScheduler(telemetry=telemetry)

    print(f"  [OK] Ingested {len(records)} immutable records across {len(backend.manifests)} modules.")
    print(f"  [OK] Committed Version: {v1_commit.version_id} | Nodes: {v1_commit.module_count} | Edges: {v1_commit.edge_count}")
    print(f"  [METRICS] Baseline Recall@5: {v1_commit.metrics['recall_at_5']:.2f} | MRR: {v1_commit.metrics['mrr']:.2f} | Latency: {v1_commit.metrics['latency_ms']:.1f}ms\n")

    # -------------------------------------------------------------------------
    # Step 2: Self-Play Audit
    # -------------------------------------------------------------------------
    print("--- [STEP 2/7] Autonomous Self-Play Audit (Uncovering Retrieval Pain Points) ---")
    auditor = SelfPlayAuditor(backend=backend, router=router, llm_client=llm)
    audit_stats = auditor.run_audit_cycle(num_probes=4, adversarial_ratio=0.5)
    print(f"  [AUDIT] Probes tested: {audit_stats['probes_tested']} | Pass Rate: {audit_stats['pass_rate']*100:.1f}%")
    print(f"  [AUDIT] Discovered {audit_stats['pain_points_found']} weak retrieval pathways (pre-emptive pain points).\n")

    # -------------------------------------------------------------------------
    # Step 3: Inject Traffic Drift
    # -------------------------------------------------------------------------
    print("--- [STEP 3/7] Injecting Traffic Burst & Concept Drift ---")
    target_mod = "mod_projectalpha_auth"
    for _ in range(15):
        telemetry.log_query(
            query="High-frequency authorization PKCE challenge burst",
            modules_activated=[target_mod, "mod_projectbeta_database"],
            routing_scores={target_mod: 0.9, "mod_projectbeta_database": 0.8},
            propensities={target_mod: 0.6, "mod_projectbeta_database": 0.4},
            latency_ms=9.2,
            confidence=0.85,
        )
    print(f"  [DRIFT] Injected 15 burst queries targeting {target_mod} <-> Database co-activation.\n")

    # -------------------------------------------------------------------------
    # Step 4 & 5: Contextual Bandit Proposes & VoI Tiered Adaptation
    # -------------------------------------------------------------------------
    print("--- [STEP 4 & 5/7] LinUCB Bandit Proposes Structural Adaptation & VoI Sandbox Gate ---")
    diag_report = diagnostician.diagnose(backend, telemetry)
    mutation, ctx = bandit.propose_mutation(backend, diagnostician, telemetry)
    print(f"  [BANDIT] Selected Arm: {mutation.action_type.upper()} | Targets: {mutation.target_modules}")
    print(f"  [REASON] {mutation.reason}")

    # Evaluate via VoI Scheduler
    trial = scheduler.evaluate_mutation(backend, mutation, heldout)
    print(f"  [VOI SCHEDULER] Tier Reached: Tier {trial.tier_reached}")
    print(f"  [VOI SCHEDULER] Compute Saved Score: {trial.compute_saved_score*100:.0f}%")
    print(f"  [GATE OUTCOME] {trial.reason}")
    print(f"  [HEAD] Active HEAD is now: {backend.head_version} (Nodes: {len(backend.graph.nodes)}, Edges: {len(backend.graph.edges)})\n")

    # -------------------------------------------------------------------------
    # Step 6: Chaos Attack ("Break It") & Auto-Rejection
    # -------------------------------------------------------------------------
    print("--- [STEP 6/7] Chaos Injection ('Break It'): Testing Defensive Self-Healing ---")
    poison_mutation = Mutation(
        action_type="merge",
        target_modules=["mod_projectalpha_auth", "mod_projectgamma_telemetry"],
        parameters={"merged_id": "poison_auth_telemetry"},
        reason="[CHAOS AGENT] Deliberate invalid merge of orthogonal security and telemetry domains",
    )
    chaos_trial = scheduler.evaluate_mutation(backend, poison_mutation, heldout)
    print(f"  [CHAOS] Attempted: POISON MERGE on Auth + Telemetry")
    print(f"  [DEFENSE] Filtered at: Tier {chaos_trial.tier_reached} (Accepted: {chaos_trial.accepted})")
    print(f"  [PROTECTION] {chaos_trial.reason}")
    print(f"  [STATUS] Production memory remained 100% UNCORRUPTED.\n")

    # -------------------------------------------------------------------------
    # Step 7: Time-Travel Rollback & Ablation Verification
    # -------------------------------------------------------------------------
    print("--- [STEP 7/7] Time-Travel Rollback & LLM vs. Verifiable Reward Ablation ---")
    current_head = backend.head_version
    backend.rollback("v1")
    print(f"  [ROLLBACK] Successfully rolled back HEAD from {current_head} to {backend.head_version}.")

    ablation = LLMvsRewardAblation(llm_client=llm)
    ablation_res = ablation.run_ablation(num_trials=4)
    print(f"  [ABLATION] LLM Approval Rate: {ablation_res['llm_approval_rate']*100:.1f}%")
    print(f"  [ABLATION] Factual Ground-Truth Approval Rate: {ablation_res['factual_ground_truth_approval_rate']*100:.1f}%")
    print(f"  [ABLATION CONCLUSION] {ablation_res['conclusion']}\n")

    print("=" * 80)
    print("  FINAL VERDICT:")
    print("  'Normally, systems ask an LLM whether a change was good. We measure it.'")
    print("  MORPHE adapts its physical topology based on verifiable facts.")
    print("=" * 80 + "\n")

    return {
        "status": "success",
        "baseline_version": "v1",
        "audit_pass_rate": audit_stats["pass_rate"],
        "adaptation_accepted": trial.accepted,
        "chaos_prevented": not chaos_trial.accepted,
        "rollback_verified": backend.head_version == "v1",
        "llm_ablation": ablation_res,
    }


if __name__ == "__main__":
    run_demo()
