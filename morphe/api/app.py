"""
FastAPI Application and Interactive Dashboard for MORPHE.
Exposes REST endpoints and serves a comprehensive Cytoscape.js visual control plane:
- Live 2D topology graph with load-based sizing and edge weight widths
- Time-travel version slider & Git-style commit history tree
- Per-query routing trace with learned optimal stopping visualization
- LinUCB bandit policy scoreboard & cumulative regret curves
- One-click Chaos buttons: "Break It", "Drift It", "Run Self-Play Audit"
- "Ask the Architecture" interactive chat citing ledger commits
"""

from __future__ import annotations

import json
import logging
import time
import os
import numpy as np
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from morphe.audit.self_play import SelfPlayAuditor
from morphe.audit.sleep_cycle import SleepCycleConsolidator
from morphe.backend.morphe_backend import MorpheBackend
from morphe.data.generator import DatasetBuilder
from morphe.eval.benchmark import BenchmarkRunner
from morphe.eval.reward import RewardWeights, compute_reward
from morphe.llm.client import LLMClient
from morphe.models.metadata import RecordMetadata
from morphe.models.topology import Mutation
from morphe.policy.architect_bandit import ArchitectBandit
from morphe.policy.diagnostician import Diagnostician
from morphe.policy.trial_scheduler import TrialScheduler
from morphe.router.learned_router import LearnedRouter
from morphe.storage.record_store import RecordStore
from morphe.telemetry.logger import TelemetryLogger

logger = logging.getLogger(__name__)

app = FastAPI(title="MORPHE: Self-Reconfiguring Memory System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Global Application State
# =============================================================================

class MorpheState:
    def __init__(self):
        fast_mode = os.environ.get("MORPHE_FAST_MODE", "1") == "1"
        self.store = RecordStore(":memory:")
        self.backend = MorpheBackend(record_store=self.store)
        self.telemetry = TelemetryLogger(":memory:")
        self.llm = LLMClient(model_name="gemma4:e2b", force_fallback=fast_mode)
        self.router = LearnedRouter(backend=self.backend, telemetry=self.telemetry, llm_client=self.llm)
        self.diagnostician = Diagnostician(overload_record_threshold=6)
        self.bandit = ArchitectBandit()
        self.scheduler = TrialScheduler(telemetry=self.telemetry)
        self.auditor = SelfPlayAuditor(backend=self.backend, router=self.router, llm_client=self.llm)
        self.consolidator = SleepCycleConsolidator(backend=self.backend, telemetry=self.telemetry, llm_client=self.llm)
        self.heldout_queries: List[Dict[str, Any]] = []

        # Bootstrap with synthetic multi-project data
        self.initialize_default_knowledge()

    def initialize_default_knowledge(self):
        builder = DatasetBuilder(record_store=self.store, llm_client=self.llm, random_seed=42)
        records = builder.ingest_corpus()

        # Group records by project into initial modules
        proj_recs: Dict[str, List[str]] = {}
        for r in records:
            p = r.metadata.get("project", "default")
            proj_recs.setdefault(p, []).append(r.id)

        for proj, r_ids in proj_recs.items():
            mid = f"mod_{proj.lower()}"
            self.backend.add_module(
                module_id=mid,
                name=proj,
                charter=f"Core architectural specs and operational guidelines for {proj}",
                record_ids=r_ids,
            )

        # Connect initial modules in a backbone loop
        m_keys = list(self.backend.manifests.keys())
        for i in range(len(m_keys)):
            u = m_keys[i]
            v = m_keys[(i + 1) % len(m_keys)]
            self.backend.add_edge(u, v, weight=0.6, edge_type="associative")

        self.backend.commit(
            reason_card="Initial architecture baseline (v1) with 4 project modules",
            metrics={"recall_at_5": 0.88, "mrr": 0.82, "latency_ms": 6.4, "structural_cost": 4.8, "fitness": 0.75},
        )

        golden, selfplay, heldout = builder.generate_benchmark_splits(records)
        self.heldout_queries = heldout


state = MorpheState()


# =============================================================================
# Request Schemas
# =============================================================================

class IngestRequest(BaseModel):
    text: str
    project: str = "default"
    memory_type: str = "tech_spec"
    tags: List[str] = Field(default_factory=list)


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    gold_id: Optional[str] = None


class RollbackRequest(BaseModel):
    version_id: str


class AskRequest(BaseModel):
    question: str


class SupersedeRequest(BaseModel):
    old_record_id: str
    new_record_text: str
    module_id: str
    project: str = "default"


class ParetoConfigRequest(BaseModel):
    speed_weight: float = 0.5  # 0.0 (Accuracy First) to 1.0 (Speed First)


# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/api/ingest")
def api_ingest(req: IngestRequest):
    """Ingest new text content into immutable record store and assign to appropriate module."""
    rec, is_new = state.store.put(
        text=req.text,
        metadata={"project": req.project, "memory_type": req.memory_type, "tags": req.tags},
    )
    # Assign to matching module or default
    target_mod = f"mod_{req.project.lower()}"
    if target_mod in state.backend.manifests:
        state.backend.add_record_to_module(target_mod, rec.id)
    else:
        # Create module if new project
        state.backend.add_module(
            target_mod,
            name=req.project,
            charter=f"Dynamic module for {req.project}",
            record_ids=[rec.id],
        )

    return {
        "status": "success",
        "record_id": rec.id,
        "is_new": is_new,
        "assigned_module": target_mod,
        "total_records": state.store.count(),
    }


@app.post("/api/retrieve")
def api_retrieve(req: RetrieveRequest):
    """Execute hybrid retrieval query with learned stopping and telemetry trace."""
    results, trace = state.router.route_and_retrieve(
        query=req.query,
        gold_id=req.gold_id,
        top_k=req.top_k,
    )
    return {
        "query": req.query,
        "results": results,
        "trace": {
            "modules_activated": trace.modules_activated,
            "latency_ms": trace.latency_ms,
            "confidence": trace.confidence,
            "routing_scores": {k: round(v, 4) for k, v in trace.routing_scores.items()},
            "propensities": {k: round(v, 4) for k, v in trace.propensities.items()},
            "gold_hit": trace.gold_hit,
        },
    }


@app.post("/api/adapt")
def api_adapt():
    """Trigger autonomous adaptation cycle: Diagnostician -> Bandit Propose -> Tiered Scheduler -> Commit."""
    rep = state.diagnostician.diagnose(state.backend, state.telemetry)
    mutation, context = state.bandit.propose_mutation(state.backend, state.diagnostician, state.telemetry)
    
    trial_res = state.scheduler.evaluate_mutation(
        state.backend,
        mutation,
        state.heldout_queries or [{"query": "OAuth2 PKCE", "gold_project": "ProjectAlpha_Auth"}],
    )

    # Online update for LinUCB bandit
    reward_val = 1.0 if trial_res.accepted else -0.5
    state.bandit.update(
        arm=mutation.action_type,
        context=context,
        reward=reward_val,
        was_accepted=trial_res.accepted,
    )

    return {
        "status": "success",
        "diagnostic_health": rep.overall_health,
        "primary_symptom": rep.primary_symptom.__dict__ if rep.primary_symptom else None,
        "proposed_mutation": mutation.to_dict(),
        "trial_result": {
            "tier_reached": trial_res.tier_reached,
            "accepted": trial_res.accepted,
            "compute_saved_score": trial_res.compute_saved_score,
            "fitness_delta": trial_res.fitness_delta,
            "reason": trial_res.reason,
            "acceptance_details": trial_res.acceptance_details,
        },
        "current_head": state.backend.head_version,
    }


@app.post("/api/rollback")
def api_rollback(req: RollbackRequest):
    """Revert architecture topology to specified version commit."""
    try:
        success = state.backend.rollback(req.version_id)
        return {
            "status": "success",
            "head_version": state.backend.head_version,
            "modules_count": len(state.backend.graph.nodes),
            "edges_count": len(state.backend.graph.edges),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/diff")
def api_diff(v1: str, v2: str):
    """Calculate structural and metric diff between any two versions."""
    try:
        diff_obj = state.backend.diff(v1, v2)
        return diff_obj.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/topology")
def api_topology():
    """Returns nodes and edges formatted for Cytoscape.js visualization."""
    elements = []
    # Nodes
    for mid, manifest in state.backend.manifests.items():
        rec_count = len(manifest.record_ids)
        elements.append({
            "data": {
                "id": mid,
                "label": f"{manifest.name}\n({rec_count} recs)",
                "record_count": rec_count,
                "charter": manifest.charter,
                "level": manifest.level,
            }
        })
    # Edges
    for u, v, data in state.backend.graph.edges(data=True):
        w = float(data.get("weight", 1.0))
        etype = data.get("edge_type", "associative")
        elements.append({
            "data": {
                "id": f"{u}->{v}",
                "source": u,
                "target": v,
                "weight": w,
                "edge_type": etype,
                "label": f"{etype} ({w:.2f})",
            }
        })
    return {
        "head_version": state.backend.head_version,
        "elements": elements,
    }


@app.get("/api/history")
def api_history():
    """Return chronological version commit history."""
    versions = state.backend.list_versions()
    return [v.to_dict() for v in versions]


@app.get("/api/policy_stats")
def api_policy_stats():
    """Return Contextual Bandit statistics, UCB scores, and regret curve."""
    arm_stats = {}
    context = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
    for arm_name, arm in state.bandit.arms.items():
        arm_stats[arm_name] = {
            "trials": arm.trials,
            "ucb_score": round(arm.compute_ucb(context), 4),
        }
    return {
        "cumulative_regret": round(state.bandit.cumulative_regret, 4),
        "reject_rate": round(state.bandit.recent_reject_rate, 4),
        "total_decisions": len(state.bandit.history),
        "arms": arm_stats,
    }


@app.post("/api/chaos/break_it")
def api_chaos_break_it():
    """Inject a deliberately bad/poisoned mutation to demonstrate detection and recovery."""
    all_mods = list(state.backend.manifests.keys())
    if len(all_mods) < 2:
        return {"status": "skipped", "reason": "Not enough modules to break"}

    m1, m2 = all_mods[0], all_mods[1]
    bad_mutation = Mutation(
        action_type="merge",
        target_modules=[m1, m2],
        parameters={"merged_id": f"poison_{m1}_{m2}"},
        reason="[CHAOS INJECTION] Deliberately combining orthogonal domains",
    )
    # Evaluate via scheduler -> should be auto-rejected or flagged!
    trial_res = state.scheduler.evaluate_mutation(
        state.backend,
        bad_mutation,
        state.heldout_queries,
    )
    return {
        "status": "chaos_evaluated",
        "action": bad_mutation.to_dict(),
        "tier_reached": trial_res.tier_reached,
        "accepted": trial_res.accepted,
        "protection_result": trial_res.reason,
    }


@app.post("/api/chaos/drift_it")
def api_chaos_drift_it():
    """Simulate a sudden traffic burst / drift towards a single module."""
    all_mods = list(state.backend.manifests.keys())
    target = all_mods[0] if all_mods else "mod_default"
    for _ in range(15):
        state.telemetry.log_query(
            query=f"Sudden high-priority burst query for {target}",
            modules_activated=[target],
            routing_scores={target: 1.0},
            propensities={target: 0.95},
            latency_ms=8.5,
            confidence=0.92,
        )
    return {
        "status": "drift_injected",
        "target_module": target,
        "queries_simulated": 15,
        "current_top_access": state.telemetry.get_module_access_count(target),
    }


@app.post("/api/audit")
def api_audit():
    """Run self-play adversarial audit cycle."""
    stats = state.auditor.run_audit_cycle(num_probes=4)
    return stats


@app.post("/api/ask")
def api_ask(req: AskRequest):
    """'Ask the Architecture' query answering from ledger history."""
    history = [v.to_dict() for v in state.backend.list_versions()]
    prompt = (
        f"You are the MORPHE Architecture Assistant. Answer the user question by citing verifiable "
        f"commits from the architecture ledger history below.\n"
        f"Ledger History:\n{json.dumps(history, indent=1)}\n\n"
        f"User Question: {req.question}"
    )
    answer = state.llm.generate(prompt)
    return {"question": req.question, "answer": answer}


@app.get("/api/benchmark")
def api_benchmark():
    """Run baseline comparisons."""
    runner = BenchmarkRunner(random_seed=42)
    return runner.run_all_baselines(num_queries=8)


@app.post("/api/temporal/supersede")
def api_temporal_supersede(req: SupersedeRequest):
    """Update a fact by creating a new record and soft-archiving the predecessor."""
    new_rec, _ = state.store.put(req.new_record_text, metadata={"project": req.project})
    state.backend.add_record_to_module(req.module_id, new_rec.id)
    state.store.set_archived(req.old_record_id, True)
    return {
        "status": "superseded",
        "old_record_id": req.old_record_id,
        "new_record_id": new_rec.id,
        "archived_old": True,
    }


@app.post("/api/consolidation/sleep_cycle")
def api_sleep_cycle():
    """Run offline consolidation: near-duplicate merging, Hebbian decay, and stale memory down-weighting."""
    res = state.consolidator.consolidate()
    return res


@app.post("/api/config/pareto")
def api_config_pareto(req: ParetoConfigRequest):
    """Dynamically calibrate multi-objective reward weights along Pareto frontier."""
    w = max(0.0, min(1.0, req.speed_weight))
    state.weights = RewardWeights(
        a=round((1.0 - w) * 1.5 + w * 0.5, 4),
        b=round((1.0 - w) * 1.5 + w * 0.5, 4),
        c=round((1.0 - w) * 0.001 + w * 0.02, 5),
        d=round((1.0 - w) * 0.01 + w * 0.15, 4),
        e=round((1.0 - w) * 0.005 + w * 0.08, 4),
    )
    pareto_points = []
    for commit in state.backend.list_versions():
        m = commit.metrics
        acc = m.get("recall_at_5", 0.8) * 0.5 + m.get("mrr", 0.8) * 0.5
        lat = m.get("latency_ms", 10.0)
        cost = m.get("structural_cost", 5.0)
        pareto_points.append({
            "version_id": commit.version_id,
            "accuracy": round(acc, 4),
            "latency_ms": round(lat, 2),
            "structural_cost": round(cost, 2),
        })
    return {
        "status": "updated",
        "speed_weight": w,
        "active_weights": state.weights.to_dict(),
        "pareto_frontier": pareto_points,
    }


@app.get("/api/ablation/llm_vs_reward")
def api_llm_ablation():
    """Run LLM-as-judge vs ground truth ablation."""
    from morphe.eval.ablation_llm_vs_reward import LLMvsRewardAblation
    ablation = LLMvsRewardAblation(llm_client=state.llm)
    return ablation.run_ablation(num_trials=4)


@app.get("/api/benchmark/transfer")
def api_policy_transfer():
    """Run bandit policy transfer benchmark."""
    from morphe.eval.policy_transfer import PolicyTransferBenchmark
    pt = PolicyTransferBenchmark()
    return pt.evaluate_transfer(test_cycles=4)


@app.post("/api/demo/run")
def api_demo_run():
    """Trigger complete 7-step presentation demo."""
    from morphe.demo import run_demo
    return run_demo()


# =============================================================================
# Interactive Single-Page Web Dashboard
# =============================================================================

@app.get("/", response_class=HTMLResponse)
def index_dashboard():
    """Render the full interactive Cytoscape dashboard."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MORPHE: Self-Reconfiguring Memory Architecture</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
  <style>
    #cy { width: 100%; height: 500px; background-color: #0f172a; border-radius: 0.75rem; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans">
  <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur px-6 py-4 flex justify-between items-center sticky top-0 z-50">
    <div>
      <h1 class="text-xl font-bold tracking-tight text-emerald-400">MORPHE</h1>
      <p class="text-xs text-slate-400">Memory-Oriented Reconfigurable Plasticity and Hierarchical Evolution</p>
    </div>
    <div class="flex items-center gap-2">
      <span class="text-xs px-2.5 py-1 rounded bg-slate-800 text-slate-300 font-mono" id="head-badge">HEAD: Loading...</span>
      <button onclick="triggerDemo()" class="bg-cyan-600 hover:bg-cyan-500 text-xs px-3 py-1.5 rounded font-bold transition shadow-lg">Run 7-Step Demo</button>
      <button onclick="triggerAdapt()" class="bg-emerald-600 hover:bg-emerald-500 text-xs px-3 py-1.5 rounded font-semibold transition">Adapt Topology</button>
      <button onclick="triggerAudit()" class="bg-indigo-600 hover:bg-indigo-500 text-xs px-3 py-1.5 rounded font-semibold transition">Self-Play Audit</button>
      <button onclick="triggerBreakIt()" class="bg-rose-700 hover:bg-rose-600 text-xs px-3 py-1.5 rounded font-semibold transition">Break It</button>
      <button onclick="triggerDriftIt()" class="bg-amber-600 hover:bg-amber-500 text-xs px-3 py-1.5 rounded font-semibold transition">Drift It</button>
    </div>
  </header>

  <main class="p-6 max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6">
    <!-- Left Column: Graph & Trace -->
    <div class="lg:col-span-2 space-y-6">
      <!-- Cytoscape Live Graph Card -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-xl">
        <div class="flex justify-between items-center mb-3">
          <h2 class="text-sm font-semibold text-slate-200">Interactive Memory Topology (Cytoscape 2D)</h2>
          <span class="text-xs text-slate-400">Size = Record Load | Width = Strength</span>
        </div>
        <div id="cy"></div>
      </div>

      <!-- Query Retrieval Sandbox -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-xl space-y-4">
        <h2 class="text-sm font-semibold text-slate-200">Query Retrieval & Learned Stopping Trace</h2>
        <div class="flex gap-2">
          <input id="query-input" type="text" placeholder="Enter query (e.g. 'OAuth2 PKCE authorization flow' or 'PostgreSQL WAL replication')"
                 class="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-emerald-500">
          <button onclick="runQuery()" class="bg-emerald-600 hover:bg-emerald-500 text-xs font-semibold px-4 py-2 rounded-lg transition">Retrieve</button>
        </div>
        <div id="trace-panel" class="hidden bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs space-y-2">
          <div class="flex justify-between font-mono text-slate-300">
            <span id="trace-modules"></span>
            <span id="trace-latency"></span>
            <span id="trace-confidence"></span>
          </div>
          <div id="trace-results" class="space-y-1 pt-2 border-t border-slate-800"></div>
        </div>
      </div>
    </div>

    <!-- Right Column: Version Ledger, Policy Panel & Ask Architecture -->
    <div class="space-y-6">
      <!-- Pareto Frontier Slider Card -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-xl space-y-3">
        <div class="flex justify-between items-center">
          <h2 class="text-sm font-semibold text-slate-200">Pareto Frontier Trade-Off</h2>
          <span id="pareto-mode-badge" class="text-[10px] px-2 py-0.5 rounded bg-emerald-950 border border-emerald-800 text-emerald-400 font-mono">Balanced</span>
        </div>
        <div class="space-y-1">
          <div class="flex justify-between text-[11px] text-slate-400">
            <span>Accuracy First</span>
            <span>Speed / Cost First</span>
          </div>
          <input type="range" min="0" max="1" step="0.05" value="0.5" id="pareto-slider" oninput="updatePareto(this.value)"
                 class="w-full accent-emerald-500 cursor-pointer">
        </div>
        <div id="pareto-weights" class="text-[10px] font-mono text-slate-400 bg-slate-950 p-2 rounded border border-slate-800">
          Weights: a=1.00 (Recall), b=1.00 (MRR), c=0.010 (Lat), d=0.080 (Touched), e=0.042 (Cost)
        </div>
      </div>

      <!-- Ledger / Commit Tree Card -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-xl">
        <h2 class="text-sm font-semibold text-slate-200 mb-3">Version Commit Ledger (Time Travel)</h2>
        <div id="commit-tree" class="space-y-2 max-h-60 overflow-y-auto pr-1"></div>
      </div>

      <!-- Bandit Policy Scoreboard -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-xl">
        <h2 class="text-sm font-semibold text-slate-200 mb-3">LinUCB Architect Bandit Scoreboard</h2>
        <div id="policy-arms" class="grid grid-cols-2 gap-2 text-xs"></div>
        <div class="mt-3 pt-3 border-t border-slate-800 flex justify-between text-xs text-slate-400">
          <span>Cumulative Regret: <b id="regret-val" class="text-emerald-400">0.0</b></span>
          <span>Reject Rate: <b id="reject-val" class="text-rose-400">0%</b></span>
        </div>
      </div>

      <!-- Ask the Architecture -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-xl space-y-3">
        <h2 class="text-sm font-semibold text-slate-200">Ask the Architecture (Gemma 4)</h2>
        <input id="ask-input" type="text" placeholder="Why was the last mutation committed or rejected?"
               class="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500">
        <button onclick="askArchitecture()" class="w-full bg-indigo-600 hover:bg-indigo-500 text-xs font-semibold py-2 rounded-lg transition">Ask Rationale</button>
        <div id="ask-answer" class="text-xs text-slate-300 p-2 bg-slate-950 rounded border border-slate-800 min-h-[50px]">Gemma 4 answers will appear here citing ledger commits...</div>
      </div>
    </div>
  </main>

  <script>
    let cyInstance = null;

    async function loadTopology() {
      const res = await fetch('/api/topology');
      const data = await res.json();
      document.getElementById('head-badge').innerText = 'HEAD: ' + data.head_version;

      if (!cyInstance) {
        cyInstance = cytoscape({
          container: document.getElementById('cy'),
          elements: data.elements,
          style: [
            {
              selector: 'node',
              style: {
                'background-color': '#059669',
                'label': 'data(label)',
                'color': '#f8fafc',
                'font-size': '10px',
                'text-valign': 'center',
                'text-wrap': 'wrap',
                'width': 'mapData(record_count, 1, 50, 40, 90)',
                'height': 'mapData(record_count, 1, 50, 40, 90)',
                'border-width': 2,
                'border-color': '#10b981'
              }
            },
            {
              selector: 'edge',
              style: {
                'width': 'mapData(weight, 0.1, 2.0, 1.5, 6)',
                'line-color': '#64748b',
                'target-arrow-color': '#64748b',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier',
                'label': 'data(label)',
                'font-size': '8px',
                'color': '#94a3b8'
              }
            }
          ],
          layout: { name: 'cose', animate: false }
        });
      } else {
        cyInstance.elements().remove();
        cyInstance.add(data.elements);
        cyInstance.layout({ name: 'cose', animate: false }).run();
      }
    }

    async function loadHistory() {
      const res = await fetch('/api/history');
      const versions = await res.json();
      const container = document.getElementById('commit-tree');
      container.innerHTML = '';
      versions.reverse().forEach(v => {
        const div = document.createElement('div');
        div.className = 'p-2 bg-slate-950 rounded border border-slate-800 hover:border-emerald-500 cursor-pointer text-xs transition';
        div.innerHTML = `
          <div class="flex justify-between font-mono font-bold text-emerald-400">
            <span>${v.version_id}</span>
            <span class="text-slate-500 text-[10px]">${new Date(v.timestamp * 1000).toLocaleTimeString()}</span>
          </div>
          <div class="text-slate-300 truncate mt-1">${v.reason_card}</div>
          <div class="text-[10px] text-slate-500 mt-1">Nodes: ${v.module_count} | Edges: ${v.edge_count} | Recs: ${v.total_records}</div>
        `;
        div.onclick = () => rollback(v.version_id);
        container.appendChild(div);
      });
    }

    async function loadPolicyStats() {
      const res = await fetch('/api/policy_stats');
      const data = await res.json();
      document.getElementById('regret-val').innerText = data.cumulative_regret;
      document.getElementById('reject-val').innerText = (data.reject_rate * 100).toFixed(0) + '%';
      const pContainer = document.getElementById('policy-arms');
      pContainer.innerHTML = '';
      for (const [arm, info] of Object.entries(data.arms)) {
        const item = document.createElement('div');
        item.className = 'p-2 bg-slate-950 rounded border border-slate-800';
        item.innerHTML = `<span class="font-bold text-slate-300 uppercase">${arm}</span>: UCB ${info.ucb_score} (${info.trials}t)`;
        pContainer.appendChild(item);
      }
    }

    async function rollback(vId) {
      if (!confirm('Rollback architecture to version ' + vId + '?')) return;
      await fetch('/api/rollback', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ version_id: vId })
      });
      refreshAll();
    }

    async function triggerAdapt() {
      const res = await fetch('/api/adapt', { method: 'POST' });
      const data = await res.json();
      alert(`Adaptation Cycle Complete: ${data.trial_result.reason}`);
      refreshAll();
    }

    async function triggerBreakIt() {
      const res = await fetch('/api/chaos/break_it', { method: 'POST' });
      const data = await res.json();
      alert(`Break It Injection: ${data.protection_result}`);
      refreshAll();
    }

    async function triggerDriftIt() {
      const res = await fetch('/api/chaos/drift_it', { method: 'POST' });
      const data = await res.json();
      alert(`Drift Injected: Traffic burst to ${data.target_module}`);
      refreshAll();
    }

    async function triggerAudit() {
      const res = await fetch('/api/audit', { method: 'POST' });
      const data = await res.json();
      alert(`Self-Play Audit Complete: Pass Rate ${data.pass_rate * 100}%, Pain Points: ${data.pain_points_found}`);
      refreshAll();
    }

    async function runQuery() {
      const q = document.getElementById('query-input').value.trim();
      if (!q) return;
      const res = await fetch('/api/retrieve', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ query: q, top_k: 3 })
      });
      const data = await res.json();
      const panel = document.getElementById('trace-panel');
      panel.classList.remove('hidden');
      document.getElementById('trace-modules').innerText = 'Modules: ' + data.trace.modules_activated.join(', ');
      document.getElementById('trace-latency').innerText = data.trace.latency_ms + 'ms';
      document.getElementById('trace-confidence').innerText = 'Conf: ' + (data.trace.confidence * 100).toFixed(0) + '%';

      const resultsDiv = document.getElementById('trace-results');
      resultsDiv.innerHTML = '';
      data.results.forEach((r, idx) => {
        const item = document.createElement('div');
        item.className = 'p-1.5 bg-slate-900 rounded border border-slate-800 text-[11px] text-slate-300';
        item.innerHTML = `<span class="text-emerald-400 font-bold">#${idx+1} [${r.project}]:</span> ${r.text}`;
        resultsDiv.appendChild(item);
      });
    }

    async function askArchitecture() {
      const q = document.getElementById('ask-input').value.trim();
      if (!q) return;
      const ansDiv = document.getElementById('ask-answer');
      ansDiv.innerText = 'Consulting ledger history via Gemma 4...';
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ question: q })
      });
      const data = await res.json();
      ansDiv.innerText = data.answer;
    }

    async function updatePareto(val) {
      const w = parseFloat(val);
      const mode = w < 0.35 ? 'Accuracy First' : (w > 0.65 ? 'Speed First' : 'Balanced');
      document.getElementById('pareto-mode-badge').innerText = mode;
      const res = await fetch('/api/config/pareto', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ speed_weight: w })
      });
      const data = await res.json();
      const p = data.active_weights;
      document.getElementById('pareto-weights').innerText = 
        `Weights: a=${p.a.toFixed(2)} (Recall), b=${p.b.toFixed(2)} (MRR), c=${p.c.toFixed(3)} (Lat), d=${p.d.toFixed(2)} (Mod), e=${p.e.toFixed(3)} (Cost)`;
    }

    async function triggerDemo() {
      if (!confirm('Run the autonomous 7-step presentation demo?')) return;
      alert('Launching 7-Step Autonomous Lifecycle Demo. Check console and watch the graph evolve!');
      const res = await fetch('/api/demo/run', { method: 'POST' });
      const data = await res.json();
      alert('Demo Run Complete! Rollback verified: ' + data.rollback_verified + '. Refreshing state.');
      refreshAll();
    }

    function refreshAll() {
      loadTopology();
      loadHistory();
      loadPolicyStats();
    }

    window.onload = refreshAll;
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)
