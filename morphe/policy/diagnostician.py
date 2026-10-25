"""
Diagnostician for MORPHE.
Rule-based, interpretable structural diagnostic engine.
Scans current topology and telemetry to identify structural weaknesses with numeric severity.
Outputs symptoms that feed the Contextual Bandit's state vector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from morphe.backend.morphe_backend import MorpheBackend
from morphe.telemetry.logger import TelemetryLogger

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticSymptom:
    """A detected structural symptom in the memory graph."""
    symptom_type: str  # "overloaded", "redundant", "missing_shortcut", "cold", "fragmented"
    severity: float    # 0.0 to 1.0
    target_modules: List[str]
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""


@dataclass
class DiagnosticReport:
    """Consolidated diagnostic report of the architecture."""
    symptoms: List[DiagnosticSymptom]
    overall_health: float  # 1.0 = optimal, 0.0 = severe pathology
    primary_symptom: Optional[DiagnosticSymptom] = None

    def get_severity_vector(self) -> Dict[str, float]:
        """Summary severity per symptom type for bandit context."""
        vec = {
            "overloaded": 0.0,
            "redundant": 0.0,
            "missing_shortcut": 0.0,
            "cold": 0.0,
            "fragmented": 0.0,
        }
        for s in self.symptoms:
            if s.symptom_type in vec:
                vec[s.symptom_type] = max(vec[s.symptom_type], s.severity)
        return vec


class Diagnostician:
    """
    Evaluates graph health and identifies candidates for split, merge, shortcut, prune, and promote.
    """

    def __init__(
        self,
        overload_record_threshold: int = 150,
        redundancy_cosine_threshold: float = 0.88,
        shortcut_co_access_threshold: int = 5,
        cold_edge_threshold: float = 0.05,
    ):
        self.overload_thresh = overload_record_threshold
        self.redundancy_thresh = redundancy_cosine_threshold
        self.shortcut_thresh = shortcut_co_access_threshold
        self.cold_thresh = cold_edge_threshold

    def diagnose(
        self,
        backend: MorpheBackend,
        telemetry: Optional[TelemetryLogger] = None,
    ) -> DiagnosticReport:
        """Run all diagnostic checks across graph and telemetry."""
        symptoms: List[DiagnosticSymptom] = []

        # 1. Check for overloaded modules (candidate for Split)
        for mid, manifest in backend.manifests.items():
            count = len(manifest.record_ids)
            if count >= self.overload_thresh:
                severity = min(1.0, count / (self.overload_thresh * 1.5))
                symptoms.append(DiagnosticSymptom(
                    symptom_type="overloaded",
                    severity=severity,
                    target_modules=[mid],
                    details={"record_count": count, "threshold": self.overload_thresh},
                    recommendation=f"Module '{mid}' is overloaded with {count} records. Candidate for Split.",
                ))
            elif count > 20:
                # Moderate load check (relative severity)
                severity = count / self.overload_thresh
                if severity > 0.4:
                    symptoms.append(DiagnosticSymptom(
                        symptom_type="overloaded",
                        severity=round(severity, 3),
                        target_modules=[mid],
                        details={"record_count": count},
                        recommendation=f"Module '{mid}' approaching overload ({count} records).",
                    ))

        # 2. Check for redundant modules (candidate for Merge)
        modules = list(backend.manifests.items())
        for i in range(len(modules)):
            for j in range(i + 1, len(modules)):
                m1_id, man1 = modules[i]
                m2_id, man2 = modules[j]
                if man1.centroid is not None and man2.centroid is not None:
                    cos_sim = float(np.dot(man1.centroid, man2.centroid))
                    if cos_sim >= self.redundancy_thresh:
                        sev = min(1.0, (cos_sim - self.redundancy_thresh) / (1.0 - self.redundancy_thresh + 1e-6))
                        symptoms.append(DiagnosticSymptom(
                            symptom_type="redundant",
                            severity=round(sev, 3),
                            target_modules=[m1_id, m2_id],
                            details={"cosine_similarity": round(cos_sim, 4)},
                            recommendation=f"Modules '{m1_id}' and '{m2_id}' are redundant (cosine={cos_sim:.2f}). Candidate for Merge.",
                        ))

        # 3. Check for missing shortcuts (candidate for Shortcut)
        if telemetry is not None:
            top_pairs = telemetry.get_top_co_accessed_pairs(limit=15)
            for (m1, m2), co_count in top_pairs:
                if m1 in backend.manifests and m2 in backend.manifests:
                    if co_count >= self.shortcut_thresh and not backend.graph.has_edge(m1, m2):
                        sev = min(1.0, co_count / (self.shortcut_thresh * 2.5))
                        symptoms.append(DiagnosticSymptom(
                            symptom_type="missing_shortcut",
                            severity=round(sev, 3),
                            target_modules=[m1, m2],
                            details={"co_access_count": co_count},
                            recommendation=f"Modules '{m1}' and '{m2}' co-accessed {co_count} times without direct edge. Candidate for Shortcut.",
                        ))

        # 4. Check for cold edges (candidate for Prune)
        for u, v, data in backend.graph.edges(data=True):
            w = float(data.get("weight", 1.0))
            if w <= self.cold_thresh:
                symptoms.append(DiagnosticSymptom(
                    symptom_type="cold",
                    severity=round(1.0 - (w / max(1e-6, self.cold_thresh)), 3),
                    target_modules=[u, v],
                    details={"edge": (u, v), "weight": w},
                    recommendation=f"Edge ({u}->{v}) has decayed to weight {w:.3f}. Candidate for Prune.",
                ))

        # Compute overall health
        if not symptoms:
            overall_health = 1.0
            primary = None
        else:
            symptoms.sort(key=lambda x: x.severity, reverse=True)
            primary = symptoms[0]
            max_sev = primary.severity
            overall_health = max(0.0, 1.0 - 0.4 * max_sev - 0.1 * min(len(symptoms), 5))

        return DiagnosticReport(
            symptoms=symptoms,
            overall_health=round(overall_health, 3),
            primary_symptom=primary,
        )
