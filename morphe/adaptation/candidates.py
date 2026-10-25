"""
Candidate mutation models and ranking scores for MORPHE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from morphe.topology.mutations import MutationAction, MutationType


@dataclass
class MutationCandidate:
    """
    An eligible candidate mutation discovered by the Mutation Engine.
    Carries eligibility justification and priority score.
    """
    action: MutationAction
    priority_score: float = 0.0
    symptom: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def mutation_type(self) -> MutationType:
        return self.action.mutation_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mutation_type": self.mutation_type.value,
            "source_module": self.action.source_module,
            "target_module": self.action.target_module,
            "new_module_id": self.action.new_module_id,
            "priority_score": round(self.priority_score, 4),
            "symptom": self.symptom,
            "evidence": self.evidence,
        }

