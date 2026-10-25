"""
Standardized metadata model for MORPHE records.
Supports:
- project: string identifier (e.g. 'ProjectA', 'CoreAuth', 'Telemetry')
- memory_type: Semantic categorization ('preference', 'fact', 'doc', 'interaction', 'tech_spec', etc.)
- timestamp: ISO 8601 string or epoch float
- tags: List of keyword tags
- source_id / doc_id: Document provenance pointer
- temporal validity window: valid_from, valid_to for time-travel queries
"""

from __future__ import annotations

import datetime
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class MemoryType(str, Enum):
    """Core memory classification types."""
    FACT = "fact"
    PREFERENCE = "preference"
    DOC = "doc"
    INTERACTION = "interaction"
    TECH_SPEC = "tech_spec"
    CODE = "code"
    SUMMARY = "summary"
    UNKNOWN = "unknown"

    @classmethod
    def from_str(cls, val: str) -> MemoryType:
        try:
            return cls(val.lower())
        except ValueError:
            return cls.UNKNOWN


@dataclass
class RecordMetadata:
    """
    Structured metadata associated with an immutable Record.
    """
    project: str = "default"
    memory_type: MemoryType = MemoryType.FACT
    timestamp: float = field(default_factory=time.time)
    iso_time: str = field(default="")
    source_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    valid_from: Optional[float] = None
    valid_to: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.iso_time:
            self.iso_time = datetime.datetime.fromtimestamp(
                self.timestamp, tz=datetime.timezone.utc
            ).isoformat()
        if isinstance(self.memory_type, str) and not isinstance(self.memory_type, MemoryType):
            self.memory_type = MemoryType.from_str(self.memory_type)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        d = dict(self.extra)
        mem_val = self.memory_type.value if isinstance(self.memory_type, MemoryType) else str(self.memory_type)
        d.update({
            "project": self.project,
            "memory_type": mem_val,
            "type": mem_val,
            "timestamp": self.timestamp,
            "iso_time": self.iso_time,
            "source_id": self.source_id,
            "tags": list(self.tags),
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
        })
        return d

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> RecordMetadata:
        """Parse dictionary or return default metadata."""
        if not data:
            return cls()
        
        # Support aliases like 'type' -> 'memory_type'
        mem_type = data.get("memory_type", data.get("type", MemoryType.FACT))
        if isinstance(mem_type, str):
            mem_type = MemoryType.from_str(mem_type)

        known_keys = {
            "project", "memory_type", "type", "timestamp", "iso_time",
            "source_id", "tags", "valid_from", "valid_to", "extra"
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}
        if "extra" in data and isinstance(data["extra"], dict):
            extra.update(data["extra"])

        return cls(
            project=data.get("project", "default"),
            memory_type=mem_type,
            timestamp=float(data.get("timestamp", time.time())),
            iso_time=data.get("iso_time", ""),
            source_id=data.get("source_id"),
            tags=list(data.get("tags", [])),
            valid_from=data.get("valid_from"),
            valid_to=data.get("valid_to"),
            extra=extra,
        )
