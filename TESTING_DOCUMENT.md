# MORPHE Adversarial & Hard Edge-Case Test Specification

This document details the test scenarios, stress invariants, failure modes tested, and instructions to execute and inspect the test suite.

---

## 1. Matrix of Tested Hard Edge Cases

| Subsystem | Hard Case / Adversarial Stress | Expected Invariant & Failure Defense | Test Function in [`test_hard_cases.py`](file:///c:/Users/Keya/Desktop/InnoviseMORPHE/tests/test_hard_cases.py) |
| :--- | :--- | :--- | :--- |
| **Record Store** | Multi-lingual text, emojis (🧠, 🚀), unicode control bytes (`\u0000`, `\u200b`), and massive 100,000-char strings. | Text is byte-accurately preserved, SHA-256 hash is invariant, and no memory corruption occurs. | `test_hard_record_store_extreme_inputs` |
| **Record Store** | Missing embedding upon initial ingest, followed by identical text with vector. | Deduplication detects identical text, preserves the original record ID, and upgrades the vector in SQLite without creating duplicate records. | `test_hard_record_store_extreme_inputs` |
| **Record Store** | Zero-norm vectors (all zeros). | Does not trigger division by zero or NaN norms during cosine normalization. | `test_hard_record_store_zero_vector_handling` |
| **Hybrid Chunker** | Continuous raw text stream with zero punctuation marks (`.`, `!`, `?`). | Chunker avoids hanging or losing data; produces valid chunks respecting length bounds without data loss. | `test_hard_chunker_no_punctuation_stream` |
| **Hybrid Chunker** | Abbreviations (`Dr.`, `U.S.A.`, `Jan.`) and decimal points (`3.14159`, `10.5 GHz`). | Chunker protects abbreviations and decimals; splits only on true sentence boundaries. | `test_hard_chunker_abbreviations_and_decimals` |
| **Hybrid Chunker** | Sub-minimum strings shorter than `min_chars`. | Chunker yields single chunk without dropping content. | `test_hard_chunker_sub_minimum_string` |
| **Backend & Ledger** | Calling `rollback()` on non-existent or corrupted version IDs. | Returns `False` gracefully without crashing or corrupting the HEAD pointer. | `test_hard_backend_invalid_rollback_and_missing_versions` |
| **Backend & Ledger** | Calling `diff(v1, v2)` on invalid or uncommitted version IDs. | Explicitly raises `KeyError` with clear diagnostic information. | `test_hard_backend_invalid_rollback_and_missing_versions` |
| **Backend Fork** | Chained cascading splits in an isolated sandbox branch. | Mutations applied in sandbox forks never leak or pollute parent backend nodes, edges, or ledger history. | `test_hard_backend_cascading_splits_and_isolation` |
| **Mutation Engine** | Overloaded module where all records have identical embeddings. | K-Means convergence warning handled; system falls back to balanced partition without dividing by zero. | `test_hard_mutation_split_identical_embeddings` |
| **Mutation Engine** | Merge proposed on semantically identical modules whose combined size would immediately exceed the split limit. | Merge is rejected to prevent thrashing cycles (immediate split after merge). | `test_hard_mutation_merge_size_limit_rejection` |
| **Temporal Memory** | Chained supersessions ($A \rightarrow B \rightarrow C$) with historical vs current queries. | Current query prefers newest fact $C$; historical query accurately resolves to prior facts $A$ and $B$; intermediate versions are restorable via rollback. | `test_hard_temporal_chained_supersedes` |
| **Sleep Cycle** | Module with completely orthogonal embeddings. | Consolidator detects zero similarity; never combines orthogonal facts. | `test_hard_sleep_cycle_empty_and_orthogonal` |

---

## 2. Test Execution Commands

### A. Run Only the Adversarial / Hard Cases
Run the dedicated hard test suite with verbose output:
```powershell
python -m pytest tests/test_hard_cases.py -v -s
```

### B. Run the Entire MORPHE Test Suite (53 Tests)
Run all unit, integration, and edge-case tests:
```powershell
python -m pytest tests/ -v
```

### C. Run with Detailed Timing and Durations
Inspect the slowest test cases to ensure real-time responsiveness:
```powershell
python -m pytest tests/ --durations=10
```

### D. Interactive Python Verification Script
To manually inspect the behavior in a Python shell or notebook:
```python
import numpy as np
from morphe.core.backend import MorpheBackend
from morphe.storage.record_store import RecordStore
from morphe.temporal.manager import TemporalMemoryManager
from morphe.topology.models import ModuleManifest, TopologySnapshot

# 1. Initialize store and create a sequence of facts
store = RecordStore(":memory:")
fA, _ = store.put("We use MySQL 5.7.")
fB, _ = store.put("Migrated from MySQL to PostgreSQL 14.")
fC, _ = store.put("Upgraded database to distributed CockroachDB.")

# 2. Build backend
m = ModuleManifest.create("mod_db", [fA.id, fB.id, fC.id])
snap = TopologySnapshot.create(modules={"mod_db": m}, edges=[])
backend = MorpheBackend(record_store=store, initial_snapshot=snap)
v0 = backend.head_version_id

# 3. Apply chained supersessions
temporal = TemporalMemoryManager(backend)
temporal.record_supersession(fB.id, fA.id, "mod_db")
v1 = backend.commit("MySQL -> Postgres")

temporal.record_supersession(fC.id, fB.id, "mod_db")
v2 = backend.commit("Postgres -> CockroachDB")

# 4. Test Current vs Historical Queries
candidates = [(fA.id, 0.9), (fB.id, 0.9), (fC.id, 0.9)]
current_res = temporal.resolve_temporal_records(candidates, "What is our current database?")
historical_res = temporal.resolve_temporal_records(candidates, "What did we use historically or before?")

print("=== Current Query Top Result ===")
print(store.get(current_res[0][0]).text)  # CockroachDB

print("\n=== Historical Query Top Result ===")
print(store.get(historical_res[0][0]).text)  # MySQL or Postgres

print("\n=== Rollback Test ===")
backend.rollback(v1.version_id)
print("Rolled back HEAD to:", backend.head_version_id)
print("Ledger entry count:", len(backend.list_versions()))
```

