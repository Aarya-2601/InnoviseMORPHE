# MORPHE: Post-Stress Test Issues, Bottlenecks & Scaling Roadmap

> **Target Audience:** Engineering & RL Teammates  
> **Topic:** Scaling MORPHE from a Prototype to Long-Horizon Agentic AI Memory  
> **Author:** Antigravity Pairing Assistant  
> **Date:** October 2026  

---

## 1. Executive Summary & Stress Test Overview

We stress-tested the MORPHE repository against a synthetic **huge, multi-turn technical conversation** with large, complex answers (covering Raft consensus, RocksDB vs CockroachDB migrations, JWT vs HttpOnly session cookies, HNSW vector search, Kafka event streaming, and Kubernetes GitOps), as well as the repository's real-world chat log ([`data/gemini_chat_text.txt`](./data/gemini_chat_text.txt)).

### What Works Reliably
* **Sentence-Aligned Chunking**: Long, dense technical prose and code blocks are cleanly split via `HybridChunker` without dropping tokens.
* **Content-Addressable Storage (CAS)**: SHA-256 deduplication works out of the box; identical chunks in repeated turns are collapsed to single immutable records.
* **Ledger Versioning & Rollback**: Commits, parent pointers, deltas, and rollbacks function deterministically in SQLite.
* **Adaptation Diagnostics**: The `MutationEngine` accurately identifies overloaded modules based on threshold size.

However, scaling to large dialogues uncovered **six critical issues** across retrieval, topology generation, temporal memory, and codebase architecture that directly impact how the RL layer and the systems layer interact.

---

## 2. The 6 Core Issues Discovered

### Issue 1: Hardcoded 5-Module Clustering Bottleneck (`k <= 5`)
* **Location:** [`morphe/conversations/pipeline.py`](./morphe/conversations/pipeline.py)
* **The Code:**
  ```python
  k = max(2, min(5, num_records // 4)) if num_records >= 4 else 1
  ```
* **Why it breaks:** Capping $K$ at $5$ forces all records into at most five modules regardless of whether the transcript has 50 turns or 50,000 turns across 100 disparate topics.
* **Impact on RL & Retrieval:** Modules become massive and internally heterogeneous. The `MutationEngine` immediately flags every module as overloaded, while the semantic boundaries between concepts blur, making centroid routing ineffective.

### Issue 2: Hardcoded String Matching for `SUPERSEDES` Edges
* **Location:** [`morphe/conversations/pipeline.py`](./morphe/conversations/pipeline.py)
* **The Code:**
  ```python
  lt = c.text.lower()
  if "ppo" in lt and ("proximal policy" in lt or "deep reinforcement" in lt):
      ppo_record_id = rec.id
  if "contextual bandit" in lt or "linucb" in lt:
      bandit_record_id = rec.id
  ```
* **Why it breaks:** The supersession detector only looks for the literal strings `"ppo"` and `"contextual bandit"`.
* **Impact on RL & Retrieval:** In any real conversation where decisions or facts pivot (e.g., migrating from RocksDB to CockroachDB, or switching ports/tokens), `superseded_detected` remains `False`. No `SUPERSEDES` edge is drawn, old invalid facts are never soft-archived, and the agent continues to retrieve contradictory historical information.

### Issue 3: Echo-Chamber Retrieval (Returns User Questions Instead of Answers)
* **Location:** [`morphe/conversations/pipeline.py`](./morphe/conversations/pipeline.py)
* **The Code:**
  ```python
  hybrid_score = dense_sim + 0.6 * overlap
  ```
* **Why it breaks:** Both user prompts and assistant answers are stored in the same pool without distinction. When a user queries a concept (e.g., *"How do we hash the graphs during commits?"*), the query has a near 100% lexical and semantic match with the **prior user question** in the transcript.
* **Impact:** The system returns the past user prompt instead of the assistant's technical solution.

### Issue 4: Naive Regex Splits Truncating Synthesized Answers
* **Location:** [`morphe/conversations/pipeline.py`](./morphe/conversations/pipeline.py)
* **The Code:**
  ```python
  sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_lead) if len(s.strip()) > 15]
  ```
* **Why it breaks:** Splitting on punctuation without negative lookbehinds splits on numbered list items (`1.`, `2.`) or abbreviations.
* **Impact:** For a response like *"1. The Shift from PPO to Contextual Bandits..."*, the synthesized output was truncated to:  
  `"From 3 Reward Bandit Split: Here are the critical updates incorporated into this final plan: 1."`  
  cutting off before the actual fact.

### Issue 5: Linear $O(N)$ Python Scan in Retrieval
* **Location:** [`morphe/conversations/pipeline.py`](./morphe/conversations/pipeline.py)
* **The Code:**
  ```python
  for m_id, module in snapshot.modules.items():
      records = self.record_store.get_many(module.record_ids)
      for r_id, rec in records.items():
          dense_sim = float(np.dot(q_vec, rec.embedding))
  ```
* **Why it breaks:** Retrieval iterates sequentially through every module and every record in Python using `np.dot`. It bypasses the module centroids, spreading activation, and FAISS vector indexes.
* **Latency scaling:**
  * 150 records: ~25 ms (fine in tests).
  * 10,000 records: ~1.5 to 2.5 seconds.
  * 100,000 records: **15 to 25 seconds per query**.
  For long-horizon agents making multiple memory calls per planning step, a linear scan causes latency lockups.

### Issue 6: Architectural Split: Two Disconnected Backends
The repository currently maintains two separate, divergent backend implementations:

| Subsystem / Feature | `morphe.core.backend.MorpheBackend` | `morphe.backend.morphe_backend.MorpheBackend` |
| :--- | :--- | :--- |
| **Commit Ledger** | Persistent SQLite `VersionLedger` | In-memory Python dictionaries (`self.ledger: Dict`) |
| **Topology State** | Immutable `TopologySnapshot` dataclass | Mutable NetworkX DiGraph + manifests dict |
| **Mutation Action** | Uses `MutationAction` class | Uses `Mutation` dataclass or plain `dict` |
| **RL Evaluation Engine** | Missing `evaluate()` method | Built-in `evaluate()` (Recall@K, MRR, latency, cost) |
| **Connected Code** | Web Server (`server.py`), Y2K UI, Conversational Pipeline | LinUCB Bandit, Trial Scheduler, Learned Router, Benchmarks |

* **Why it matters:** The RL policies (LinUCB Bandit, IPS Off-Policy Screening) are tested on the in-memory mock backend, while the actual user ingestion pipeline and Y2K Cytoscape visualization run on the SQLite backend. They are currently decoupled.

---

## 3. Recommended Architectural Fixes & Next Steps

### 1. In Place of the $K \le 5$ Module Cap
* **Dynamic Square-Root Scaling:** Set $K = \max\left(2, \left\lfloor\sqrt{N / 2}\right\rfloor\right)$ with an optimal module size of **20–40 chunks**.
* **Autonomous Splitting:** Rely on the `MutationEngine` to split modules via 2-means clustering as record counts grow, rather than forcing a fixed cluster count at ingestion.
* **Hierarchical Modularity:** Use the existing `parent_module_id` field in `ModuleManifest` to group topic sub-modules under domain super-modules.

### 2. Generalizing `SUPERSEDES` Edge Detection
* **Entity Overlap + Pivot Signals:** Check if incoming text addresses existing entities while containing change-of-mind phrases (*"instead of"*, *"migrated from"*, *"deprecated"*, *"replaced with"*, *"scratch that"*).
* **Contradiction Invariant:** Link $\text{New Record} \xrightarrow{\text{SUPERSEDES}} \text{Old Record}$, mark the old record as `archived: true`, and apply exponential temporal decay ($w(t) = w_0 \cdot e^{-\lambda \Delta t}$).

### 3. Fixing Echo-Chamber Retrieval
* **Speaker Role Filtering:** Tag records with `speaker="User"` or `"Assistant"`. Down-weight or exclude user questions when performing factual knowledge retrieval.
* **Asymmetric Q&A Indexing:** Embed the question/intent for vector matching, but return the assistant's corresponding response as the retrieved payload.

### 4. Fixing Answer Truncation
* **Regex Protection:** Use negative lookbehinds `r'(?<!\b\d)(?<!\b[A-Za-z])(?<=[.!?])\s+'` so numbers (`1.`, `2.`) and abbreviations (`e.g.`) are not mistaken for sentence boundaries.
* **Chunk-Level Returns:** Return whole chunks as generated by `HybridChunker` rather than slicing them into sentence fragments.

### 5. Replacing the Linear Scan with Sub-Linear $O(\log N)$ Graph Retrieval
1. **Centroid Routing ($O(M)$):** Compare query vector against Module Centroids first; shortlist top 2–3 modules.
2. **Graph Expansion:** Follow `SHORTCUT` and `SUPERSEDES` edges from shortlisted modules via spreading activation.
3. **In-Module FAISS Search:** Query records only within the activated modules using `ModuleFaissIndex`.

### 6. Unifying the Backends
Merge the two backends into a single canonical engine:
* Keep the SQLite `VersionLedger` from `morphe.core.backend` for persistence and Git-style rollback.
* Port the `evaluate()` method and bandit action mappings from `morphe.backend.morphe_backend` into it.
* Standardize on `MutationAction` so the LinUCB Bandit directly mutates the persistent graph rendered in the Y2K Cytoscape UI.
