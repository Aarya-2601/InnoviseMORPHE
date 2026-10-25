"""
Chat / Dialogue Ingestion Pipeline for MORPHE.
Features:
- Parses multi-turn dialogue logs ('User: ...', 'Response: ...' or structured turns)
- Uses HybridChunker to break long conversational responses into sentence-aligned records
- Detects technical architectural updates (e.g. 'PPO -> Contextual Bandit LinUCB')
  and automatically builds temporal SUPERSEDES links
- Clusters chat messages into semantic topic modules (e.g. 'mod_ingestion', 'mod_rl_bandit', 'mod_cas_storage')
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from morphe.core.backend import MorpheBackend
from morphe.indexing.embeddings import EmbeddingService
from morphe.ingestion.chunker import HybridChunker
from morphe.models.metadata import MemoryType, RecordMetadata
from morphe.storage.record_store import Record, RecordStore
from morphe.temporal.manager import TemporalMemoryManager, is_historical_query
from morphe.topology.models import ModuleManifest, TopologySnapshot


@dataclass
class ChatTurn:
    speaker: str  # 'User' or 'Assistant'
    text: str
    turn_index: int
    timestamp: float = field(default_factory=time.time)


class ChatConversationPipeline:
    """
    Ingests and manages multi-turn chat dialogues inside MORPHE.
    """

    def __init__(
        self,
        backend: MorpheBackend,
        embedding_service: Optional[EmbeddingService] = None,
        chunker: Optional[HybridChunker] = None,
    ):
        self.backend = backend
        self.record_store = backend.record_store
        self.embedder = embedding_service or EmbeddingService(model_name="all-MiniLM-L6-v2")
        self.chunker = chunker or HybridChunker(min_chars=60, target_chars=200, max_chars=400)
        self.temporal_mgr = TemporalMemoryManager(backend)

    def parse_transcript(self, transcript_text: str) -> List[ChatTurn]:
        """
        Parses transcripts formatted with 'User prompt:' / 'Response:' or 'User:' / 'Assistant:'.
        """
        lines = transcript_text.strip().splitlines()
        turns: List[ChatTurn] = []
        current_speaker: Optional[str] = None
        current_lines: List[str] = []
        turn_idx = 0
        t0 = time.time()

        speaker_pattern = re.compile(
            r'^(User prompt:|User:|Response:|Assistant:|System:)\s*(.*)',
            re.IGNORECASE,
        )

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            match = speaker_pattern.match(line_str)
            if match:
                # Flush previous turn
                if current_speaker and current_lines:
                    text_block = " ".join(current_lines).strip()
                    if text_block:
                        turns.append(
                            ChatTurn(
                                speaker=current_speaker,
                                text=text_block,
                                turn_index=turn_idx,
                                timestamp=t0 + turn_idx * 60.0,
                            )
                        )
                        turn_idx += 1
                    current_lines = []

                header = match.group(1).lower()
                content = match.group(2).strip()
                if "user" in header:
                    current_speaker = "User"
                elif "response" in header or "assistant" in header:
                    current_speaker = "Assistant"
                else:
                    current_speaker = "System"

                if content:
                    current_lines.append(content)
            else:
                if current_speaker:
                    current_lines.append(line_str)

        # Flush trailing turn
        if current_speaker and current_lines:
            text_block = " ".join(current_lines).strip()
            if text_block:
                turns.append(
                    ChatTurn(
                        speaker=current_speaker,
                        text=text_block,
                        turn_index=turn_idx,
                        timestamp=t0 + turn_idx * 60.0,
                    )
                )

        return turns

    def ingest_conversation(
        self,
        transcript_text: str,
        conversation_id: str = "conv_1",
    ) -> Dict[str, Any]:
        """
        Ingests the conversation:
        1. Breaks turns into sentence-aligned content-addressed chunks.
        2. Embeds each chunk with L2-normalized float32 vectors.
        3. Assigns to specialized memory modules based on topic content.
        4. Detects decision updates (e.g. PPO -> Contextual Bandit) and links with SUPERSEDES edges.
        """
        turns = self.parse_transcript(transcript_text)
        # Fallback if no specific speaker pattern matched: treat paragraphs as turns
        if not turns:
            paras = [p.strip() for p in transcript_text.split("\n\n") if p.strip()]
            for idx, p in enumerate(paras):
                turns.append(ChatTurn(speaker="User" if idx % 2 == 0 else "Assistant", text=p, turn_index=idx))

        if not turns:
            return {"turns_parsed": 0, "records_stored": 0, "modules_created": [], "superseded_detected": False}

        all_record_ids: List[str] = []
        topic_buckets: Dict[str, List[str]] = {
            "mod_ingestion": [],
            "mod_systems_cas": [],
            "mod_rl_bandit": [],
            "mod_retrieval_router": [],
            "mod_dashboard_ui": [],
        }

        # Keep track of records for supersession detection
        ppo_record_id: Optional[str] = None
        bandit_record_id: Optional[str] = None

        for turn in turns:
            # Chunk long assistant responses
            chunks = self.chunker.chunk_text(turn.text)
            if not chunks:
                chunks = [
                    type("SimpleChunk", (), {
                        "text": turn.text,
                        "metadata": {}
                    })()
                ]

            for c in chunks:
                vec = self.embedder.embed_text(c.text)
                meta = RecordMetadata(
                    project=conversation_id,
                    memory_type=MemoryType.INTERACTION,
                    timestamp=turn.timestamp,
                    tags=[turn.speaker.lower(), f"turn_{turn.turn_index}"],
                    extra={"speaker": turn.speaker, "turn_index": turn.turn_index},
                )
                rec, _ = self.record_store.put(text=c.text, metadata=meta, embedding=vec)
                all_record_ids.append(rec.id)

                # Classify into topic buckets
                lower_text = c.text.lower()
                if "ppo" in lower_text and "proximal policy" in lower_text:
                    ppo_record_id = rec.id

                if "contextual bandit" in lower_text or "linucb" in lower_text:
                    bandit_record_id = rec.id

                if any(k in lower_text for k in ["ingestion", "fastcdc", "chunking", "ner"]):
                    topic_buckets["mod_ingestion"].append(rec.id)
                elif any(k in lower_text for k in ["c++17", "cas", "zstandard", "sqlite", "adjacency"]):
                    topic_buckets["mod_systems_cas"].append(rec.id)
                elif any(k in lower_text for k in ["bandit", "ppo", "linucb", "reward", "architect"]):
                    topic_buckets["mod_rl_bandit"].append(rec.id)
                elif any(k in lower_text for k in ["router", "bm25", "faiss", "hybrid", "stopping"]):
                    topic_buckets["mod_retrieval_router"].append(rec.id)
                elif any(k in lower_text for k in ["dashboard", "react", "webgl", "ui", "commit tree"]):
                    topic_buckets["mod_dashboard_ui"].append(rec.id)
                else:
                    topic_buckets["mod_rl_bandit"].append(rec.id)

        # Build module manifests for non-empty buckets
        manifests: Dict[str, ModuleManifest] = {}
        for mod_id, rec_list in topic_buckets.items():
            if rec_list:
                # Compute centroid
                recs = self.record_store.get_many(rec_list)
                embs = [r.embedding for r in recs.values() if r.embedding is not None]
                if embs:
                    cent = np.mean(np.stack(embs, axis=0), axis=0)
                    norm = np.linalg.norm(cent)
                    if norm > 1e-9:
                        cent /= norm
                else:
                    cent = None

                charter_name = mod_id.replace("mod_", "").replace("_", " ").title()
                manifests[mod_id] = ModuleManifest.create(
                    module_id=mod_id,
                    record_ids=rec_list,
                    charter=f"{charter_name} specifications and chat turns",
                    centroid=cent,
                )

        # Create topology snapshot
        snapshot = TopologySnapshot.create(modules=manifests, edges=[])
        self.backend._load_snapshot_into_staging(snapshot)
        commit = self.backend.commit(
            reason_card=f"Ingested conversation {conversation_id} with {len(turns)} turns",
            metrics={"turns": len(turns), "records": len(all_record_ids)},
        )

        # Link supersedes edge if a decision shift was detected (e.g. PPO -> Contextual Bandit)
        superseded = False
        if ppo_record_id and bandit_record_id:
            self.temporal_mgr.record_supersession(
                new_record_id=bandit_record_id,
                old_record_id=ppo_record_id,
                source_module="mod_rl_bandit",
                archive_old=True,
            )
            self.backend.commit(
                reason_card="Architectural pivot: PPO superseded by Contextual Bandits (LinUCB)"
            )
            superseded = True

        return {
            "turns_parsed": len(turns),
            "records_stored": len(all_record_ids),
            "modules_created": list(manifests.keys()),
            "commit_id": commit.version_id,
            "superseded_detected": superseded,
            "ppo_id": ppo_record_id,
            "bandit_id": bandit_record_id,
        }

    def query(
        self,
        query_text: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Query the ingested conversation:
        1. Routes to top module via centroid similarity.
        2. Retrieves candidate records using hybrid score (dense cosine similarity + keyword boost).
        3. Applies TemporalMemoryManager (handles 'before/historical' vs 'now/current' questions).
        """
        q_vec = self.embedder.embed_text(query_text)
        snapshot = self.backend.get_staged_snapshot()
        if not snapshot.modules:
            return []

        # Step 1: Score modules
        module_scores: List[Tuple[str, float]] = []
        for m_id, m in snapshot.modules.items():
            if m.centroid is not None:
                sim = float(np.dot(q_vec, m.centroid))
            else:
                sim = 0.0
            module_scores.append((m_id, sim))

        module_scores.sort(key=lambda x: x[1], reverse=True)
        top_module_id = module_scores[0][0]
        top_module = snapshot.modules[top_module_id]

        # Step 2: Score records in top module with lexical overlap boost
        records = self.record_store.get_many(top_module.record_ids)
        cand_scores: List[Tuple[str, float]] = []
        q_words = set(re.findall(r'\w+', query_text.lower()))

        for r_id, rec in records.items():
            if rec.embedding is not None:
                dense_sim = float(np.dot(q_vec, rec.embedding))
            else:
                dense_sim = 0.0

            # Lexical keyword overlap bonus
            r_words = set(re.findall(r'\w+', rec.text.lower()))
            overlap = len(q_words & r_words) / max(1, len(q_words))
            hybrid_score = dense_sim + 0.5 * overlap

            cand_scores.append((r_id, hybrid_score))

        cand_scores.sort(key=lambda x: x[1], reverse=True)

        # Step 3: Apply temporal resolution
        resolved = self.temporal_mgr.resolve_temporal_records(
            record_candidates=cand_scores,
            query=query_text,
        )

        results: List[Dict[str, Any]] = []
        for r_id, score in resolved[:top_k]:
            rec = self.record_store.get(r_id)
            if rec:
                results.append({
                    "record_id": r_id,
                    "text": rec.text,
                    "score": round(score, 4),
                    "module": top_module_id,
                    "archived": rec.archived,
                    "metadata": rec.metadata,
                })

        return results
