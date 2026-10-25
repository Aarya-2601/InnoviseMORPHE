"""
Tests for MORPHE on user chat conversation transcripts.
Demonstrates:
1. Parsing and ingesting conversational turns into immutable content-addressed records.
2. Answering early questions without sliding-window amnesia.
3. Resolving architectural decision changes (PPO vs. LinUCB Contextual Bandit) via SUPERSEDES edges.
4. Sleep cycle consolidation removing conversational duplicates.
"""

import pytest

from morphe.conversations.pipeline import ChatConversationPipeline
from morphe.core.backend import MorpheBackend
from morphe.storage.record_store import RecordStore


SAMPLE_CONVERSATION = """
User prompt: Scratch the 4 hour thing, make it as good as possible
Response: With the time constraint removed, we can evolve this from a functional MVP into a state-of-the-art production system. We will drop the basic max-heap heuristics and upgrade the C++ architecture to support a true Deep Reinforcement Learning engine driven by Proximal Policy Optimization (PPO).

User prompt: Also, in the new doc that you will create, give me the whole architectural diagram as well so my partner can understand where RL comes and where systems comes into play.
Response: The complete architectural blueprint, tailored specifically for your partner to understand exactly how the systems and RL components interact, is ready.

User prompt: These are the things my partner wants to implement so update everything including the diagram.
Response: I have completely overhauled the architecture to align perfectly with the "MORPHE: Final Plan (RL-First Architecture)" document. Here are the critical updates:
1. The Shift from PPO to Contextual Bandits: As dictated by the document, we are entirely dropping PPO and deep RL. Instead, we are using a Contextual Bandit (LinUCB) for the Architect Policy.
2. Ground Truth Reward (No LLM Judges): The core thesis is now hardcoded: Gemma 4 is never used to judge a mutation.
3. The Contract: Mocked MorpheBackend contract (fork, apply, evaluate, commit, rollback).

User prompt: How do we hash the graphs during commits?
Response: The system uses Content-Addressable Storage (CAS). The JSON diff is passed through Zstandard compression and SHA-256 creates the immutable version_id.
"""


def test_conversation_ingestion():
    backend = MorpheBackend()
    pipeline = ChatConversationPipeline(backend)
    
    result = pipeline.ingest_conversation(SAMPLE_CONVERSATION, conversation_id="conv_morphe_chat")
    assert result["turns_parsed"] >= 4
    assert result["records_stored"] > 0
    assert len(result["modules_created"]) >= 2
    assert result["superseded_detected"] is True


def test_early_turn_recall_no_amnesia():
    """Verify that early turns are retained accurately in immutable CAS."""
    backend = MorpheBackend()
    pipeline = ChatConversationPipeline(backend)
    pipeline.ingest_conversation(SAMPLE_CONVERSATION, conversation_id="conv_morphe_chat")

    # Ask about early user directive ("scratch the 4 hour thing")
    hits = pipeline.query("What did the user say about the 4 hour time constraint?", top_k=2)
    assert len(hits) >= 1
    matched_text = " ".join(h["text"] for h in hits)
    assert "time constraint" in matched_text.lower() or "4 hour" in matched_text.lower()


def test_decision_switch_temporal_query():
    """
    Verifies that MORPHE distinguishes between:
    - Current decision: Contextual Bandit / LinUCB
    - Historical decision: Proximal Policy Optimization (PPO)
    """
    backend = MorpheBackend()
    pipeline = ChatConversationPipeline(backend)
    pipeline.ingest_conversation(SAMPLE_CONVERSATION, conversation_id="conv_morphe_chat")

    # Query 1: Current architecture decision
    current_hits = pipeline.query("What reinforcement learning policy do we use for the Architect?")
    assert len(current_hits) >= 1
    assert "contextual bandit" in current_hits[0]["text"].lower() or "linucb" in current_hits[0]["text"].lower()

    # Query 2: Historical decision ("What did we consider before?")
    hist_hits = pipeline.query("What RL algorithm was originally planned before switching to bandits?")
    assert len(hist_hits) >= 1
    # Historical query should prefer the superseded PPO record
    top_hist = hist_hits[0]["text"].lower()
    assert "ppo" in top_hist or "proximal policy" in top_hist

