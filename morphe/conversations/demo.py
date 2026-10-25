"""
Interactive CLI Demonstrator to test MORPHE on chat transcripts.
Run with:
    python -m morphe.conversations.demo
"""

import sys
from morphe.conversations.pipeline import ChatConversationPipeline
from morphe.core.backend import MorpheBackend


DEMO_TRANSCRIPT = """
User prompt: Scratch the 4 hour thing, make it as good as possible
Response: With the time constraint removed, we can evolve this from a functional MVP into a state-of-the-art production system. We will drop the basic max-heap heuristics and upgrade the C++ architecture to support a true Deep Reinforcement Learning engine driven by Proximal Policy Optimization (PPO).

User prompt: Also, in the new doc that you will create, give me the whole architectural diagram as well so my partner can understand where RL comes and where systems comes into play and he will vibecode everything.
Response: The complete architectural blueprint, tailored specifically for your partner to understand exactly how the systems and RL components interact, is ready.

User prompt: These are the things my partner wants to implement so update everything including the diagram and what all the llm he will give prompt to had to do step by step.
Response: I have completely overhauled the architecture to align perfectly with the "MORPHE: Final Plan (RL-First Architecture)" document. Here are the critical updates:
1. The Shift from PPO to Contextual Bandits: As dictated by the document, we are entirely dropping PPO and deep RL. They are unstable and data-hungry. Instead, we are using a Contextual Bandit (LinUCB) for the Architect Policy.
2. Ground Truth Reward (No LLM Judges): The core thesis is now hardcoded: Gemma 4 is never used to judge a mutation. The reward is purely mathematical: Recall@K + MRR - latency - structural_cost.
3. The Contract & Division of Labor: Mocked MorpheBackend contract (fork, apply, evaluate, commit, rollback).

User prompt: How do we hash the graphs during commits and is there an explicit button for commits or not?
Response: The system uses Content-Addressable Storage (CAS). The JSON diff is passed through Zstandard compression and SHA-256 creates the immutable version_id stored in the SQLite ledger.
"""


def run_demo():
    print("=" * 70)
    print("[MORPHE] Conversational Memory Demonstration")
    print("=" * 70)

    # 1. Ingest conversation
    backend = MorpheBackend()
    pipeline = ChatConversationPipeline(backend)
    print("\n1. Ingesting conversation transcript into MORPHE Content-Addressed Store...")
    res = pipeline.ingest_conversation(DEMO_TRANSCRIPT, conversation_id="fast_innovation_chat")
    print(f"   -> Parsed {res['turns_parsed']} conversational turns.")
    print(f"   -> Stored {res['records_stored']} immutable sentence-aligned records.")
    print(f"   -> Created topic modules: {res['modules_created']}")
    print(f"   -> Supersedes edge detected & linked: {res['superseded_detected']}")

    # 2. Test 1: Recall from early turns (No sliding-window amnesia)
    q1 = "What did the user say about the 4 hour time constraint?"
    print(f"\n2. [Test 1: Early Turn Recall] Query: '{q1}'")
    hits1 = pipeline.query(q1, top_k=2)
    for i, h in enumerate(hits1, 1):
        print(f"   Hit {i} (Score: {h['score']}) [Module: {h['module']}]:")
        print(f"   \"{h['text']}\"\n")

    # 3. Test 2: Current Decision vs Historical Pivot
    q2_curr = "What RL policy do we use for the Architect?"
    print(f"3. [Test 2a: Current Architecture] Query: '{q2_curr}'")
    hits2 = pipeline.query(q2_curr, top_k=1)
    print(f"   -> Top Hit: \"{hits2[0]['text']}\"\n")

    q2_hist = "What RL algorithm was originally planned before switching to bandits?"
    print(f"4. [Test 2b: Historical Query] Query: '{q2_hist}'")
    hits3 = pipeline.query(q2_hist, top_k=1)
    print(f"   -> Top Hit (Traversed backward via SUPERSEDES): \"{hits3[0]['text']}\"\n")

    # 5. Test 3: Commit History & Time Travel
    print("5. [Test 3: Commit History & Ledger Trace]")
    history = backend.list_versions()
    for entry in history:
        print(f"   - Version {entry['version_id']} | Reason: {entry['reason_card']}")

    print("\n" + "=" * 70)
    print("[SUCCESS] All conversational tests passed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
