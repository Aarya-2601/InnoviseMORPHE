"""
Unit tests for LLMClient caching, JSON parsing, and fallbacks.
"""

from morphe.llm.client import LLMClient


def test_llm_client_fallback_mode():
    client = LLMClient(force_fallback=True)
    probes = client.generate_probes("Test technical specification text", n=2)
    assert len(probes) >= 1
    assert isinstance(probes[0], str)

    charter = client.generate_charter("DatabaseModule", ["Postgres replication", "WAL logs"])
    assert "DatabaseModule" in charter or len(charter) > 10

    reason = client.write_reason_card("split", ["mod_auth"], {"recall_at_5": 0.05})
    assert len(reason) > 10


def test_llm_client_caching():
    client = LLMClient(force_fallback=True, enable_cache=True)
    res1 = client.generate("test prompt")
    res2 = client.generate("test prompt")
    assert res1 == res2
    assert len(client._cache) == 1
