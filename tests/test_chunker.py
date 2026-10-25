"""
Tests for Hybrid Sentence-Aligned Content-Defined Chunking.
"""

from morphe.ingestion.chunker import HybridChunker, sentence_split


def test_sentence_split():
    text = (
        "MORPHE uses content-defined chunking. It prevents boundary shifts across entire documents! "
        "Does it preserve semantic readability? Absolutely. "
        "\n\nParagraph two starts here. It also contains two sentences."
    )
    sentences = sentence_split(text)
    assert len(sentences) == 6
    assert sentences[0] == "MORPHE uses content-defined chunking."
    assert sentences[1] == "It prevents boundary shifts across entire documents!"
    assert sentences[2] == "Does it preserve semantic readability?"
    assert sentences[3] == "Absolutely."
    assert sentences[4] == "Paragraph two starts here."
    assert sentences[5] == "It also contains two sentences."


def test_sentence_boundary_preservation():
    """Verify that chunks never cut mid-sentence."""
    chunker = HybridChunker(min_chars=50, target_chars=120, max_chars=200)
    
    text = (
        "First sentence sets the context of the architecture. "
        "Second sentence explains the router policy with entropy and margin. "
        "Third sentence details the architect bandit LinUCB with two levels. "
        "Fourth sentence reviews the value of information trial scheduler. "
        "Fifth sentence covers three-factor plasticity with verified reward."
    )
    
    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 2
    
    # Check that each chunk ends with proper sentence punctuation
    for c in chunks:
        assert c.text[-1] in ".!?"
        assert c.char_len >= 50 or c == chunks[-1]


def test_content_addressing_stability_and_local_edit():
    """
    Key MORPHE invariant:
    When a document is edited in the middle, unchanged sections retain
    identical chunks with identical SHA-256 hashes!
    """
    chunker = HybridChunker(min_chars=60, target_chars=150, max_chars=250)

    doc_v1 = (
        "Module A handles authentication protocols and user session keys. "
        "It maintains token lifespans and refresh rotation policies. "
        "Module B manages database schemas with PostgreSQL and migration scripts. "
        "It runs table vacuuming and connection pooling automatically. "
        "Module C handles analytical pipelines and telemetry aggregation. "
        "It summarizes query latency and compute resource consumption."
    )

    # In version 2, ONLY Module B is modified ("PostgreSQL" -> "MongoDB")
    doc_v2 = (
        "Module A handles authentication protocols and user session keys. "
        "It maintains token lifespans and refresh rotation policies. "
        "Module B manages database schemas with MongoDB and document migrations. "
        "It coordinates replica sets and sharding rebalancing automatically. "
        "Module C handles analytical pipelines and telemetry aggregation. "
        "It summarizes query latency and compute resource consumption."
    )

    chunks_v1 = chunker.chunk_text(doc_v1)
    chunks_v2 = chunker.chunk_text(doc_v2)

    ids_v1 = [c.id for c in chunks_v1]
    ids_v2 = [c.id for c in chunks_v2]

    # Module A chunk and Module C chunk should be IDENTICAL
    assert ids_v1[0] == ids_v2[0]
    assert chunks_v1[0].text == chunks_v2[0].text

    assert ids_v1[-1] == ids_v2[-1]
    assert chunks_v1[-1].text == chunks_v2[-1].text

    # The middle chunk (Module B) must be different
    assert ids_v1[1] != ids_v2[1]


def test_oversized_sentence_fallback():
    """Check handling when a single sentence is longer than min_chars."""
    chunker = HybridChunker(min_chars=30, target_chars=60, max_chars=100)
    text = "This is a single very long sentence that has significantly more words than the minimum chunk length requirement."
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].char_len == len(text)
