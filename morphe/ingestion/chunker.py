"""
Sentence-aligned Content-Defined Chunking (Hybrid Chunking).
Plan specification:
1. Split text into sentences.
2. Run FastCDC / Gear rolling-hash cut points, but allow a cut ONLY at a sentence boundary,
   so chunks stay readable, semantically coherent, and embed well.
3. Compute SHA-256 for each chunk to produce deterministic content addresses.
4. Stable boundaries: edits only shift boundaries locally near the edit.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


def sentence_split(text: str) -> List[str]:
    """
    Split text into clean, trimmed sentences preserving readability.
    Handles standard punctuation (. ! ?) with abbreviations/decimals avoided.
    """
    if not text or not text.strip():
        return []

    # First split on multiline / paragraph breaks to preserve paragraphs
    paragraphs = re.split(r'\n{2,}', text.strip())
    sentences: List[str] = []

    # Regex pattern to split sentences inside each paragraph
    pattern = r'(?<=[.!?])\s+(?=[A-Z0-9"\'])'

    for para in paragraphs:
        para_clean = para.strip()
        if not para_clean:
            continue
        raw_parts = re.split(pattern, para_clean)
        for part in raw_parts:
            part_clean = part.strip()
            if part_clean:
                sentences.append(part_clean)
    return sentences


# 256-entry pseudo-random Gear table for rolling hash
GEAR_TABLE = [
    0x8a91428a, 0xa385e17a, 0xbe51b2e0, 0x86333b35, 0x5b60ef44, 0xb074a3bc, 0x05fb6a51, 0xb9a63a24,
    0x10cbd634, 0x221130f4, 0xd31a4ee5, 0x80a17f47, 0x07ab29ee, 0x531ff55b, 0x488e0524, 0x4c402399,
    0xc58c351e, 0x3d4f005b, 0x98855e23, 0xb336f00d, 0x2741472c, 0x42aa0ade, 0x8a59f728, 0xec60b96c,
    0x70bc9f48, 0x738449ba, 0x5acc1dff, 0x3f60b051, 0xae301fcf, 0xf7e434f4, 0x76bb387b, 0xd30e4e70,
    0x4a9e3b4d, 0xe96f0bbf, 0x9b9b7bfb, 0x83151ede, 0x6154e570, 0x027b14d3, 0x0f282411, 0xf2a3b499,
    0x4ecf8047, 0x21b01915, 0x708b3054, 0xc4412211, 0x98340c4a, 0x43187a32, 0x4c865747, 0x64a6b146,
    0x76c4950d, 0x9d4a2f93, 0x439c317b, 0x5329f52d, 0xba08b15e, 0x00aee157, 0x8382e36c, 0x6f57b30c,
    0x2a2977d2, 0x3ac0766e, 0x99480766, 0x9ce88044, 0x11e0a4d9, 0x51e1536f, 0x34e479a4, 0x64e9e30d,
    0xccf12bf2, 0x06441e8f, 0x49ee7224, 0x578f4e22, 0x4182a577, 0x86f3e080, 0x7763ced1, 0x2223db5e,
    0xf510037a, 0xbebad45c, 0xa11764b3, 0x2be50a30, 0xcdb4e6d6, 0xb34728ca, 0x7b654433, 0x7376300d,
    0xcc3d3372, 0x619cd3e4, 0x9bbd8541, 0x22878e61, 0x72b65870, 0xded064b8, 0xb6110052, 0x60e5283b,
    0xec516579, 0xa7907f12, 0x60509801, 0xd3e58292, 0xd20b7214, 0xaa26e184, 0x7047b11b, 0x40ec34d0,
    0x79f44a36, 0x42134d12, 0x4c0e00a0, 0x7be239db, 0x44814779, 0x808d52cf, 0xab4501b2, 0x545742f1,
    0x4e7a6ce8, 0x642260e1, 0xf6e08878, 0x76fe0b66, 0x7a2d216b, 0x6b2477f0, 0x6ed3256a, 0xb45766ac,
    0x263a4295, 0x2e63cb28, 0x70094707, 0xec164b35, 0x99765299, 0x708319a2, 0x2b8a4f6e, 0x5a516637,
    0x72f4435e, 0x5fb3a81a, 0x8142aa77, 0x47720979, 0x9ce7b364, 0x42e9bc29, 0x464e0225, 0x833138cc,
    0x662a61e3, 0x303567e1, 0x521fb634, 0xb99b72f8, 0x7aec33ce, 0x859392e0, 0x72922e0e, 0x02c0fe0b,
    0xa0c7d719, 0xa1d7ee3d, 0x367d0230, 0x41930d60, 0xd473dc91, 0x488b6e29, 0x808c3d9e, 0xdf2d4f87,
    0x38515229, 0xd63224c2, 0x85fc4e1b, 0xb9932697, 0x409b9171, 0x7058b7a4, 0x52b21b76, 0xd45a91ac,
    0xf2e4b72d, 0x3a10e6e8, 0xf7380b28, 0x84d69124, 0x4cce64ec, 0x1bae46ec, 0xac60e36e, 0xb6c02a30,
    0xbe51622b, 0x64450e3b, 0xfec64150, 0x5059738b, 0x645656ac, 0x3caa4704, 0x47430034, 0x01100888,
    0xb88b806a, 0xa91b3370, 0x679442b3, 0x24869be3, 0x315077e6, 0x302e4d04, 0x2b11485c, 0x60805777,
    0xb59e0bfa, 0xb32f4512, 0xac24e8c2, 0x37be21a7, 0x45b4ee90, 0xc0ed3774, 0x8400a409, 0x144b126d,
    0xc6729514, 0x8b044514, 0xdf444830, 0xe8da1ffc, 0xcb2da77d, 0xf097c32e, 0xa1130472, 0x52242223,
    0x03210012, 0x33e70244, 0x147900b6, 0x90220119, 0xb9ac1bcf, 0xac7df999, 0x2ba45e22, 0x3ac55b08,
    0x2735da02, 0xca24ff22, 0xad4a4349, 0x2fe3b049, 0x754142b4, 0xa5cd16e1, 0x4e238e05, 0x8b1236c4,
    0x956508e9, 0x9840eb24, 0x16b4472b, 0x729fe029, 0x8ac284de, 0x341b8504, 0x37a85640, 0x37295bfa,
    0x4ac60008, 0xbc0b8fec, 0xb7a3aa3c, 0x83b51baf, 0x21649250, 0x80036660, 0x9c6420e6, 0x5fe54085,
    0x43825555, 0xd0bd4c12, 0x2288a320, 0x7507fc58, 0x4ce4460c, 0xa5ae3b54, 0x73241b9c, 0x344b30ae,
    0x49756621, 0x0e997b66, 0x22b0ba99, 0x440e43e2, 0x01122334, 0xe4130095, 0x82b49e12, 0xd6b3ac10,
    0x4b1e5212, 0xd3ac3411, 0xa1e7d825, 0x2ac54497, 0xb79c0250, 0x86400a42, 0x421b4432, 0x4770114f,
    0xecb5109b, 0xa9235e01, 0x40b30114, 0x342e4726, 0xa905b220, 0x22304910, 0x59b1524d, 0xeb642011
]


@dataclass
class Chunk:
    """Representation of an ingested, sentence-aligned chunk."""
    id: str
    text: str
    sentence_count: int
    char_len: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class HybridChunker:
    """
    Hybrid Sentence-Aligned FastCDC Chunker.
    Combines:
    1. Sentence boundaries (avoids splitting mid-sentence or mid-phrase).
    2. Content-defined rolling hash cut condition.
    3. Min and Max length limits.
    """

    def __init__(
        self,
        min_chars: int = 100,
        target_chars: int = 400,
        max_chars: int = 800,
        mask_bits: int = 2,  # mask for cut condition on sentence boundary
        proposition_fn: Optional[Callable[[str], List[str]]] = None,
    ):
        self.min_chars = min_chars
        self.target_chars = target_chars
        self.max_chars = max_chars
        self.mask = (1 << mask_bits) - 1
        self.proposition_fn = proposition_fn

    @staticmethod
    def _rolling_hash(bytes_data: bytes) -> int:
        """Fast Gear rolling hash over byte buffer."""
        h = 0
        for b in bytes_data:
            h = ((h << 1) + GEAR_TABLE[b]) & 0xFFFFFFFF
        return h

    def is_cut_point(self, sentence: str) -> int:
        """
        Check if the sentence boundary qualifies as a content-defined cut point.
        Evaluated on the trailing byte sequence of the sentence.
        Returns the hash value.
        """
        trailing_bytes = sentence[-64:].encode("utf-8")
        h = self._rolling_hash(trailing_bytes)
        return (h & self.mask) == 0 or (h & 0xF) == 5

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Split document into sentence-aligned, content-defined chunks.
        """
        if not text or not text.strip():
            return []

        base_meta = dict(metadata or {})
        sentences = sentence_split(text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        current_sentences: List[str] = []
        current_len = 0

        for sentence in sentences:
            # If a single sentence exceeds max_chars, check if proposition_fn (Gemma) can split it
            if len(sentence) > self.max_chars:
                if self.proposition_fn is not None:
                    try:
                        props = self.proposition_fn(sentence)
                    except Exception:
                        props = [sentence]
                else:
                    props = [sentence]
            else:
                props = [sentence]

            for s in props:
                s_len = len(s)
                # If adding this sentence would strictly exceed max_chars, flush current chunk
                if current_sentences and (current_len + s_len > self.max_chars):
                    chunk_text = " ".join(current_sentences)
                    chunk_id = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            text=chunk_text,
                            sentence_count=len(current_sentences),
                            char_len=len(chunk_text),
                            metadata=dict(base_meta),
                        )
                    )
                    current_sentences = []
                    current_len = 0

                current_sentences.append(s)
                current_len += s_len + 1  # count space

                # Content-defined boundary check on sentence boundary
                if current_len >= self.min_chars:
                    if self.is_cut_point(s) or current_len >= self.target_chars:
                        chunk_text = " ".join(current_sentences)
                        chunk_id = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                        chunks.append(
                            Chunk(
                                id=chunk_id,
                                text=chunk_text,
                                sentence_count=len(current_sentences),
                                char_len=len(chunk_text),
                                metadata=dict(base_meta),
                            )
                        )
                        current_sentences = []
                        current_len = 0

        # Flush any trailing sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk_id = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    sentence_count=len(current_sentences),
                    char_len=len(chunk_text),
                    metadata=dict(base_meta),
                )
            )

        return chunks
