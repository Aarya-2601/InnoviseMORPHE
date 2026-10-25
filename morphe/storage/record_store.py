"""
Content-addressed record store implementation for MORPHE.
Features:
- SHA-256 content-addressing (record ID is SHA-256 of normalized text)
- Immutable records
- Automatic deduplication
- Metadata tracking (project, memory_type, timestamp, custom tags)
- Optional embedding vector caching (numpy float32)
- SQLite persistent metadata & object index
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from morphe.models.metadata import MemoryType, RecordMetadata


@dataclass(frozen=True)
class Record:
    """
    Immutable memory record.
    ID is the SHA-256 hex digest of the normalized text.
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, hash=False, compare=False)
    created_at: float = field(default_factory=time.time)
    archived: bool = False

    @staticmethod
    def compute_id(text: str) -> str:
        """Compute SHA-256 hash for normalized text."""
        normalized = text.strip().encode("utf-8")
        return hashlib.sha256(normalized).hexdigest()

    @classmethod
    def create(
        cls,
        text: str,
        metadata: Optional[Union[Dict[str, Any], RecordMetadata]] = None,
        embedding: Optional[Union[np.ndarray, Sequence[float]]] = None,
        created_at: Optional[float] = None,
        archived: bool = False,
    ) -> Record:
        """Factory method to construct an immutable record."""
        record_id = cls.compute_id(text)
        if isinstance(metadata, RecordMetadata):
            meta = metadata.to_dict()
        else:
            meta = RecordMetadata.from_dict(metadata).to_dict()
        
        emb_arr = None
        if embedding is not None:
            emb_arr = np.asarray(embedding, dtype=np.float32)

        return cls(
            id=record_id,
            text=text,
            metadata=meta,
            embedding=emb_arr,
            created_at=created_at if created_at is not None else time.time(),
            archived=archived,
        )

    @property
    def parsed_metadata(self) -> RecordMetadata:
        """Returns typed metadata helper."""
        return RecordMetadata.from_dict(self.metadata)

    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        """Serialize record to dictionary."""
        data: Dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "archived": self.archived,
        }
        if include_embedding and self.embedding is not None:
            data["embedding"] = self.embedding.tolist()
        return data


class RecordStore:
    """
    Content-addressed immutable record store.
    Records are stored in SQLite and optionally on-disk binary embeddings.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = ":memory:"):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite tables for records and embeddings."""
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    project TEXT NOT NULL DEFAULT 'default',
                    memory_type TEXT NOT NULL DEFAULT 'fact',
                    timestamp REAL NOT NULL,
                    embedding_blob BLOB,
                    embedding_dim INTEGER,
                    created_at REAL NOT NULL,
                    archived INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_records_project ON records(project)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_records_memory_type ON records(memory_type)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_records_timestamp ON records(timestamp)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_records_archived ON records(archived)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_records_created ON records(created_at)"
            )

    def put(
        self,
        text: str,
        metadata: Optional[Union[Dict[str, Any], RecordMetadata]] = None,
        embedding: Optional[Union[np.ndarray, Sequence[float]]] = None,
    ) -> tuple[Record, bool]:
        """
        Store a record if not already present (deduplication).
        Returns (Record, is_new).
        If already present, returns existing Record and is_new=False.
        If existing record had no embedding but a new one is provided, it updates embedding.
        """
        record_id = Record.compute_id(text)
        existing = self.get(record_id)
        
        if existing is not None:
            # Check if embedding update is needed
            if existing.embedding is None and embedding is not None:
                emb_arr = np.asarray(embedding, dtype=np.float32)
                with self.conn:
                    self.conn.execute(
                        "UPDATE records SET embedding_blob = ?, embedding_dim = ? WHERE id = ?",
                        (emb_arr.tobytes(), len(emb_arr), record_id),
                    )
                return self.get(record_id), False  # type: ignore
            return existing, False

        record = Record.create(text=text, metadata=metadata, embedding=embedding)
        emb_blob = record.embedding.tobytes() if record.embedding is not None else None
        emb_dim = len(record.embedding) if record.embedding is not None else None

        pmeta = record.parsed_metadata
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO records (
                    id, text, metadata_json, project, memory_type, timestamp,
                    embedding_blob, embedding_dim, created_at, archived
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.text,
                    json.dumps(record.metadata),
                    pmeta.project,
                    pmeta.memory_type.value,
                    pmeta.timestamp,
                    emb_blob,
                    emb_dim,
                    record.created_at,
                    1 if record.archived else 0,
                ),
            )

        return record, True

    def put_record(self, record: Record) -> tuple[Record, bool]:
        """Store an already constructed Record."""
        return self.put(
            text=record.text,
            metadata=record.metadata,
            embedding=record.embedding,
        )

    def put_batch(
        self,
        items: Sequence[Union[str, tuple[str, Optional[Dict[str, Any]]]]],
        embeddings: Optional[Sequence[Optional[Union[np.ndarray, Sequence[float]]]]] = None,
    ) -> List[tuple[Record, bool]]:
        """
        Batch insert records with automatic deduplication.
        """
        results: List[tuple[Record, bool]] = []
        for idx, item in enumerate(items):
            emb = embeddings[idx] if embeddings is not None and idx < len(embeddings) else None
            if isinstance(item, tuple):
                text, meta = item
                res = self.put(text=text, metadata=meta, embedding=emb)
            else:
                res = self.put(text=item, embedding=emb)
            results.append(res)
        return results

    def get(self, record_id: str) -> Optional[Record]:
        """Retrieve a record by its SHA-256 ID."""
        cursor = self.conn.execute(
            """
            SELECT id, text, metadata_json, embedding_blob, embedding_dim, created_at, archived
            FROM records WHERE id = ?
            """,
            (record_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        emb: Optional[np.ndarray] = None
        if row["embedding_blob"] is not None and row["embedding_dim"] is not None:
            emb = np.frombuffer(row["embedding_blob"], dtype=np.float32)

        return Record(
            id=row["id"],
            text=row["text"],
            metadata=json.loads(row["metadata_json"]),
            embedding=emb,
            created_at=row["created_at"],
            archived=bool(row["archived"]),
        )

    def get_many(self, record_ids: Sequence[str]) -> Dict[str, Record]:
        """Retrieve multiple records by IDs."""
        if not record_ids:
            return {}
        placeholders = ",".join("?" for _ in record_ids)
        cursor = self.conn.execute(
            f"""
            SELECT id, text, metadata_json, embedding_blob, embedding_dim, created_at, archived
            FROM records WHERE id IN ({placeholders})
            """,
            list(record_ids),
        )
        out: Dict[str, Record] = {}
        for row in cursor.fetchall():
            emb: Optional[np.ndarray] = None
            if row["embedding_blob"] is not None and row["embedding_dim"] is not None:
                emb = np.frombuffer(row["embedding_blob"], dtype=np.float32)
            rec = Record(
                id=row["id"],
                text=row["text"],
                metadata=json.loads(row["metadata_json"]),
                embedding=emb,
                created_at=row["created_at"],
                archived=bool(row["archived"]),
            )
            out[rec.id] = rec
        return out

    def contains(self, record_id: str) -> bool:
        """Check if a record exists."""
        cursor = self.conn.execute("SELECT 1 FROM records WHERE id = ?", (record_id,))
        return cursor.fetchone() is not None

    def set_archived(self, record_id: str, archived: bool = True) -> bool:
        """
        Set archive status (soft archive - records remain immutable in store).
        Returns True if record was found and updated.
        """
        with self.conn:
            cursor = self.conn.execute(
                "UPDATE records SET archived = ? WHERE id = ?",
                (1 if archived else 0, record_id),
            )
            return cursor.rowcount > 0

    def list_all_ids(self, include_archived: bool = True) -> List[str]:
        """List all stored record IDs."""
        if include_archived:
            cursor = self.conn.execute("SELECT id FROM records ORDER BY created_at ASC")
        else:
            cursor = self.conn.execute("SELECT id FROM records WHERE archived = 0 ORDER BY created_at ASC")
        return [row[0] for row in cursor.fetchall()]

    def count(self, include_archived: bool = True) -> int:
        """Count total records."""
        if include_archived:
            cursor = self.conn.execute("SELECT COUNT(*) FROM records")
        else:
            cursor = self.conn.execute("SELECT COUNT(*) FROM records WHERE archived = 0")
        return cursor.fetchone()[0]

    def find_by_metadata(
        self,
        project: Optional[str] = None,
        memory_type: Optional[Union[str, MemoryType]] = None,
        since_timestamp: Optional[float] = None,
        until_timestamp: Optional[float] = None,
        include_archived: bool = False,
    ) -> List[Record]:
        """
        Query records filtered by indexed metadata fields.
        """
        query = "SELECT id, text, metadata_json, embedding_blob, embedding_dim, created_at, archived FROM records WHERE 1=1"
        params: List[Any] = []

        if not include_archived:
            query += " AND archived = 0"
        if project is not None:
            query += " AND project = ?"
            params.append(project)
        if memory_type is not None:
            mtype_val = memory_type.value if isinstance(memory_type, MemoryType) else str(memory_type)
            query += " AND memory_type = ?"
            params.append(mtype_val)
        if since_timestamp is not None:
            query += " AND timestamp >= ?"
            params.append(since_timestamp)
        if until_timestamp is not None:
            query += " AND timestamp <= ?"
            params.append(until_timestamp)

        query += " ORDER BY timestamp ASC"

        cursor = self.conn.execute(query, params)
        records: List[Record] = []
        for row in cursor.fetchall():
            emb: Optional[np.ndarray] = None
            if row["embedding_blob"] is not None and row["embedding_dim"] is not None:
                emb = np.frombuffer(row["embedding_blob"], dtype=np.float32)
            records.append(
                Record(
                    id=row["id"],
                    text=row["text"],
                    metadata=json.loads(row["metadata_json"]),
                    embedding=emb,
                    created_at=row["created_at"],
                    archived=bool(row["archived"]),
                )
            )
        return records

    def close(self) -> None:
        """Close SQLite connection."""
        self.conn.close()

