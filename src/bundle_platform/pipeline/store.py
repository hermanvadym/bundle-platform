# pipeline/store.py
"""
Qdrant vector store for diagent RAG pipeline.

Wraps qdrant-client with a simple interface suited for log chunk storage.
Supports in-memory (for tests) and on-disk (for persistent session caches).

Collection schema per chunk:
  vector:   float[384]  — BAAI/bge-small-en-v1.5 embedding
  payload:
    file_path:        str
    category:         str
    start_line:       int
    end_line:         int
    text:             str
    severity:         str | None
    bundle_type:      str       ("rhel" or "esxi")
    timestamp_start:  float | None  (Unix epoch of first log line in chunk)
    timestamp_end:    float | None  (Unix epoch of last log line in chunk)
"""

from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from bundle_platform.pipeline.exceptions import RagUnavailable

# Embedding dimension for BAAI/bge-small-en-v1.5
VECTOR_SIZE = 384
_COLLECTION = "sos_chunks"


class VectorStore:
    """Thin wrapper around QdrantClient for chunk storage and retrieval."""

    def __init__(self, client: QdrantClient) -> None:
        self._client = client
        self._ensure_collection()

    @classmethod
    def in_memory(cls) -> "VectorStore":
        """Create a transient in-memory store (for tests)."""
        return cls(QdrantClient(":memory:"))

    @classmethod
    def from_path(cls, path: Path) -> "VectorStore":
        """Create or open a persistent on-disk store at the given directory."""
        path.mkdir(parents=True, exist_ok=True)
        try:
            return cls(QdrantClient(path=str(path)))
        except Exception as exc:
            raise RagUnavailable(
                f"vector store unavailable at {path}: {exc}"
            ) from exc

    def upsert(self, vectors: list[list[float]], payloads: list[dict]) -> None:
        """Insert or replace chunk vectors. IDs are derived from position hash."""
        points = [
            PointStruct(
                id=abs(hash((p["file_path"], p["start_line"]))) % (2**63),
                vector=v,
                payload=p,
            )
            for v, p in zip(vectors, payloads, strict=True)
        ]
        self._client.upsert(collection_name=_COLLECTION, points=points)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        category: str | None = None,
        severity: str | None = None,
        time_window: tuple[datetime | None, datetime | None] | None = None,
    ) -> list[dict]:
        """
        Return top_k chunks closest to query_vector.

        Optional filters narrow by category, severity, and/or time window.
        The time_window filter checks timestamp_start >= window_start and
        timestamp_end <= window_end on each chunk's stored metadata.
        Returns a list of payload dicts.
        """
        must = []
        if category is not None:
            must.append(FieldCondition(key="category", match=MatchValue(value=category)))
        if severity is not None:
            must.append(FieldCondition(key="severity", match=MatchValue(value=severity)))
        if time_window:
            start, end = time_window
            if start:
                must.append(
                    FieldCondition(
                        key="timestamp_start", range=Range(gte=start.timestamp())
                    )
                )
            if end:
                must.append(
                    FieldCondition(
                        key="timestamp_end", range=Range(lte=end.timestamp())
                    )
                )

        query_filter = Filter(must=must) if must else None

        response = self._client.query_points(
            collection_name=_COLLECTION,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )
        return [hit.payload for hit in response.points if hit.payload]

    def close(self) -> None:
        """Release file locks and resources (required before re-opening from_path stores)."""
        self._client.close()

    def count(self) -> int:
        """Return number of chunks stored."""
        return self._client.count(collection_name=_COLLECTION).count

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't already exist."""
        existing = {c.name for c in self._client.get_collections().collections}
        if _COLLECTION not in existing:
            self._client.create_collection(
                collection_name=_COLLECTION,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
