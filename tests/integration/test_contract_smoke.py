from __future__ import annotations

import os

import pytest

QDRANT_URL = os.environ.get("QDRANT_URL", "")
COLLECTION = os.environ.get("BUNDLE_PLATFORM_COLLECTION", "")
EMBED_MODEL = os.environ.get("BUNDLE_PLATFORM_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

pytestmark = pytest.mark.skipif(
    not QDRANT_URL or not COLLECTION,
    reason="Set QDRANT_URL and BUNDLE_PLATFORM_COLLECTION to run integration tests",
)


def test_retriever_returns_results() -> None:
    from bundle_platform.pipeline.embedder import Embedder
    from bundle_platform.pipeline.store import VectorStore

    embedder = Embedder(model_name=EMBED_MODEL)
    store = VectorStore(url=QDRANT_URL, collection_name=COLLECTION)

    query_vector = embedder.embed_query("storage error vmkernel")
    results = store.search(query_vector, top_k=5)

    assert results, (
        f"Expected at least one result from collection '{COLLECTION}' at {QDRANT_URL}. "
        "Check that: (1) log-analyse completed a run for this run_id, "
        "(2) BUNDLE_PLATFORM_EMBED_MODEL matches log-analyse's semantic_vectorization.model_name."
    )
    first = results[0]
    assert "file_path" in first or "source_file" in first, (
        "Result payload missing 'file_path'/'source_file' — payload schema mismatch."
    )


def test_embed_model_produces_expected_dimension() -> None:
    from bundle_platform.pipeline.embedder import Embedder

    embedder = Embedder(model_name=EMBED_MODEL)
    vector = embedder.embed_query("test query")
    assert len(vector) > 0, "Embedder returned empty vector"
    print(f"Embedding dimension: {len(vector)}")
