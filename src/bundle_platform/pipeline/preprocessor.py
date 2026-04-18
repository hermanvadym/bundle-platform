# pipeline/preprocess.py
"""
Preprocessing pipeline for diagent RAG.

Orchestrates: chunk_manifest → embed in batches → upsert to Qdrant.
Results are cached to disk at ~/.cache/diagent/<bundle-id>/qdrant/
so repeated sessions on the same bundle skip preprocessing entirely.

Cache key: bundle filename + file size + mtime (avoids hashing 3 GB files).
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bundle_platform.pipeline.chunker import chunk_manifest
from bundle_platform.pipeline.embedder import Embedder
from bundle_platform.pipeline.image_describer import describe_images
from bundle_platform.pipeline.store import VectorStore
from bundle_platform.shared.timestamps import ts_to_float as _ts_to_float
from bundle_platform.tools.generic import FileManifest

_CACHE_BASE = Path.home() / ".cache" / "diagent"
_DONE_FILE = "preprocessed.json"



def _bundle_hash(bundle_path: Path) -> str:
    """Return a short sha256 of the first 64 KB of the bundle file.

    Using content bytes (not mtime/size) avoids cache collisions when bundles
    are copied or have identical metadata but different contents.
    """
    h = hashlib.sha256()
    with bundle_path.open("rb") as fh:
        h.update(fh.read(64 * 1024))
    return h.hexdigest()[:16]


def cache_dir(bundle_path: Path) -> Path:
    """Return the cache directory for a given bundle archive."""
    content_hash = _bundle_hash(bundle_path)
    bundle_id = f"{bundle_path.stem}_{content_hash}"
    return _CACHE_BASE / bundle_id


def is_preprocessed(bundle_path: Path) -> bool:
    """Return True if a valid preprocessed index exists for this bundle."""
    done = cache_dir(bundle_path) / _DONE_FILE
    return done.exists()


def preprocess_bundle(
    manifest: FileManifest,
    bundle_root: Path,
    bundle_path: Path,
    embedder: Embedder | None = None,
    bundle_type: str = "unknown",
) -> VectorStore:
    """
    Chunk, embed, and index all eligible files from the bundle.

    Saves the index to disk so subsequent sessions can call load_store()
    instead of reprocessing. Progress is printed to stdout.

    Args:
        manifest:     FileManifest from index_files().
        bundle_root:  Path to the unpacked bundle directory.
        bundle_path:  Path to the original archive (used for cache key).
        embedder:     Embedder instance (created if not provided).

    Returns:
        A VectorStore backed by the on-disk Qdrant collection.
    """
    if embedder is None:
        embedder = Embedder()

    qdrant_path = cache_dir(bundle_path) / "qdrant"
    store = VectorStore.from_path(qdrant_path)

    print("Chunking files...", flush=True)
    chunks = chunk_manifest(bundle_root, manifest, bundle_type=bundle_type)
    image_chunks = describe_images(manifest, bundle_root, bundle_type=bundle_type)
    chunks.extend(image_chunks)
    print(
        f"  {len(chunks)} chunks from {len(manifest.entries)} files"
        f" ({len(image_chunks)} screenshot description(s))",
        flush=True,
    )

    if not chunks:
        _mark_done(bundle_path, chunk_count=0)
        return store

    # Embed in parallel batches, then upsert sequentially.
    # Embedding is the bottleneck (network round-trip per batch); upserts are local disk I/O.
    # ThreadPoolExecutor saturates the embedding API without overloading Qdrant.
    batch_size = 64
    total = len(chunks)
    batches = [chunks[i : i + batch_size] for i in range(0, total, batch_size)]

    def _embed_batch(batch):
        return embedder.embed_texts([c.text for c in batch])

    embedded: int = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for batch, vectors in zip(batches, pool.map(_embed_batch, batches), strict=True):
            payloads = [
                {
                    "file_path": c.file_path,
                    "category": c.category,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "text": c.text,
                    "severity": c.severity,
                    "bundle_type": c.bundle_type,
                    "timestamp_start": _ts_to_float(c.timestamp_start),
                    "timestamp_end": _ts_to_float(c.timestamp_end),
                }
                for c in batch
            ]
            store.upsert(vectors, payloads)
            embedded += len(batch)
            print(f"  Embedded {embedded}/{total} chunks...", end="\r", flush=True)

    print(f"  Embedded {total}/{total} chunks. Done.     ", flush=True)
    _mark_done(bundle_path, chunk_count=total)
    return store


def load_store(bundle_path: Path) -> VectorStore:
    """Load the preprocessed Qdrant store for this bundle from disk."""
    qdrant_path = cache_dir(bundle_path) / "qdrant"
    return VectorStore.from_path(qdrant_path)


def _mark_done(bundle_path: Path, chunk_count: int) -> None:
    done_path = cache_dir(bundle_path) / _DONE_FILE
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.write_text(json.dumps({"chunks": chunk_count}))
