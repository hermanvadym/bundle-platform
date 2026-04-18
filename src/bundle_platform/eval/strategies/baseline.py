from __future__ import annotations

import os
from pathlib import Path

from bundle_platform.eval.strategy import RetrievedContext
from bundle_platform.pipeline.embedder import Embedder
from bundle_platform.pipeline.retriever import Retriever
from bundle_platform.pipeline.store import VectorStore


class BaselineStrategy:
    """Vanilla hybrid retrieval with no pre/post processing.

    Uses the same Embedder + Retriever as the main agent pipeline,
    so baseline scores reflect the raw ingestion quality before any
    strategy improvements are applied.
    """

    name = "baseline"

    def __init__(self) -> None:
        self._retriever: Retriever | None = None

    def preprocess(self, bundle_root: Path) -> None:
        """Initialise retriever pointed at the store path from environment.

        Called once before any retrieve() calls for a given bundle.
        Reads BUNDLE_PLATFORM_STORE_PATH or uses an in-memory store for
        testing. Reads BUNDLE_PLATFORM_EMBED_MODEL from environment —
        same variables the main agent uses.
        """
        store_path = os.environ.get("BUNDLE_PLATFORM_STORE_PATH")
        if store_path:
            store = VectorStore.from_path(Path(store_path))
        else:
            store = VectorStore.in_memory()

        embedder = Embedder()
        self._retriever = Retriever(
            store=store,
            embedder=embedder,
            bundle_root=bundle_root,
        )

    def retrieve(self, question: str) -> RetrievedContext:
        if self._retriever is None:
            raise RuntimeError("preprocess() must be called before retrieve()")
        text = self._retriever.retrieve(question)
        # Parse source file paths from retriever output headers.
        # Format: === filepath (lines X-Y) === or === filepath (keyword matches) ===
        files = sorted({
            line.removeprefix("=== ").split(" (")[0].strip()
            for line in text.splitlines()
            if line.startswith("=== ") and " ===" in line
        })
        return RetrievedContext(text=text, source_files=files)
