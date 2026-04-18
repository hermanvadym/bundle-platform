"""
Hybrid retriever: semantic search via Qdrant + targeted grep on top files.

Retrieval flow per query:
  1. Embed query → Qdrant ANN search → top-20 chunks
  2. Identify top-3 unique files by match score
  3. Run grep on those files using keywords extracted from the query
  4. Merge semantic chunks + grep hits, dedup by line overlap
  5. Format as a context block capped at ~15 000 chars (~3 750 tokens)

The returned string is inserted directly into the user message sent to Claude,
replacing the multi-tool-call loop with a single bounded API call.
"""

import gzip
import re
from datetime import datetime
from pathlib import Path

from bundle_platform.pipeline.embedder import Embedder
from bundle_platform.pipeline.store import VectorStore

_MAX_CONTEXT_CHARS = 15_000
_SEMANTIC_TOP_K = 20
_GREP_TOP_FILES = 3
_GREP_MAX_LINES = 30

_TRUNCATION_SUFFIX = "\n[truncated]"

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall on in at to for of and or "
    "but not what how why when where which who whom this that these those "
    "with from by about into through during including until against among "
    "i you we they it its my your our their if any some all".split()
)


def _extract_keywords(query: str) -> list[str]:
    """Extract meaningful words from a query, filtering stopwords."""
    words = re.findall(r"[a-zA-Z0-9_-]{3,}", query.lower())
    return [w for w in words if w not in _STOPWORDS]


class Retriever:
    """Hybrid retriever combining Qdrant semantic search with targeted grep."""

    def __init__(self, store: VectorStore, embedder: Embedder, bundle_root: Path) -> None:
        self._store = store
        self._embedder = embedder
        self._bundle_root = bundle_root

    def retrieve(
        self,
        question: str,
        time_window: tuple[datetime | None, datetime | None] | None = None,
    ) -> str:
        """
        Retrieve context relevant to question.

        time_window, if provided, is forwarded to the vector store so only
        chunks whose timestamps fall in the window are considered. This is
        the primary mechanism for time-scoped analysis.

        Returns empty string if no chunks found.
        """
        query_vector = self._embedder.embed_query(question)
        semantic_chunks = self._store.search(
            query_vector, top_k=_SEMANTIC_TOP_K, time_window=time_window
        )

        if not semantic_chunks:
            return ""

        top_files = _unique_top_files(semantic_chunks, n=_GREP_TOP_FILES)
        keywords = _extract_keywords(question)
        grep_chunks = self._grep_files(top_files, keywords) if keywords else []

        return _format_context(semantic_chunks + grep_chunks)

    def _grep_files(self, file_paths: list[str], keywords: list[str]) -> list[dict]:
        """Run regex grep across the given files using extracted keywords."""
        if not keywords:
            return []

        pattern = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)
        results: list[dict] = []

        for file_path in file_paths:
            full_path = self._bundle_root / file_path
            # Guard against path traversal from corrupted Qdrant payloads
            if not full_path.resolve().is_relative_to(self._bundle_root.resolve()):
                continue
            try:
                if full_path.suffix == ".gz":
                    text = gzip.open(full_path, "rt", encoding="utf-8", errors="replace").read()
                else:
                    text = full_path.read_text(errors="replace")
                lines = text.splitlines()
            except OSError:
                continue

            matched: list[str] = []
            for i, line in enumerate(lines, start=1):
                if pattern.search(line):
                    matched.append(f"{i}: {line}")
                    if len(matched) >= _GREP_MAX_LINES:
                        break

            if matched:
                results.append({
                    "file_path": file_path,
                    "category": "grep",
                    "start_line": None,
                    "end_line": None,
                    "text": "\n".join(matched),
                    "severity": None,
                })

        return results


def _unique_top_files(chunks: list[dict], n: int) -> list[str]:
    """Return the first n unique file paths from ranked chunk results."""
    seen: list[str] = []
    for chunk in chunks:
        fp = chunk["file_path"]
        if fp not in seen:
            seen.append(fp)
        if len(seen) == n:
            break
    return seen


def _format_context(chunks: list[dict]) -> str:
    """Format chunks into a context block for Claude's user message.

    Output is capped at _MAX_CONTEXT_CHARS characters.
    """
    sections: list[str] = []
    total_chars = 0

    for chunk in chunks:
        fp = chunk["file_path"]
        sl = chunk.get("start_line", 0)
        el = chunk.get("end_line", 0)
        text = chunk.get("text", "")

        if sl and el:
            header = f"=== {fp} (lines {sl}-{el}) ==="
        else:
            header = f"=== {fp} (keyword matches) ==="

        section = f"{header}\n{text}"
        # Account for the "\n\n" separator that join() inserts between sections
        separator = "\n\n" if sections else ""
        combined_len = len(separator) + len(section)
        if total_chars + combined_len > _MAX_CONTEXT_CHARS:
            # remaining chars available for the section content (after separator and suffix)
            remaining = _MAX_CONTEXT_CHARS - total_chars - len(separator) - len(_TRUNCATION_SUFFIX)
            if remaining > len(header) + 20:
                sections.append(section[:remaining] + _TRUNCATION_SUFFIX)
            break

        sections.append(section)
        total_chars += combined_len

    return "\n\n".join(sections)
