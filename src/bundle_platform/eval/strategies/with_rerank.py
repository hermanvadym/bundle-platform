# src/bundle_platform/eval/strategies/with_rerank.py
from __future__ import annotations

from bundle_platform.eval.strategies.baseline import BaselineStrategy
from bundle_platform.eval.strategy import RetrievedContext
from bundle_platform.pipeline.reranker import CrossEncoderReranker

_RERANK_TOP_N = 10


def _parse_context_block(text: str) -> list[dict]:
    """Parse retriever output text into list of chunk dicts.

    Handles both header formats emitted by retriever.py:
      === filepath (lines X-Y) ===
      === filepath (keyword matches) ===
    """
    chunks: list[dict] = []
    current_path = ""
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("=== ") and " (lines " in line:
            if current_lines:
                chunks.append({"file_path": current_path, "text": "\n".join(current_lines)})
                current_lines = []
            current_path = line.split(" (lines ")[0].removeprefix("=== ").strip()
        else:
            current_lines.append(line)

    if current_lines:
        chunks.append({"file_path": current_path, "text": "\n".join(current_lines)})
    return chunks


def _format_chunks(chunks: list[dict]) -> str:
    """Format chunk dicts back to the retriever text format."""
    parts = []
    for chunk in chunks:
        header = f"=== {chunk.get('file_path', 'unknown')} ==="
        parts.append(f"{header}\n{chunk.get('text', '')}")
    return "\n\n".join(parts)


class WithRerankStrategy(BaselineStrategy):
    """Baseline retrieval with cross-encoder re-ranking applied to top-K results.

    The retriever returns the top-K chunks by vector similarity. This strategy
    re-scores those chunks using a cross-encoder (query + chunk together), which
    captures query-document interaction that the bi-encoder embedding misses.

    This typically improves evidence quality without changing recall — the same
    chunks are retrieved, but the most relevant one is ranked first.
    """

    name = "with_rerank"

    def __init__(self) -> None:
        super().__init__()
        self._reranker = CrossEncoderReranker()

    def retrieve(self, question: str) -> RetrievedContext:
        base = super().retrieve(question)
        chunks = _parse_context_block(base.text)
        if not chunks:
            return base
        reranked = self._reranker.rerank(question, chunks, top_n=_RERANK_TOP_N)
        source_files = sorted({c.get("file_path", "") for c in reranked if c.get("file_path")})
        return RetrievedContext(text=_format_chunks(reranked), source_files=source_files)
