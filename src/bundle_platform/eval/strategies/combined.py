# src/bundle_platform/eval/strategies/combined.py
from __future__ import annotations

from bundle_platform.eval.strategies.with_drain3 import WithDrain3Strategy
from bundle_platform.eval.strategies.with_rerank import (
    _format_chunks,
    _parse_context_block,
)
from bundle_platform.eval.strategy import RetrievedContext
from bundle_platform.pipeline.deduplicator import collapse_consecutive_duplicates
from bundle_platform.pipeline.reranker import CrossEncoderReranker

_RERANK_TOP_N = 10


class CombinedStrategy(WithDrain3Strategy):
    """All three tools applied in sequence: dedup → drain3 → rerank.

    Pipeline:
      1. Retrieve chunks via baseline hybrid search
      2. Collapse consecutive duplicate lines (dedup)
      3. Apply Drain3 template extraction (drain3)
      4. Re-rank with cross-encoder (rerank)

    This is the 'kitchen sink' strategy. It may or may not beat individual
    strategies on any given bundle type — run the scorecard to find out.
    When it performs worse than the best individual strategy, test pairs
    (with_dedup + with_rerank) without Drain3.
    """

    name = "combined"

    def __init__(self) -> None:
        super().__init__()
        self._reranker = CrossEncoderReranker()

    def retrieve(self, question: str) -> RetrievedContext:
        # Call BaselineStrategy directly to get raw retrieved context; bypasses
        # WithDrain3Strategy.retrieve intentionally so this class controls the
        # full pipeline order.
        from bundle_platform.eval.strategies.baseline import BaselineStrategy

        base = BaselineStrategy.retrieve(self, question)

        deduped_text = "\n".join(collapse_consecutive_duplicates(base.text.splitlines()))

        templated_lines = [self._miner.add_log_message(line) for line in deduped_text.splitlines()]
        templated_text = "\n".join(templated_lines)

        chunks = _parse_context_block(templated_text)
        if not chunks:
            return RetrievedContext(text=templated_text, source_files=base.source_files)

        reranked = self._reranker.rerank(question, chunks, top_n=_RERANK_TOP_N)
        source_files = sorted({c.get("file_path", "") for c in reranked if c.get("file_path")})
        return RetrievedContext(text=_format_chunks(reranked), source_files=source_files)
