# src/bundle_platform/eval/strategies/with_drain3.py
from __future__ import annotations

from pathlib import Path

from bundle_platform.eval.strategies.baseline import BaselineStrategy
from bundle_platform.eval.strategy import RetrievedContext
from bundle_platform.pipeline.template_miner import TemplateMinerWrapper


class WithDrain3Strategy(BaselineStrategy):
    """Baseline retrieval with Drain3 template extraction applied to retrieved text.

    After retrieval, each log line in the retrieved context is passed through
    the Drain3 template miner, which replaces variable tokens with <*>.
    The templated text is then passed to the LLM answerer.

    Why post-retrieval rather than pre-ingestion?
    Applying Drain3 at eval time lets us compare templated vs raw text on the
    same Qdrant index without re-indexing. For production use, pre-ingestion
    templating would be more efficient but harder to A/B test.

    Warning: if evidence_regex_match drops vs baseline, Drain3 is stripping
    tokens that are the evidence (e.g., device IDs). Check the scorecard.
    """

    name = "with_drain3"

    def __init__(self) -> None:
        super().__init__()
        self._miner = TemplateMinerWrapper()

    def retrieve(self, question: str) -> RetrievedContext:
        base = super().retrieve(question)
        templated_lines = [
            self._miner.add_log_message(line)
            for line in base.text.splitlines()
        ]
        return RetrievedContext(
            text="\n".join(templated_lines),
            source_files=base.source_files,
        )
