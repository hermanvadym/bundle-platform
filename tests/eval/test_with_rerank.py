# tests/eval/test_with_rerank.py
from __future__ import annotations

from unittest.mock import MagicMock

from bundle_platform.eval.strategies.with_rerank import WithRerankStrategy
from bundle_platform.eval.strategy import RetrievedContext


def test_rerank_strategy_name() -> None:
    assert WithRerankStrategy.__new__(WithRerankStrategy).name == "with_rerank"


def test_rerank_applied_to_retrieved_chunks() -> None:
    strategy = WithRerankStrategy.__new__(WithRerankStrategy)
    strategy._retriever = MagicMock()
    strategy._retriever.retrieve.return_value = (
        "=== var/log/vmkernel.log (lines 1-5) ===\n"
        "oom_kill process mysqld\n\n"
        "=== var/log/hostd.log (lines 1-3) ===\n"
        "host agent started"
    )
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        {
            "file_path": "var/log/vmkernel.log",
            "text": "oom_kill process mysqld",
            "rerank_score": 0.9,
        },
    ]
    strategy._reranker = mock_reranker

    result = strategy.retrieve("OOM killed process")
    assert isinstance(result, RetrievedContext)
    assert "var/log/vmkernel.log" in result.source_files
    mock_reranker.rerank.assert_called_once()
