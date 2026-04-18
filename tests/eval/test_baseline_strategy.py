from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bundle_platform.eval.strategies.baseline import BaselineStrategy
from bundle_platform.eval.strategy import RetrievedContext


def test_baseline_name() -> None:
    assert BaselineStrategy().name == "baseline"


def test_retrieve_raises_before_preprocess() -> None:
    with pytest.raises(RuntimeError, match="preprocess"):
        BaselineStrategy().retrieve("question")


def test_retrieve_returns_retrieved_context(tmp_path: Path) -> None:
    strategy = BaselineStrategy()
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = (
        "=== var/log/messages (lines 1-10) ===\noom_kill mysqld"
    )
    strategy._retriever = mock_retriever

    result = strategy.retrieve("What was OOM killed?")
    assert isinstance(result, RetrievedContext)
    assert "var/log/messages" in result.source_files
    assert "oom_kill" in result.text
