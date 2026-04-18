# tests/pipeline/test_reranker.py
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bundle_platform.pipeline.reranker import CrossEncoderReranker


def _chunks(texts: list[str]) -> list[dict]:
    return [{"text": t, "file_path": f"file_{i}.log"} for i, t in enumerate(texts)]


def test_reranker_empty_input_returns_empty() -> None:
    reranker = CrossEncoderReranker()
    assert reranker.rerank("any query", [], top_n=5) == []


def test_reranker_preserves_chunk_fields() -> None:
    reranker = CrossEncoderReranker()
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.9]
    reranker._model = mock_model

    result = reranker.rerank("vmkernel error", _chunks(["error in vmkernel"]), top_n=1)
    assert len(result) == 1
    assert "file_path" in result[0]
    assert "text" in result[0]
    assert "rerank_score" in result[0]


def test_reranker_returns_top_n() -> None:
    reranker = CrossEncoderReranker()
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.9, 0.5]
    reranker._model = mock_model

    result = reranker.rerank("query", _chunks(["a", "b", "c"]), top_n=2)
    assert len(result) == 2
    assert result[0]["text"] == "b"  # highest score 0.9


def test_reranker_adds_rerank_score() -> None:
    reranker = CrossEncoderReranker()
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.75]
    reranker._model = mock_model

    result = reranker.rerank("q", _chunks(["text"]), top_n=1)
    assert result[0]["rerank_score"] == pytest.approx(0.75)
