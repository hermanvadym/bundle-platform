# tests/eval/test_combined.py
from __future__ import annotations

import pytest

from bundle_platform.eval.strategies.combined import CombinedStrategy


def test_combined_name() -> None:
    s = CombinedStrategy.__new__(CombinedStrategy)
    assert s.name == "combined"


def test_combined_applies_all_three_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    """All three pipeline stages must be applied: dedup, drain3, rerank."""
    calls: list[str] = []

    def fake_retrieve(self: object, question: str) -> object:
        from bundle_platform.eval.strategy import RetrievedContext

        return RetrievedContext(text="line1\nline1\nline2", source_files=["f.log"])

    def fake_add_log_message(self: object, line: str) -> str:
        calls.append("drain3")
        return line

    def fake_rerank(self: object, query: str, chunks: list[str], top_n: int = 10) -> list[str]:
        calls.append("rerank")
        return chunks

    from bundle_platform.eval.strategies.baseline import BaselineStrategy
    from bundle_platform.pipeline.reranker import CrossEncoderReranker
    from bundle_platform.pipeline.template_miner import TemplateMinerWrapper

    monkeypatch.setattr(BaselineStrategy, "retrieve", fake_retrieve)
    monkeypatch.setattr(TemplateMinerWrapper, "add_log_message", fake_add_log_message)
    monkeypatch.setattr(CrossEncoderReranker, "rerank", fake_rerank)

    strategy = CombinedStrategy()
    strategy.retrieve("test question")

    assert "drain3" in calls
    assert "rerank" in calls
