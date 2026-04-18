# tests/eval/test_combined.py
from __future__ import annotations

from bundle_platform.eval.strategies.combined import CombinedStrategy


def test_combined_name() -> None:
    s = CombinedStrategy.__new__(CombinedStrategy)
    assert s.name == "combined"
