# tests/pipeline/test_deduplicator.py
from __future__ import annotations

from bundle_platform.pipeline.deduplicator import deduplicate


def test_removes_all_duplicates() -> None:
    assert deduplicate(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_preserves_first_occurrence_order() -> None:
    assert deduplicate(["c", "a", "b", "a"]) == ["c", "a", "b"]


def test_empty_input() -> None:
    assert deduplicate([]) == []
