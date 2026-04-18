# tests/pipeline/test_deduplicator.py
from __future__ import annotations

from bundle_platform.pipeline.deduplicator import collapse_consecutive_duplicates, deduplicate


def test_removes_all_duplicates() -> None:
    assert deduplicate(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_preserves_first_occurrence_order() -> None:
    assert deduplicate(["c", "a", "b", "a"]) == ["c", "a", "b"]


def test_empty_input() -> None:
    assert deduplicate([]) == []


def test_collapse_consecutive_preserves_non_adjacent_duplicates() -> None:
    lines = ["a", "a", "b", "a"]
    assert collapse_consecutive_duplicates(lines) == ["a", "b", "a"]


def test_collapse_consecutive_empty() -> None:
    assert collapse_consecutive_duplicates([]) == []


def test_collapse_consecutive_no_duplicates() -> None:
    assert collapse_consecutive_duplicates(["a", "b", "c"]) == ["a", "b", "c"]
