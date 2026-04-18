from __future__ import annotations

from bundle_platform.eval.strategies.with_dedup import (
    WithDedupStrategy,
    collapse_consecutive_duplicates,
)


def test_collapse_identical_consecutive_lines() -> None:
    assert collapse_consecutive_duplicates(["a", "a", "a", "b", "c", "c"]) == ["a", "b", "c"]


def test_collapse_preserves_non_consecutive_duplicates() -> None:
    assert collapse_consecutive_duplicates(["a", "b", "a", "b"]) == ["a", "b", "a", "b"]


def test_with_dedup_name() -> None:
    assert WithDedupStrategy().name == "with_dedup"
