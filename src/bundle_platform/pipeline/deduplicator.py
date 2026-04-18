# src/bundle_platform/pipeline/deduplicator.py
from __future__ import annotations


def deduplicate(lines: list[str]) -> list[str]:
    """Remove duplicate lines while preserving first-occurrence order.

    Unlike collapse_consecutive_duplicates (which only removes adjacent repeats),
    this removes ALL duplicates regardless of position. Use when the same log
    line appears scattered across a file and exact repetitions add no value.

    The trade-off: deduplication can remove context clues (e.g., three identical
    SCSI errors over 30 minutes indicate a sustained problem, not a one-off).
    Use collapse_consecutive_duplicates when ordering matters; use this when
    you only care about which events occurred, not how many times.
    """
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out
