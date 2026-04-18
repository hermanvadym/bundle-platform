from __future__ import annotations

from bundle_platform.eval.strategies.baseline import BaselineStrategy
from bundle_platform.eval.strategy import RetrievedContext


def collapse_consecutive_duplicates(lines: list[str]) -> list[str]:
    """Collapse runs of identical adjacent lines to one occurrence.

    Only consecutive duplicates are removed — non-adjacent identical lines
    are preserved. This targets the common ESXi/KVM pattern where the same
    error line is repeated hundreds of times in a row.
    """
    out: list[str] = []
    for line in lines:
        if not out or out[-1] != line:
            out.append(line)
    return out


class WithDedupStrategy(BaselineStrategy):
    """Baseline retrieval with consecutive-duplicate line removal.

    Inherits preprocessing and retrieval from BaselineStrategy. After
    retrieval, duplicate adjacent lines are collapsed so the re-ranker
    and LLM see each unique event once rather than hundreds of repetitions.
    """

    name = "with_dedup"

    def retrieve(self, question: str) -> RetrievedContext:
        base = super().retrieve(question)
        deduped = "\n".join(
            collapse_consecutive_duplicates(base.text.splitlines())
        )
        return RetrievedContext(text=deduped, source_files=base.source_files)
