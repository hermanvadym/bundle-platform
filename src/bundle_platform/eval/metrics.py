from __future__ import annotations

import re

from bundle_platform.eval.golden import GoldenQuestion
from bundle_platform.eval.strategy import RetrievedContext


def score_deterministic(
    question: GoldenQuestion,
    context: RetrievedContext,
    answer: str,
) -> dict[str, float]:
    """Compute deterministic eval metrics for one question/answer pair.

    Returns a dict with three float scores, each in [0.0, 1.0]:
      evidence_file_recall  — fraction of expected source files retrieved
      evidence_regex_match  — 1.0 if regex matches retrieved text, else 0.0
      answer_keyword_match  — fraction of expected keywords in answer
    """
    # evidence_file_recall: what fraction of expected files were retrieved
    expected = set(question.expected_files)
    retrieved = set(context.source_files)
    file_recall = len(expected & retrieved) / len(expected) if expected else 1.0

    # evidence_regex_match: does the retrieved text contain the expected pattern
    if question.expected_evidence_regex:
        regex_match = 1.0 if re.search(
            question.expected_evidence_regex, context.text
        ) else 0.0
    else:
        regex_match = 1.0

    # answer_keyword_match: fraction of expected keywords present in answer
    if question.expected_answer_contains:
        found = sum(
            1 for kw in question.expected_answer_contains if kw in answer
        )
        keyword_match = found / len(question.expected_answer_contains)
    else:
        keyword_match = 1.0

    return {
        "evidence_file_recall": file_recall,
        "evidence_regex_match": regex_match,
        "answer_keyword_match": keyword_match,
    }
