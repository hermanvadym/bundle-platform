from __future__ import annotations

from bundle_platform.eval.report import render_markdown
from bundle_platform.eval.runner import Scorecard


def test_render_contains_strategy_name() -> None:
    card = Scorecard(rows=[{
        "strategy": "baseline", "question_id": "q1",
        "evidence_file_recall": 1.0, "evidence_regex_match": 1.0,
        "answer_keyword_match": 1.0, "seconds": 0.5,
    }])
    md = render_markdown(card)
    assert "baseline" in md


def test_render_aggregates_per_strategy() -> None:
    card = Scorecard(rows=[
        {"strategy": "baseline", "question_id": "q1",
         "evidence_file_recall": 1.0, "evidence_regex_match": 1.0,
         "answer_keyword_match": 1.0, "seconds": 0.5},
        {"strategy": "baseline", "question_id": "q2",
         "evidence_file_recall": 0.5, "evidence_regex_match": 0.0,
         "answer_keyword_match": 0.5, "seconds": 0.5},
    ])
    md = render_markdown(card)
    assert "baseline" in md
    assert "0.75" in md
