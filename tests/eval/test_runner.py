from __future__ import annotations

from pathlib import Path

from bundle_platform.eval.golden import GoldenQuestion
from bundle_platform.eval.runner import run_scorecard
from bundle_platform.eval.strategy import RetrievedContext


class StubStrategy:
    name = "stub"

    def preprocess(self, bundle_root: Path) -> None:
        pass

    def retrieve(self, question: str) -> RetrievedContext:
        return RetrievedContext(
            text="kernel: oom_kill mysqld",
            source_files=["var/log/messages"],
        )


def _answerer(question: str, context: str) -> str:
    return "mysqld was killed by OOM at 03:14"


def test_run_scorecard_produces_one_row_per_question(tmp_path: Path) -> None:
    question = GoldenQuestion(
        id="q1", bundle="b.tar",
        question="What was OOM killed?",
        expected_files=["var/log/messages"],
        expected_evidence_regex="oom_kill",
        expected_answer_contains=["mysqld", "OOM"],
    )
    card = run_scorecard(
        bundle_root=tmp_path,
        questions=[question],
        strategies=[StubStrategy()],
        answerer=_answerer,
        seeds=1,
    )
    assert len(card.rows) == 1
    row = card.rows[0]
    assert row["strategy"] == "stub"
    assert row["question_id"] == "q1"
    assert row["evidence_file_recall"] == 1.0
    assert row["evidence_regex_match"] == 1.0
    assert "seconds" in row


def test_run_scorecard_multiple_strategies(tmp_path: Path) -> None:
    question = GoldenQuestion(
        id="q1", bundle="b.tar", question="Q",
        expected_files=["f.log"],
        expected_evidence_regex="",
        expected_answer_contains=[],
    )

    class StratA:
        name = "a"
        def preprocess(self, r: Path) -> None: pass
        def retrieve(self, q: str) -> RetrievedContext:
            return RetrievedContext(text="", source_files=[])

    class StratB:
        name = "b"
        def preprocess(self, r: Path) -> None: pass
        def retrieve(self, q: str) -> RetrievedContext:
            return RetrievedContext(text="", source_files=[])

    card = run_scorecard(
        bundle_root=tmp_path,
        questions=[question],
        strategies=[StratA(), StratB()],
        answerer=lambda q, c: "",
        seeds=1,
    )
    assert len(card.rows) == 2
    strategies = {r["strategy"] for r in card.rows}
    assert strategies == {"a", "b"}
