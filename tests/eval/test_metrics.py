from __future__ import annotations

from bundle_platform.eval.golden import GoldenQuestion
from bundle_platform.eval.metrics import score_deterministic
from bundle_platform.eval.strategy import RetrievedContext


def _q(**kw: object) -> GoldenQuestion:
    defaults: dict = dict(
        id="q1", bundle="b.tar", question="?",
        expected_files=["var/log/messages"],
        expected_evidence_regex="oom_kill",
        expected_answer_contains=["mysqld", "OOM"],
    )
    defaults.update(kw)
    return GoldenQuestion(**defaults)


def test_all_expected_files_surfaced() -> None:
    ctx = RetrievedContext(text="", source_files=["var/log/messages", "other"])
    assert score_deterministic(_q(), ctx, answer="")["evidence_file_recall"] == 1.0


def test_partial_file_recall() -> None:
    ctx = RetrievedContext(text="", source_files=["a.log"])
    scores = score_deterministic(_q(expected_files=["a.log", "b.log"]), ctx, answer="")
    assert scores["evidence_file_recall"] == 0.5


def test_evidence_regex_match() -> None:
    ctx = RetrievedContext(text="kernel: oom_kill process 9841", source_files=[])
    assert score_deterministic(_q(), ctx, answer="")["evidence_regex_match"] == 1.0


def test_evidence_regex_miss() -> None:
    ctx = RetrievedContext(text="nothing here", source_files=[])
    assert score_deterministic(_q(), ctx, answer="")["evidence_regex_match"] == 0.0


def test_answer_keyword_match_partial() -> None:
    ctx = RetrievedContext(text="", source_files=[])
    assert score_deterministic(_q(), ctx, answer="mysqld died")["answer_keyword_match"] == 0.5


def test_no_regex_scores_full() -> None:
    ctx = RetrievedContext(text="", source_files=[])
    scores = score_deterministic(_q(expected_evidence_regex=""), ctx, answer="")
    assert scores["evidence_regex_match"] == 1.0
