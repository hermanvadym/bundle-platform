from __future__ import annotations

from pathlib import Path

import pytest

from bundle_platform.eval.golden import load_golden_set

FIXTURES = Path(__file__).parent / "fixtures" / "golden"


def test_load_valid_set_parses_all_fields() -> None:
    questions = load_golden_set(FIXTURES / "valid")
    assert len(questions) == 1
    q = questions[0]
    assert q.id == "001_oom"
    assert q.bundle == "sosreport.tar.xz"
    assert q.question.startswith("What process")
    assert q.expected_files == ["var/log/messages"]
    assert q.expected_evidence_regex == "oom_kill.*mysqld"
    assert "mysqld" in q.expected_answer_contains


def test_missing_required_field_raises() -> None:
    with pytest.raises(ValueError, match="missing required field 'question'"):
        load_golden_set(FIXTURES / "invalid")


def test_empty_dir_returns_empty_list(tmp_path: Path) -> None:
    assert load_golden_set(tmp_path) == []
