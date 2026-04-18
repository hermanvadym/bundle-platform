from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_REQUIRED = ("id", "bundle", "question", "expected_files")


@dataclass
class GoldenQuestion:
    id: str
    bundle: str
    question: str
    expected_files: list[str]
    expected_evidence_regex: str = ""
    expected_answer_contains: list[str] = field(default_factory=list)


def load_golden_set(directory: Path) -> list[GoldenQuestion]:
    """Load all YAML golden Q&A files from a directory.

    Each file must contain a single YAML document (a mapping).
    Required fields: id, bundle, question, expected_files.
    Raises ValueError if a required field is missing.
    Returns [] if the directory is empty or contains no .yaml files.
    """
    questions: list[GoldenQuestion] = []
    for path in sorted(directory.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key in _REQUIRED:
            if key not in raw:
                raise ValueError(
                    f"{path.name}: missing required field {key!r}"
                )
        questions.append(
            GoldenQuestion(
                id=raw["id"],
                bundle=raw["bundle"],
                question=raw["question"],
                expected_files=raw["expected_files"],
                expected_evidence_regex=raw.get("expected_evidence_regex", ""),
                expected_answer_contains=raw.get("expected_answer_contains", []),
            )
        )
    return questions
