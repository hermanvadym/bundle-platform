from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class RetrievedContext:
    text: str
    source_files: list[str]


@runtime_checkable
class Strategy(Protocol):
    name: str

    def preprocess(self, bundle_root: Path) -> None: ...
    def retrieve(self, question: str) -> RetrievedContext: ...
