from __future__ import annotations

from pathlib import Path

from bundle_platform.eval.strategy import RetrievedContext, Strategy


class FakeStrategy:
    name = "fake"

    def preprocess(self, bundle_root: Path) -> None:
        pass

    def retrieve(self, question: str) -> RetrievedContext:
        return RetrievedContext(text="ctx", source_files=["a.log"])


def test_fake_matches_protocol(tmp_path: Path) -> None:
    s: Strategy = FakeStrategy()
    s.preprocess(tmp_path)
    r = s.retrieve("q")
    assert r.text == "ctx"
    assert r.source_files == ["a.log"]
