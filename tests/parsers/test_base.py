from pathlib import Path

import pytest

from bundle_platform.parsers.base import BundleAdapter


def test_cannot_instantiate_base():
    with pytest.raises(TypeError):
        BundleAdapter()  # type: ignore[abstract]


def test_concrete_without_methods_fails():
    class Incomplete(BundleAdapter):
        bundle_type = "test"
        # missing validate, tag_file, etc.

    with pytest.raises(TypeError):
        Incomplete()


def test_concrete_with_all_methods_ok():
    class Complete(BundleAdapter):
        bundle_type = "test"

        def validate(self, root: Path) -> None:
            pass

        def tag_file(self, path: str) -> str:
            return "other"

        def timestamp_format(self, path: str) -> str:
            return "unknown"

        def error_sweep_categories(self) -> frozenset[str]:
            return frozenset()

    obj = Complete()
    assert obj.bundle_type == "test"
    assert obj.failure_patterns() == ""  # stub default
