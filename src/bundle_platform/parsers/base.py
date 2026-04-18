"""BundleAdapter ABC — bundle-type-specific behaviour lives here.

Each parser module (parsers/rhel.py, parsers/esxi.py, …) exposes a
get_adapter() returning a concrete BundleAdapter subclass. New bundle types
are added by creating a new parser module and registering it in
parsers/__init__.load_adapter().

failure_patterns() is stubbed to return "" — proper detector implementations
live in detectors/ (Phase 7).
"""

from abc import ABC, abstractmethod
from pathlib import Path


class BundleAdapter(ABC):
    bundle_type: str

    @abstractmethod
    def validate(self, root: Path) -> None:
        """Raise ValueError if `root` is not a valid bundle of this type."""

    @abstractmethod
    def tag_file(self, path: str) -> str:
        """Return the category string for a file path within the bundle."""

    @abstractmethod
    def timestamp_format(self, path: str) -> str:
        """Return 'iso8601', 'syslog', 'bracket', or 'unknown'."""

    @abstractmethod
    def error_sweep_categories(self) -> frozenset[str]:
        """Categories find_errors() should scan for this bundle type."""

    def failure_patterns(self) -> str:
        """Prompt-ready description of common failure patterns. Stubbed until Phase 7."""
        return ""
