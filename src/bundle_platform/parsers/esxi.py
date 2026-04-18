"""
ESXi vm-support bundle parser.

A vm-support bundle is a diagnostic archive created by the `vm-support` command on
ESXi hosts. This module knows the internal structure of vm-support bundles — which
directories must exist, which file paths belong to which diagnostic category, and
how to detect timestamp formats in each log file type.

Having this knowledge centralized here means the rest of the codebase (agent, tools)
can work with abstract categories like "system_logs" instead of hardcoded paths.
"""

from pathlib import Path

from bundle_platform.parsers.base import BundleAdapter

# Rules are checked in order — FIRST match wins.
# More specific paths (e.g. var/log/vmkernel) must come BEFORE broader prefixes
# (e.g. var/log/) to avoid incorrect categorization.
_CATEGORY_RULES: list[tuple[str, str]] = [
    ("var/log/vmkernel", "system_logs"),
    ("var/log/vmkwarning", "system_logs"),
    ("var/log/hostd", "host_agent"),
    ("var/log/vpxa", "host_agent"),
    ("var/log/fdm", "network"),
    ("var/log/lacp", "network"),
    ("var/log/net-cdp", "network"),
    ("var/log/storageRM", "storage"),
    ("var/log/vmkiscsid", "storage"),
    ("var/log/nmp", "storage"),
    ("vmfs/", "vm_logs"),
    ("commands/", "commands"),
    ("etc/", "config"),
]

# Timestamp format rules are checked in order — FIRST match wins.
_TIMESTAMP_RULES: list[tuple[str, str]] = [
    ("var/log/vmkernel", "iso8601"),
    ("var/log/vmkwarn", "iso8601"),
    ("var/log/hostd", "iso8601"),
    ("var/log/vpxa", "iso8601"),
    ("var/log/fdm", "iso8601"),
    ("var/log/storageRM", "iso8601"),
    ("var/log/syslog", "syslog"),
    ("var/log/auth", "syslog"),
    ("var/log/vobd", "bracket"),
]

# These two directories must exist in every valid ESXi vm-support bundle.
# Their absence means the archive is not a vm-support bundle
# (e.g. a wrong file, a truncated archive, or a different diagnostic format).
_REQUIRED_DIRS = ["var/log", "commands"]


def validate(bundle_root: Path) -> None:
    """
    Verify that bundle_root is a valid ESXi vm-support bundle directory.

    Why: Before indexing or analyzing a bundle, we want to fail fast with a clear
    error rather than silently producing empty or wrong results from a non-vm-support
    archive (e.g. a backup tarball or a RHEL sosreport).

    How: Checks for the two directories that every vm-support bundle contains by
    definition. Does not do deep content validation — just structure.

    Raises:
        ValueError: If any required directory is missing. The message names the
                    specific missing directory so the user knows exactly what's wrong.
    """
    for required in _REQUIRED_DIRS:
        if not (bundle_root / required).exists():
            raise ValueError(
                f"Not a valid vm-support bundle: missing '{required}' in {bundle_root}"
            )


def tag_file(path: str) -> str:
    """
    Assign a diagnostic category to a file based on its path within the bundle.

    Why: The agent needs to reason about files by category (e.g. "show me all logs")
    without knowing every specific path. Tagging files at index time means the agent
    can filter by category using list_files() rather than guessing paths.

    How: Walks the ordered _CATEGORY_RULES list and returns the category for the
    first matching prefix. Falls back to "other" for unrecognized paths.

    Args:
        path: Relative file path within the vm-support bundle
              (e.g. "var/log/vmkernel.log").

    Returns:
        Category string: one of "system_logs", "host_agent", "network", "storage",
        "vm_logs", "commands", "config", or "other".
    """
    for prefix, category in _CATEGORY_RULES:
        # Strip trailing slash from prefix for exact-match comparison.
        # This handles the edge case where `path` is exactly "commands" (no trailing slash)
        # but the rule has prefix "commands/".
        if path.startswith(prefix) or path == prefix.rstrip("/"):
            return category
    return "other"


def timestamp_format(path: str) -> str:
    """
    Detect the timestamp format used in a log file based on its path.

    Why: The agent needs to understand how to parse timestamps in different logs
    so it can correlate events across multiple sources by time.

    How: Walks the ordered _TIMESTAMP_RULES list and returns the format for the
    first matching prefix. Falls back to "unknown" for unrecognized paths.

    Args:
        path: Relative file path within the vm-support bundle
              (e.g. "var/log/vmkernel.log").

    Returns:
        Format string: one of "iso8601", "syslog", "bracket", or "unknown".
    """
    for prefix, fmt in _TIMESTAMP_RULES:
        if path.startswith(prefix) or path == prefix.rstrip("/"):
            return fmt
    return "unknown"


class _EsxiAdapter(BundleAdapter):
    bundle_type: str = "esxi"

    def validate(self, root: Path) -> None:
        validate(root)

    def tag_file(self, path: str) -> str:
        return tag_file(path)

    def timestamp_format(self, path: str) -> str:
        return timestamp_format(path)

    def error_sweep_categories(self) -> frozenset[str]:
        return frozenset({"system_logs", "host_agent", "storage"})


def get_adapter() -> BundleAdapter:
    return _EsxiAdapter()
