"""
RHEL sosreport parser.

A sosreport is a diagnostic archive created by the `sos` tool on RHEL systems.
This module knows the internal structure of sosreport bundles — which directories
must exist, and which file paths belong to which diagnostic category.

Having this knowledge centralized here means the rest of the codebase (agent, tools)
can work with abstract categories like "system_logs" instead of hardcoded paths.
"""

from pathlib import Path

from bundle_platform.parsers.base import BundleAdapter

# Rules are checked in order — FIRST match wins.
# More specific paths (e.g. var/log/audit/) must come BEFORE broader prefixes
# (e.g. var/log/) to avoid incorrect categorization.
# Example: without the audit/ rule first, audit.log would be tagged "system_logs"
# instead of "audit".
_CATEGORY_RULES: list[tuple[str, str]] = [
    ("var/log/audit/", "audit"),
    ("var/log/libvirt/", "kvm_logs"),        # before var/log/ catch-alls
    ("var/log/qemu/", "kvm_logs"),
    ("var/log/messages", "system_logs"),
    ("var/log/dmesg", "system_logs"),
    ("var/log/boot.log", "system_logs"),
    ("sos_commands/virsh_", "kvm_commands"), # before sos_commands/ catch-all
    ("sos_commands/virt-", "kvm_commands"),
    ("sos_commands/", "sos_commands"),
    ("etc/libvirt/", "kvm_config"),          # before etc/ catch-all
    ("etc/qemu/", "kvm_config"),
    ("etc/multipath.conf", "storage"),          # multipath before etc/ fallback
    ("etc/sysconfig/network-scripts/", "network"),  # network-scripts before etc/ fallback
    ("proc/cmdline", "kernel"),
    ("proc/modules", "kernel"),
    ("etc/", "config"),  # catch-all for everything else under etc/
]

# These three directories must exist in every valid sosreport.
# The sos tool always creates them; their absence means the archive is not a sosreport
# (e.g. a wrong file, a truncated archive, or a different diagnostic format).
_REQUIRED_DIRS = ["sos_commands", "var/log", "etc"]


def validate(bundle_root: Path) -> None:
    """
    Verify that bundle_root is a valid RHEL sosreport directory.

    Why: Before indexing or analyzing a bundle, we want to fail fast with a clear
    error rather than silently producing empty or wrong results from a non-sosreport
    archive (e.g. a backup tarball or an ESXi vm-support bundle).

    How: Checks for the three directories that every sosreport contains by definition.
    Does not do deep content validation — just structure.

    Raises:
        ValueError: If any required directory is missing. The message names the
                    specific missing directory so the user knows exactly what's wrong.
    """
    for required in _REQUIRED_DIRS:
        if not (bundle_root / required).exists():
            raise ValueError(
                f"Not a valid sosreport: missing '{required}' in {bundle_root}"
            )


def tag_file(path: str) -> str:
    """
    Assign a diagnostic category to a file based on its path within the bundle.

    Why: The agent needs to reason about files by category (e.g. "show me all log files")
    without knowing every specific path. Tagging files at index time means the agent
    can filter by category using list_files() rather than guessing paths.

    How: Walks the ordered _CATEGORY_RULES list and returns the category for the
    first matching prefix. Falls back to "other" for unrecognized paths.

    Args:
        path: Relative file path within the sosreport bundle (e.g. "var/log/messages").

    Returns:
        Category string: one of "system_logs", "audit", "sos_commands", "kernel",
        "storage", "network", "config", or "other".
    """
    for prefix, category in _CATEGORY_RULES:
        # Strip trailing slash from prefix for exact-match comparison.
        # This handles the edge case where `path` is exactly "sos_commands" (no trailing slash)
        # but the rule has prefix "sos_commands/".
        if path.startswith(prefix) or path == prefix.rstrip("/"):
            return category
    if path.endswith(".csv"):
        return "event_archive"
    if path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff")):
        return "screenshots"
    return "other"


class _RhelAdapter(BundleAdapter):
    bundle_type: str = "rhel"

    def validate(self, root: Path) -> None:
        validate(root)

    def tag_file(self, path: str) -> str:
        return tag_file(path)

    def timestamp_format(self, path: str) -> str:
        return "unknown"

    def error_sweep_categories(self) -> frozenset[str]:
        return frozenset({"system_logs", "audit", "kernel", "kvm_logs", "event_archive"})


def get_adapter() -> BundleAdapter:
    return _RhelAdapter()
