"""
Bundle unpacking and file indexing tools.

This module handles the first step of analysis: taking a raw sosreport archive,
extracting it to a temporary directory, and building a FileManifest — an indexed
map of every file in the bundle with its size and diagnostic category.

The FileManifest is the foundation for everything else:
- The agent uses it to orient itself (which files exist, what are they?)
- Tools like grep_log() and read_section() take file paths from it
- The total_chars field drives the token-efficiency comparison (naive baseline)
"""

import fnmatch
import sys
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from bundle_platform.shared.caps import cap_lines


@dataclass
class FileEntry:
    """
    Represents a single file within an unpacked sosreport bundle.

    Attributes:
        path:       Relative path from bundle root (e.g. "var/log/messages").
                    Always uses forward slashes regardless of OS.
        size_bytes: Raw file size in bytes, used for display and rough filtering.
        category:   Diagnostic category assigned by the parser (e.g. "system_logs").
                    Used by list_files() to let the agent filter by type.
    """

    path: str
    size_bytes: int
    category: str


@dataclass
class FileManifest:
    """
    Complete index of an unpacked sosreport bundle.

    This is built once after unpacking and passed around to all tools and the
    agent loop. It avoids re-walking the filesystem on every query.

    Attributes:
        bundle_root:  Absolute path to the unpacked bundle directory.
                      Tools construct full paths by joining this with entry.path.
        entries:      All files in the bundle, sorted by path, with categories.
        total_chars:  Sum of character counts across all readable text files.
                      Used to compute the "naive baseline" token count —
                      what it would cost to load the entire bundle into context
                      instead of using targeted tool calls.
    """

    bundle_root: Path
    entries: list[FileEntry]
    total_chars: int


def _validate_member(member: tarfile.TarInfo, dest_root: Path) -> None:
    """Reject unsafe tar members: traversal, absolute paths, or escaping symlinks.

    Why: tarfile.data_filter catches many cases but raises tarfile-specific
    exceptions and behaviour varies by Python version. Belt-and-suspenders
    pre-validation gives us consistent ValueError with "unsafe" in the message
    regardless of Python version, and covers symlink cases data_filter misses.
    """
    # Reject absolute paths
    if Path(member.name).is_absolute() or member.name.startswith("/"):
        raise ValueError(f"unsafe path (absolute): {member.name!r}")

    # Normalize & reject traversal (escapes dest_root)
    dest_resolved = dest_root.resolve()
    member_resolved = (dest_resolved / member.name).resolve()
    if not member_resolved.is_relative_to(dest_resolved):
        raise ValueError(f"unsafe path (escapes root): {member.name!r}")

    # Reject symlinks / hardlinks escaping the root
    if member.issym() or member.islnk():
        link_target = (member_resolved.parent / member.linkname).resolve()
        if not link_target.is_relative_to(dest_resolved):
            kind = "symlink" if member.issym() else "hardlink"
            raise ValueError(f"unsafe {kind}: {member.name!r} -> {member.linkname!r}")


def _safe_data_filter(member: tarfile.TarInfo, dest_path: str) -> tarfile.TarInfo | None:
    """Like filter='data' but skips special files (sockets, FIFOs, devices).

    Sosreports snapshot /run/ which can contain socket/FIFO entries that the
    strict 'data' filter rejects with SpecialFileError.
    """
    try:
        return tarfile.data_filter(member, dest_path)
    except tarfile.SpecialFileError:
        return None


def unpack(archive_path: Path, dest_dir: Path) -> Path:
    """
    Extract a sosreport archive and return the root directory inside it.

    Why: sosreport archives are .tar.xz (or .tar.gz) files containing a single
    top-level directory named like "sosreport-hostname-date-id/". We extract to
    a temp directory and return that inner directory as the bundle root.

    How: Uses Python's tarfile module with filter="data" (Python 3.12+ security
    feature that prevents path traversal and setuid attacks in untrusted archives).
    After extraction, verifies exactly one directory exists — if not, the archive
    is malformed or not a sosreport.

    Args:
        archive_path: Path to the .tar.xz or .tar.gz sosreport archive.
        dest_dir:     Empty directory to extract into.

    Returns:
        Path to the single extracted directory (the sosreport root).

    Raises:
        ValueError: If the archive does not contain exactly one root directory.
    """
    with tarfile.open(archive_path) as tf:
        # Pre-validate every member before extraction (belt-and-suspenders):
        # raises ValueError with "unsafe" for traversal, absolute, or escaping
        # symlinks regardless of Python version. filter="data" then adds defence
        # in depth against setuid bits and device files.
        for member in tf.getmembers():
            _validate_member(member, dest_dir)
        tf.extractall(dest_dir, filter=_safe_data_filter)

    entries = list(dest_dir.iterdir())

    # A valid sosreport archive always has exactly one top-level directory.
    # Multiple entries indicate a non-standard archive or a bundling mistake.
    if len(entries) != 1 or not entries[0].is_dir():
        raise ValueError(
            f"Expected single directory in archive, found: {[e.name for e in entries]}"
        )
    return entries[0]


def index_files(bundle_root: Path, tagger: Callable[[str], str]) -> FileManifest:
    """
    Walk the bundle and build a FileManifest with categories and total char count.

    Why: Building the index once upfront is far cheaper than re-walking the
    filesystem on every tool call. The manifest gives the agent a complete map
    of the bundle before it starts asking questions.

    How: Recursively walks all files under bundle_root (sorted for determinism),
    reads each text file to count characters (for the naive baseline calculation),
    and calls the tagger function to assign a diagnostic category.

    Args:
        bundle_root: Root directory of the unpacked sosreport.
        tagger:      Function mapping a relative file path to a category string.
                     In practice this is parsers.rhel.tag_file.

    Returns:
        FileManifest with all files indexed, categorized, and total_chars summed.
    """
    entries: list[FileEntry] = []
    total_chars = 0

    # sorted() ensures consistent ordering across runs — important for deterministic
    # test output and predictable agent behavior when it calls list_files().
    for path in sorted(bundle_root.rglob("*")):
        if not path.is_file():
            continue  # skip directories

        rel = str(path.relative_to(bundle_root))
        size = path.stat().st_size

        try:
            # Read as text to count characters for the naive token baseline.
            # errors="ignore" skips undecodable bytes (binary files, corrupted logs).
            # OSError is caught in case of permission issues or race conditions.
            chars = len(path.read_text(errors="ignore"))
            total_chars += chars
        except OSError:
            # File is still indexed; char count just won't contribute to naive baseline.
            print(f"Warning: cannot read {rel} for char count (skipping baseline)", file=sys.stderr)

        entries.append(FileEntry(path=rel, size_bytes=size, category=tagger(rel)))

    return FileManifest(bundle_root=bundle_root, entries=entries, total_chars=total_chars)


def list_files(
    manifest: FileManifest,
    pattern: str | None = None,
    category: str | None = None,
) -> str:
    """
    Return a human-readable (and agent-readable) list of files from the manifest.

    This is one of the six tools exposed to Claude. The agent calls it first to
    orient itself — "what files exist in this bundle that are relevant to my query?"
    — before deciding which files to examine with grep_log() or read_section().

    Args:
        manifest: The bundle's FileManifest.
        pattern:  Optional glob pattern to match against file paths (e.g. "var/log/*").
        category: Optional category tag to filter by (e.g. "system_logs").
                  If both pattern and category are given, both filters are applied (AND).

    Returns:
        Formatted string listing matching files with size and category.
        Capped at 200 entries to stay within reasonable token limits.
        Returns a "No files match" message if nothing matches, so the agent
        knows to try different criteria rather than receiving an empty string.
    """
    entries = manifest.entries

    # Apply category filter first (cheaper than glob matching)
    if category:
        entries = [e for e in entries if e.category == category]

    # Apply glob pattern filter if given
    if pattern:
        entries = [e for e in entries if fnmatch.fnmatch(e.path, pattern)]

    if not entries:
        # Explicit message (not empty string) so the agent gets clear feedback
        return "No files match the given criteria."

    lines = [f"{e.path} ({e.size_bytes} bytes, {e.category})" for e in entries]
    # Notify the agent if results were truncated so it knows to narrow its query
    return cap_lines("\n".join(lines), limit=200)
