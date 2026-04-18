"""
Log file chunker for the RAG preprocessing pipeline.

Splits bundle files into overlapping text windows for embedding.
Each chunk carries metadata (file path, line range, category, severity)
that Qdrant stores as payload alongside the vector.

Chunking strategy:
- Files <= 100 lines: single chunk (whole file)
- Files > 100 lines: 100-line windows with 20-line overlap
- Category 'other': included (binary check still excludes core dumps and binaries)
- Binary files: skipped (detected by null bytes in first 1024 bytes)
- Empty files: skipped
"""

import gzip
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from bundle_platform.shared.timestamps import detect_timestamp_format, extract_timestamp_str
from bundle_platform.tools.generic import FileEntry, FileManifest

# Chunk size and overlap in lines.
_CHUNK_LINES = 100
_OVERLAP_LINES = 20
_STEP = _CHUNK_LINES - _OVERLAP_LINES  # 80 lines between chunk starts

# No categories are excluded from chunking — even 'other' files may contain useful
# diagnostic text. Binary files are excluded separately via null-byte detection below.
_SKIP_CATEGORIES: set[str] = set()

# Files larger than this are skipped entirely (likely binary or irrelevant).
_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB

# Regex for detecting error-level severity in chunk text.
_ERROR_RE = re.compile(
    r"\b(error|ERROR|Error|failed|FAILED|Failed|critical|CRITICAL|"
    r"oom_kill|Out of memory|Killed process|panic|PANIC|Oops:)\b"
)
_WARNING_RE = re.compile(r"\b(warning|WARNING|Warning|warn|WARN)\b")

def _extract_timestamp(line: str) -> str | None:
    """Return the first timestamp found in a log line, or None."""
    fmt = detect_timestamp_format(line)
    if fmt == "unknown":
        return None
    return extract_timestamp_str(line, fmt)


@dataclass
class LogChunk:
    """
    A window of lines from a file in the bundle, ready for embedding.

    Attributes:
        bundle_id:      Bundle name (used as Qdrant collection name).
        file_path:      Relative path from bundle root (e.g. "var/log/messages").
        category:       File category from parsers/rhel.py.
        start_line:     1-indexed first line of this chunk.
        end_line:       1-indexed last line of this chunk (inclusive).
        text:           Full text of the chunk (joined lines).
        severity:       "error", "warning", or None based on content scan.
        bundle_type:    "rhel" or "esxi" — which parser produced this chunk.
        timestamp_start: First timestamp found in the chunk's lines, or None.
        timestamp_end:   Last timestamp found in the chunk's lines, or None.
    """

    bundle_id: str
    file_path: str
    category: str
    start_line: int
    end_line: int
    text: str
    severity: str | None
    bundle_type: str = "unknown"
    timestamp_start: str | None = None
    timestamp_end: str | None = None


def chunk_file(
    bundle_root: Path, entry: FileEntry, bundle_type: str = "unknown"
) -> list[LogChunk]:
    """
    Split a single file into overlapping LogChunk windows.

    Returns an empty list for binary files, empty files, and files in
    skipped categories. Does not raise on I/O errors — returns [] instead.
    """
    if entry.category in _SKIP_CATEGORIES:
        return []
    if entry.size_bytes > _MAX_FILE_BYTES:
        return []

    path = bundle_root / entry.path
    try:
        if path.suffix == ".gz":
            # Decompress gzip — rotated logs (e.g. messages-20260401.gz) are common
            # in both RHEL sosreports and ESXi vm-support bundles
            raw = gzip.open(path, "rb").read()
        else:
            raw = path.read_bytes()
    except OSError:
        return []

    # Skip binary files (null byte in first 1024 bytes of decompressed content)
    if b"\x00" in raw[:1024]:
        return []

    try:
        text = raw.decode("utf-8", errors="replace")
    except ValueError:
        print(f"Warning: cannot decode {entry.path}, skipping chunks", file=sys.stderr)
        return []

    lines = text.splitlines()
    if not lines:
        return []

    # Derive bundle_id from the bundle root directory name
    bundle_id = bundle_root.name

    chunks: list[LogChunk] = []
    if len(lines) <= _CHUNK_LINES:
        chunk_text = "\n".join(lines)
        ts_values = [t for t in (_extract_timestamp(line) for line in lines) if t is not None]
        chunks.append(
            LogChunk(
                bundle_id=bundle_id,
                file_path=entry.path,
                category=entry.category,
                start_line=1,
                end_line=len(lines),
                text=chunk_text,
                severity=_detect_severity(chunk_text),
                bundle_type=bundle_type,
                timestamp_start=ts_values[0] if ts_values else None,
                timestamp_end=ts_values[-1] if ts_values else None,
            )
        )
    else:
        start = 0
        while start < len(lines):
            end = min(start + _CHUNK_LINES, len(lines))
            chunk_lines = lines[start:end]
            chunk_text = "\n".join(chunk_lines)
            ts_values = [
                t
                for t in (_extract_timestamp(line) for line in chunk_lines)
                if t is not None
            ]
            chunks.append(
                LogChunk(
                    bundle_id=bundle_id,
                    file_path=entry.path,
                    category=entry.category,
                    start_line=start + 1,  # 1-indexed
                    end_line=end,
                    text=chunk_text,
                    severity=_detect_severity(chunk_text),
                    bundle_type=bundle_type,
                    timestamp_start=ts_values[0] if ts_values else None,
                    timestamp_end=ts_values[-1] if ts_values else None,
                )
            )
            if end == len(lines):
                break
            start += _STEP

    return chunks


def chunk_manifest(
    bundle_root: Path, manifest: FileManifest, bundle_type: str = "unknown"
) -> list[LogChunk]:
    """
    Chunk all eligible files in the manifest.

    Skips binary files, empty files, and files
    exceeding the 50 MB size limit. Returns a flat list of all chunks.
    """
    chunks: list[LogChunk] = []
    for entry in manifest.entries:
        if entry.path.endswith(".csv"):
            from pipeline.csv_chunker import chunk_csv  # deferred: module added in Task 6
            chunks.extend(chunk_csv(bundle_root, entry, bundle_type=bundle_type))
        else:
            chunks.extend(chunk_file(bundle_root, entry, bundle_type=bundle_type))
    return chunks


def _detect_severity(text: str) -> str | None:
    """Return 'error', 'warning', or None based on keywords found in text."""
    if _ERROR_RE.search(text):
        return "error"
    if _WARNING_RE.search(text):
        return "warning"
    return None
