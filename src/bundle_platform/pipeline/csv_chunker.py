"""
CSV event archive chunker.

Parses timestamped CSV files (Windows Event Log exports, custom event archives)
into time-window LogChunk objects for embedding into the Qdrant vector store.

Chunking strategy: rows within a 5-minute window form one chunk. This preserves
temporal clustering needed for cross-file correlation — all events during an incident
stay in the same chunk for semantic retrieval.
"""

import csv
import io
from datetime import datetime, timedelta
from pathlib import Path

from bundle_platform.pipeline.chunker import LogChunk, _detect_severity
from bundle_platform.tools.generic import FileEntry

_TIMESTAMP_COLS: frozenset[str] = frozenset({"timecreated", "time", "timestamp", "datetime"})
_MESSAGE_COLS: frozenset[str] = frozenset({"message", "messages", "description", "details"})
_SEVERITY_COLS: frozenset[str] = frozenset({"leveldisplayname", "level", "severity", "type"})
_SOURCE_COLS: frozenset[str] = frozenset({"providername", "source", "channel"})
_EVENTID_COLS: frozenset[str] = frozenset({"id", "eventid", "event_id"})
_IP_COLS: frozenset[str] = frozenset({"ip", "ipaddress", "remoteaddress", "sourceip"})

_WINDOW_MINUTES = 5
_MAX_ROWS_PER_CHUNK = 500

_TS_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
)


def _find_col(fieldnames: list[str], candidates: frozenset[str]) -> str | None:
    for col in fieldnames:
        if col.lower() in candidates:
            return col
    return None


def _parse_ts(value: str) -> datetime | None:
    value = value.strip().rstrip("Z")
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _format_row(
    row: dict[str, str],
    ts_col: str,
    msg_col: str | None,
    sev_col: str | None,
    src_col: str | None,
    eid_col: str | None,
    ip_col: str | None,
) -> str:
    ts = row.get(ts_col, "")
    severity = (row.get(sev_col) or "").upper()[:8] if sev_col else ""
    source = row.get(src_col) or "" if src_col else ""
    event_id = row.get(eid_col) or "" if eid_col else ""
    message = row.get(msg_col) or "" if msg_col else ""
    ip = row.get(ip_col) or "" if ip_col else ""

    parts: list[str] = [ts]
    if severity:
        parts.append(severity)
    if event_id and source:
        parts.append(f"Event {event_id} ({source})")
    elif source:
        parts.append(f"({source})")
    if ip:
        parts.append(f"IP: {ip}")
    parts.append(f"— {message}")
    return "  ".join(p for p in parts if p)


def _make_chunk(
    window_rows: list[tuple[datetime, dict[str, str]]],
    bundle_root: Path,
    entry: FileEntry,
    bundle_type: str,
    ts_col: str,
    msg_col: str | None,
    sev_col: str | None,
    src_col: str | None,
    eid_col: str | None,
    ip_col: str | None,
) -> LogChunk:
    first_ts, last_ts = window_rows[0][0], window_rows[-1][0]
    header = f"[Event Archive — {first_ts.isoformat()} → {last_ts.isoformat()}]"
    lines = [header] + [
        _format_row(row, ts_col, msg_col, sev_col, src_col, eid_col, ip_col)
        for _, row in window_rows
    ]
    text = "\n".join(lines)
    return LogChunk(
        bundle_id=bundle_root.name,
        file_path=entry.path,
        category="event_archive",
        start_line=1,
        end_line=len(window_rows),
        text=text,
        severity=_detect_severity(text),
        bundle_type=bundle_type,
        timestamp_start=first_ts.isoformat(),
        timestamp_end=last_ts.isoformat(),
    )


def chunk_csv(
    bundle_root: Path,
    entry: FileEntry,
    bundle_type: str = "unknown",
) -> list[LogChunk]:
    """
    Parse a CSV file into time-window LogChunk objects.

    Returns an empty list if no timestamp column is found or the file is
    unreadable. Does not raise — errors are swallowed so the caller can
    continue processing remaining files.
    """
    path = bundle_root / entry.path
    try:
        # utf-8-sig strips the BOM that Excel adds to CSV exports
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return []

    try:
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            return []
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    except csv.Error:
        return []

    ts_col = _find_col(fieldnames, _TIMESTAMP_COLS)
    if not ts_col:
        return []

    msg_col = _find_col(fieldnames, _MESSAGE_COLS)
    sev_col = _find_col(fieldnames, _SEVERITY_COLS)
    src_col = _find_col(fieldnames, _SOURCE_COLS)
    eid_col = _find_col(fieldnames, _EVENTID_COLS)
    ip_col = _find_col(fieldnames, _IP_COLS)

    chunks: list[LogChunk] = []
    window_rows: list[tuple[datetime, dict[str, str]]] = []
    window_start: datetime | None = None

    for row in rows:
        ts = _parse_ts(row.get(ts_col, ""))
        if ts is None:
            continue
        if window_start is None:
            window_start = ts

        window_exceeded = ts - window_start > timedelta(minutes=_WINDOW_MINUTES)
        size_exceeded = len(window_rows) >= _MAX_ROWS_PER_CHUNK

        if (window_exceeded or size_exceeded) and window_rows:
            chunks.append(
                _make_chunk(window_rows, bundle_root, entry, bundle_type,
                            ts_col, msg_col, sev_col, src_col, eid_col, ip_col)
            )
            window_rows = []
            window_start = ts

        window_rows.append((ts, row))

    if window_rows:
        chunks.append(
            _make_chunk(window_rows, bundle_root, entry, bundle_type,
                        ts_col, msg_col, sev_col, src_col, eid_col, ip_col)
        )

    return chunks
