"""
Log analysis tools for sosreport bundles.

These functions implement targeted log extraction — the core of why this agent
uses far fewer tokens than loading entire files into context. Instead of reading
whole log files (which can be hundreds of MB), the agent calls these tools to
extract only the relevant lines.

All tools enforce output size caps to keep token usage predictable and bounded.
The caps are intentionally generous (200 lines) to avoid cutting off important
context, while still preventing a single tool call from consuming the entire
context window.
"""

import gzip
import re
from datetime import date, datetime
from pathlib import Path

from bundle_platform.shared.caps import TRUNCATION_MARKER
from bundle_platform.shared.timestamps import detect_timestamp_format as _detect_timestamp_format
from bundle_platform.shared.timestamps import extract_timestamp_str as _extract_timestamp_str
from bundle_platform.shared.timestamps import parse_timestamp as _parse_timestamp
from bundle_platform.tools.generic import FileManifest


def _open_file(path: Path):
    """Open a file for line iteration, transparently decompressing .gz files.

    Why: rotated log files in RHEL sosreports and ESXi bundles are commonly
    stored as .gz archives (e.g. messages-20260401.gz). Transparent decompression
    means tools work on both plain and rotated logs without caller changes.
    """
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


# Maximum lines returned by grep_log. Prevents a single grep on a high-frequency
# error pattern from flooding the context window with thousands of matching lines.
_MAX_GREP_LINES = 200

# Maximum lines returned by read_section. Enough to read any typical config file
# or log section in one call, while bounding the worst case.
_MAX_SECTION_LINES = 150

# Regex patterns for find_errors() (added in Task 4).
# These are defined here so Task 4 can import them from this module.
_ERROR_PATTERNS: dict[str, str] = {
    # Standard severity keywords plus OOM/kill events common in RHEL logs.
    # "Out of memory", "oom_kill", "Killed process" are high-signal OOM indicators
    # that the agent should surface during a broad error sweep.
    "error": (
        r"\b(error|ERROR|Error|failed|FAILED|Failed"
        r"|critical|CRITICAL|oom_kill|Out of memory|Killed process)\b"
    ),
    "warning": r"\b(warning|WARNING|Warning|warn|WARN)\b",
}

# Only these categories contain timestamped log entries worth sweeping for errors.
# Sweeping config files or sos_commands for "error" keywords would produce noise.
_LOG_CATEGORIES = {"system_logs", "audit", "kernel", "host_agent", "kvm_logs", "event_archive"}


def grep_log(
    bundle_root: Path | str,
    file_path: str,
    pattern: str,
    context_lines: int = 5,
) -> str:
    """
    Search a log file for a regex pattern and return matching lines with context.

    Why: Raw grep is far more token-efficient than reading entire files. A 50MB
    messages log might have 3 lines relevant to an OOM event — grep returns those
    3 lines plus context, while read_section would require knowing exact line numbers.

    How: Finds all matching line indices, then builds non-overlapping context windows
    around each match (merging windows that overlap). Returns numbered lines so the
    agent can use read_section() for more context if needed. Output is capped at
    _MAX_GREP_LINES (200) to prevent context overflow on high-frequency patterns.

    Search is case-insensitive (re.IGNORECASE) because log levels vary by source:
    "Error", "ERROR", "error" all appear in RHEL logs depending on the subsystem.

    Args:
        bundle_root:   Root directory of the unpacked bundle.
        file_path:     Relative path to the log file within the bundle.
        pattern:       Regex pattern to search for.
        context_lines: Number of lines to include before and after each match.
                       Default 5 gives enough context to understand most log events.

    Returns:
        Matched lines with context (numbered), separated by "---" between match groups.
        Returns a "No matches" message if nothing is found.
        Returns a "not found" message if the file doesn't exist.
    """
    bundle_root = Path(bundle_root)
    full_path = bundle_root / file_path
    if not full_path.resolve().is_relative_to(bundle_root.resolve()):
        return f"Error: path is outside bundle: {file_path}"
    if not full_path.exists():
        return f"File not found: {file_path}"

    try:
        with _open_file(full_path) as fh:
            lines = [line.rstrip("\n") for line in fh]
    except OSError:
        return f"File not found: {file_path}"

    # Find all line indices where the pattern matches
    try:
        match_indices = [
            i for i, line in enumerate(lines) if re.search(pattern, line, re.IGNORECASE)
        ]
    except re.error as exc:
        return f"Invalid regex pattern {pattern!r}: {exc}"
    if not match_indices:
        return f"No matches for '{pattern}' in {file_path}"

    # Build context windows around each match, then merge overlapping ones.
    # Without merging, two nearby matches would produce duplicate context lines.
    # Example: matches at lines 10 and 12 with context=3 would overlap at lines 9-13.
    ranges: list[tuple[int, int]] = []
    for idx in match_indices:
        start = max(0, idx - context_lines)
        end = min(len(lines), idx + context_lines + 1)
        if ranges and start <= ranges[-1][1]:
            # Overlaps with previous range — extend it rather than creating a new one
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))

    # Build content lines per range, then assemble with separators
    range_blocks: list[list[str]] = []
    total_content_lines = 0

    for start, end in ranges:
        block = [f"{start + i + 1}: {line}" for i, line in enumerate(lines[start:end])]
        # Apply cap — stop adding blocks once we'd exceed the limit
        if total_content_lines + len(block) > _MAX_GREP_LINES:
            remaining = _MAX_GREP_LINES - total_content_lines
            block = block[:remaining]
            range_blocks.append(block)
            total_content_lines += len(block)
            break
        range_blocks.append(block)
        total_content_lines += len(block)

    output_parts = []
    for block in range_blocks:
        output_parts.extend(block)
        output_parts.append("---")

    result = "\n".join(output_parts)
    if total_content_lines >= _MAX_GREP_LINES:
        total_range_lines = sum(end - start for start, end in ranges)
        result += "\n" + TRUNCATION_MARKER.format(shown=_MAX_GREP_LINES, total=total_range_lines)
    return result


def read_section(
    bundle_root: Path | str,
    file_path: str,
    start_line: int,
    end_line: int | None = None,
) -> str:
    """
    Read a specific line range from a file in the bundle.

    Why: When the agent already knows which lines are relevant (e.g. from a previous
    grep_log() call), read_section lets it fetch more context around those lines
    without re-running grep or reading the entire file.

    How: Lines are 1-indexed (matching standard editor line numbers and the output
    format of grep_log). The result is always capped at _MAX_SECTION_LINES (150)
    regardless of end_line, so the agent can't accidentally load a huge file.

    Args:
        bundle_root: Root directory of the unpacked bundle.
        file_path:   Relative path to the file within the bundle.
        start_line:  First line to read (1-indexed, inclusive).
        end_line:    Last line to read (1-indexed, inclusive). Optional.
                     If omitted, reads up to _MAX_SECTION_LINES (150) lines
                     starting from start_line.

    Returns:
        Numbered lines from the file. Includes a truncation notice if the
        requested range exceeds the 150-line cap.
        Returns a "not found" message if the file doesn't exist.
    """
    bundle_root = Path(bundle_root)
    if start_line < 1:
        return f"Invalid start_line {start_line}: must be >= 1"
    if end_line is not None and end_line < start_line:
        return f"Invalid range: end_line ({end_line}) < start_line ({start_line})"

    full_path = bundle_root / file_path
    if not full_path.resolve().is_relative_to(bundle_root.resolve()):
        return f"Error: path is outside bundle: {file_path}"
    if not full_path.exists():
        return f"File not found: {file_path}"

    try:
        with _open_file(full_path) as fh:
            lines = [line.rstrip("\n") for line in fh]
    except OSError:
        return f"File not found: {file_path}"

    # Convert from 1-indexed (user-facing) to 0-indexed (Python list)
    start = max(0, start_line - 1)

    # Determine the end, applying the hard cap in all cases.
    # min() ensures: (a) we don't exceed the requested end_line,
    #                (b) we don't exceed start + max cap,
    #                (c) we don't read past the end of the file.
    end = min(
        (end_line if end_line else start + _MAX_SECTION_LINES),
        start + _MAX_SECTION_LINES,
        len(lines),
    )

    result = "\n".join(f"{start + i + 1}: {line}" for i, line in enumerate(lines[start:end]))

    # Tell the agent if content was cut off so it knows to use a later start_line
    capped_by_limit = (end - start) == _MAX_SECTION_LINES and end < len(lines)
    if capped_by_limit:
        result += (
            f"\n... capped at {_MAX_SECTION_LINES} lines"
            f" (file has {len(lines)} lines, use a later start_line to read more)"
        )
    return result


_MAX_ERROR_RESULTS = 200
_MAX_CORRELATE_RESULTS = 200


def find_errors(
    bundle_root: Path | str,
    manifest: FileManifest,
    severity: str = "error",
    since: datetime | None = None,
    until: datetime | None = None,
    reference_date: date | None = None,
) -> str:
    """
    Sweep all log files in the bundle for error or warning entries.

    Why: When the user describes a problem without knowing where to look, this
    tool gives the agent a quick panoramic view of what went wrong across the
    entire bundle — without reading every file in full. It's the "start broad,
    then drill down" entry point for most diagnostic sessions.

    How: Iterates only over files in _LOG_CATEGORIES (system_logs, audit, kernel)
    to avoid noise from config files. Applies the appropriate regex from
    _ERROR_PATTERNS to each line. Returns results in "filepath:lineno: content"
    format so the agent can immediately use grep_log() or read_section() to
    get more context around any finding.

    The since/until parameters use numeric datetime comparison: each line's
    timestamp is parsed with the same format-detection used by
    correlate_timestamps(). Lines whose timestamp can't be parsed are always
    included (conservative: better to show a line than silently drop it).

    Args:
        bundle_root: Root directory of the unpacked bundle.
        manifest:    FileManifest — used to iterate only categorized log files.
        severity:    "error" or "warning" — selects which pattern to apply.
        since:       Exclude lines whose timestamp is before this datetime.
        until:       Exclude lines whose timestamp is after this datetime.

    Returns:
        Newline-separated "filepath:lineno: content" entries, capped at
        _MAX_ERROR_RESULTS. Returns a human-readable "No error entries found"
        message if nothing matches, so the agent gets clear feedback.
    """
    bundle_root = Path(bundle_root)
    pattern = _ERROR_PATTERNS.get(severity, _ERROR_PATTERNS["error"])
    results: list[str] = []

    for entry in manifest.entries:
        # Only sweep files from diagnostic log categories — config and sos_commands
        # files contain the word "error" in unrelated contexts (e.g. error_log paths)
        if entry.category not in _LOG_CATEGORIES:
            continue

        full_path = bundle_root / entry.path
        if not full_path.exists():
            continue

        try:
            with _open_file(full_path) as fh:
                lines = [line.rstrip("\n") for line in fh]
        except OSError:
            continue  # skip unreadable files rather than aborting the sweep

        for lineno, line in enumerate(lines, 1):
            if since is not None or until is not None:
                fmt = _detect_timestamp_format(line)
                ts_str = _extract_timestamp_str(line, fmt)
                if ts_str is not None:
                    line_dt = _parse_timestamp(ts_str, fmt, reference_date=reference_date)
                    if line_dt is not None:
                        if since is not None and line_dt < since:
                            continue
                        if until is not None and line_dt > until:
                            break

            if re.search(pattern, line):
                results.append(f"{entry.path}:{lineno}: {line}")
                if len(results) >= _MAX_ERROR_RESULTS:
                    break  # stop scanning this file once cap is hit

        if len(results) >= _MAX_ERROR_RESULTS:
            break  # stop scanning further files once global cap is hit

    if not results:
        return f"No {severity} entries found in log files."
    return "\n".join(results)


def correlate_timestamps(
    bundle_root: Path | str,
    file_paths: list[str],
    timestamp: str,
    window_seconds: int = 60,
    reference_date: date | None = None,
) -> str:
    """
    Find log lines within a time window of an anchor timestamp across multiple files.

    Why: Many RHEL and ESXi problems manifest as a cascade across files — an OOM
    kill in var/log/messages correlates with a kernel message in dmesg at the same
    time. This tool lets the agent say "show me what happened across these three
    files around 02:00" without knowing which exact log level recorded the event.

    Why window_seconds matters: a value of 60 captures events in the same minute;
    86400 captures a full day; 2592000 (~30 days) covers a month. The agent passes
    a value appropriate to the question being asked.

    How: Parses the anchor timestamp into a datetime, then scans each file line-by-
    line, parsing each line's timestamp and comparing the absolute delta against
    window_seconds. Lines without parseable timestamps are skipped. Output is capped
    per-file and globally.

    Args:
        bundle_root:    Root directory of the unpacked bundle.
        file_paths:     Relative paths of files to search.
        timestamp:      Anchor timestamp (ISO 8601, syslog, or bracket format).
                        Partial timestamps are NOT supported — use a full timestamp
                        string the parser can recognise.
        window_seconds: Radius of the time window in seconds (default 60).
                        Pass 86400 for a full day, 2592000 for ~30 days.

    Returns:
        Lines grouped by file with a header showing match count.
        Capped at _MAX_CORRELATE_RESULTS lines total.
        Returns a descriptive message if nothing matches or timestamps can't be parsed.
        Reports missing/unreadable files inline so the agent can adapt.
    """
    bundle_root = Path(bundle_root)
    # --- Parse the anchor timestamp ---
    anchor_dt: datetime | None = None
    anchor_fmt = _detect_timestamp_format(timestamp)
    if anchor_fmt != "unknown":
        anchor_dt = _parse_timestamp(timestamp, anchor_fmt, reference_date=reference_date)
    if anchor_dt is None:
        # Fallback: try all formats
        for fmt in ("iso8601", "syslog", "bracket"):
            anchor_dt = _parse_timestamp(timestamp, fmt, reference_date=reference_date)
            if anchor_dt is not None:
                break
    if anchor_dt is None:
        return (
            f"Cannot parse anchor timestamp: {timestamp!r}. "
            "Use ISO 8601 (e.g. '2026-04-15T02:00:00'), "
            "syslog ('Apr 15 02:00:00'), or bracket ('[2026-04-15 02:00:00') format."
        )

    _MAX_LINES_PER_FILE = 200
    results: list[str] = []
    total_matched = 0

    for file_path in file_paths:
        if total_matched >= _MAX_CORRELATE_RESULTS:
            results.append(
                f"(global cap of {_MAX_CORRELATE_RESULTS} lines reached"
                " — narrow the window or file list)"
            )
            break

        full_path = bundle_root / file_path
        if not full_path.resolve().is_relative_to(bundle_root.resolve()):
            results.append(f"# {file_path}: path outside bundle (skipped)")
            continue
        if not full_path.exists():
            results.append(f"# {file_path}: not found")
            continue

        # Detect timestamp format from the first non-empty line
        file_fmt = "unknown"
        try:
            with _open_file(full_path) as fh:
                for sample_line in fh:
                    stripped = sample_line.strip()
                    if stripped:
                        file_fmt = _detect_timestamp_format(stripped)
                        break
        except OSError as exc:
            results.append(f"# {file_path}: cannot read ({exc})")
            continue

        if file_fmt == "unknown":
            results.append(f"# {file_path}: no recognised timestamp format — skipping")
            continue

        # Collect lines within the time window
        matched: list[str] = []
        try:
            with _open_file(full_path) as fh:
                for lineno, raw in enumerate(fh, 1):
                    line = raw.rstrip("\n")
                    ts_str = _extract_timestamp_str(line, file_fmt)
                    if ts_str is None:
                        continue
                    line_dt = _parse_timestamp(ts_str, file_fmt, reference_date=reference_date)
                    if line_dt is None:
                        continue
                    if abs((line_dt - anchor_dt).total_seconds()) <= window_seconds:
                        matched.append(f"{lineno:>6}: {line}")
        except OSError as exc:
            results.append(f"# {file_path}: cannot read ({exc})")
            continue

        if matched:
            capped = matched[:_MAX_LINES_PER_FILE]
            results.append(f"# {file_path}: {len(matched)} line(s) within {window_seconds}s")
            results.extend(capped)
            total_matched += len(capped)
            if len(matched) > _MAX_LINES_PER_FILE:
                results.append(
                    TRUNCATION_MARKER.format(shown=_MAX_LINES_PER_FILE, total=len(matched))
                )
        else:
            results.append(f"# {file_path}: no lines within {window_seconds}s of {timestamp!r}")

    if not any(not r.startswith("#") for r in results):
        return (
            f"No lines found within {window_seconds}s of {timestamp!r} "
            f"in {len(file_paths)} file(s)."
        )
    return "\n".join(results)


def find_mentions(
    bundle_root: Path | str,
    keyword: str,
    file_paths: list[str],
    context_lines: int = 3,
) -> str:
    """Search multiple files for a keyword and aggregate results.

    Why: grep_log operates on one file at a time. When hunting for a PID, IP, or
    hostname across an entire bundle, the agent would need to issue dozens of
    grep_log calls. find_mentions batches that into one tool call with a single
    global output cap.
    """
    bundle_root = Path(bundle_root)
    parts: list[str] = []
    total_lines = 0

    for fp in file_paths:
        if total_lines >= _MAX_GREP_LINES:
            parts.append("... output cap reached, skipping remaining files")
            break
        raw = grep_log(bundle_root, fp, keyword, context_lines)
        header = f"# {fp}"
        # grep_log returns error/no-match strings when file is missing or no matches
        if raw.startswith("No matches"):
            parts.append(f"{header}: no matches")
            continue
        if "not found" in raw.lower() or raw.startswith("Error"):
            parts.append(f"{header}: {raw}")
            continue
        # Count content lines contributed by this file (exclude separators)
        file_content_lines = [
            ln
            for ln in raw.splitlines()
            if ln and not ln.startswith("---") and not ln.startswith("...")
        ]
        remaining = _MAX_GREP_LINES - total_lines
        if len(file_content_lines) > remaining:
            # Truncate: keep only `remaining` content lines worth of raw output
            truncated_lines: list[str] = []
            kept = 0
            for ln in raw.splitlines():
                if not ln or ln.startswith("---") or ln.startswith("..."):
                    truncated_lines.append(ln)
                else:
                    if kept >= remaining:
                        break
                    truncated_lines.append(ln)
                    kept += 1
            raw = (
                "\n".join(truncated_lines)
                + "\n"
                + TRUNCATION_MARKER.format(shown=kept, total=len(file_content_lines))
            )
            total_lines += kept
        else:
            total_lines += len(file_content_lines)
        parts.append(f"{header}\n{raw}")

    if not parts or all(": no matches" in p or "not found" in p.lower() for p in parts):
        return f"No matches for '{keyword}' in {len(file_paths)} file(s)."
    return "\n\n".join(parts)
