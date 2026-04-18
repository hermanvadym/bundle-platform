"""
Shared timestamp utilities for parsing log timestamps across RHEL and ESXi bundles.

Three formats appear across bundles:
  - ISO 8601: "2026-04-15T10:23:45Z"  (vmkernel, hostd, ESXi)
  - syslog:   "Apr 15 10:23:45"        (RHEL messages, ESXi syslog)
  - bracket:  "[2026-04-15 10:23:45]"  (vobd, some ESXi daemons)
"""

import re
from datetime import date, datetime, timedelta

ISO8601_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?")
SYSLOG_RE = re.compile(
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}"
)
BRACKET_RE = re.compile(r"\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}")

TIMESTAMP_DETECTORS: list[tuple[re.Pattern[str], str]] = [
    (ISO8601_RE, "iso8601"),
    (SYSLOG_RE, "syslog"),
    (BRACKET_RE, "bracket"),
]

PARSE_FORMATS: dict[str, list[str]] = {
    "iso8601": [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ],
    "syslog": ["%b %d %H:%M:%S", "%b  %d %H:%M:%S"],
    "bracket": ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"],
}


def detect_timestamp_format(line: str) -> str:
    """Return the timestamp format used in a log line.

    Returns one of: "iso8601", "syslog", "bracket", or "unknown".
    """
    for pattern, fmt in TIMESTAMP_DETECTORS:
        if pattern.search(line):
            return fmt
    return "unknown"


def extract_timestamp_str(line: str, fmt: str) -> str | None:
    """Extract the raw timestamp text from a log line for the given format.

    Bracket format strips the leading '[' before returning.
    Returns None if the format is unrecognised or no match found.
    """
    _re_map: dict[str, re.Pattern[str]] = {
        "iso8601": ISO8601_RE,
        "syslog": SYSLOG_RE,
        "bracket": BRACKET_RE,
    }
    pattern = _re_map.get(fmt)
    if pattern is None:
        return None
    m = pattern.search(line)
    if m is None:
        return None
    return m.group(0).lstrip("[")


def parse_timestamp(
    ts_str: str,
    fmt: str,
    reference_date: date | None = None,
) -> datetime | None:
    """Parse a timestamp string using the known format. Returns None on failure.

    Syslog timestamps omit the year.  Year is inferred as follows:
    - If reference_date is provided (e.g. bundle collection date): start with
      reference_date.year.  If the resulting datetime is more than 60 days after
      reference_date the log entry must be from the prior year (bundles cannot
      contain future-dated logs), so subtract one year.  60 days covers the
      typical Dec-to-Feb gap that the old "30 days from now" heuristic missed.
    - If reference_date is None: keep the legacy behaviour — use today's year and
      roll back one year if the result is >30 days in the future.
    """
    for date_fmt in PARSE_FORMATS.get(fmt, []):
        try:
            if fmt == "syslog":
                if reference_date is not None:
                    year = reference_date.year
                    dt = datetime.strptime(f"{year} {ts_str.strip()}", f"%Y {date_fmt}")
                    ref_dt = datetime(reference_date.year, reference_date.month, reference_date.day)
                    if dt > ref_dt + timedelta(days=60):
                        dt = dt.replace(year=year - 1)
                else:
                    year = datetime.now().year
                    dt = datetime.strptime(f"{year} {ts_str.strip()}", f"%Y {date_fmt}")
                    if dt > datetime.now() + timedelta(days=30):
                        dt = dt.replace(year=year - 1)
                return dt
            return datetime.strptime(ts_str.strip(), date_fmt)
        except ValueError:
            continue
    return None


def ts_to_float(ts: str | None) -> float | None:
    """Convert a raw log timestamp string to a Unix epoch float.

    Tries all three formats in order. Returns None if unparseable.
    Used by the RAG pipeline to store numeric timestamps in Qdrant payloads.
    """
    if ts is None:
        return None
    ts_clean = ts.lstrip("[").rstrip("]").strip()
    for fmt in ("iso8601", "bracket", "syslog"):
        result = parse_timestamp(ts_clean, fmt)
        if result is not None:
            return result.timestamp()
    return None
