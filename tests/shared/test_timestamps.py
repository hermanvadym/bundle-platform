from bundle_platform.shared.timestamps import (
    detect_timestamp_format,
    extract_timestamp_str,
    ts_to_float,
)


def test_detect_iso8601():
    assert detect_timestamp_format("2026-04-15T02:31:00") == "iso8601"


def test_detect_syslog():
    assert detect_timestamp_format("Apr 15 02:31:00") == "syslog"


def test_detect_bracket():
    assert detect_timestamp_format("[2026-04-15 02:31:00]") == "bracket"


def test_detect_unknown():
    assert detect_timestamp_format("no timestamp here") == "unknown"


def test_extract_iso8601():
    line = "2026-04-15T02:31:00 kernel: something happened"
    result = extract_timestamp_str(line, "iso8601")
    assert result == "2026-04-15T02:31:00"


def test_ts_to_float_iso8601():
    ts = ts_to_float("2026-04-15T02:31:00")
    assert isinstance(ts, float)
    assert ts > 0
