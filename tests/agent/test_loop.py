"""Tests for dispatch_tool() in agent/loop.py."""

from pathlib import Path
from unittest.mock import patch

from bundle_platform.agent.accounting import SessionStats
from bundle_platform.agent.loop import dispatch_tool
from bundle_platform.tools.generic import FileEntry, FileManifest


def _manifest(tmp_path: Path) -> FileManifest:
    return FileManifest(
        bundle_root=tmp_path,
        entries=[FileEntry(path="var/log/messages", size_bytes=100, category="system_logs")],
        total_chars=100,
    )


def _stats() -> SessionStats:
    return SessionStats()


def test_list_files_returns_string(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    stats = _stats()
    with patch("bundle_platform.tools.generic.list_files", return_value="file list") as mock_fn:
        result = dispatch_tool("list_files", {}, manifest, tmp_path, stats)
    mock_fn.assert_called_once()
    assert isinstance(result, str)
    assert stats.tool_calls == 1


def test_grep_log_returns_string(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    stats = _stats()
    with patch("bundle_platform.tools.analysis.grep_log", return_value="match line") as mock_fn:
        result = dispatch_tool(
            "grep_log",
            {"file_path": "var/log/messages", "pattern": "error"},
            manifest,
            tmp_path,
            stats,
        )
    mock_fn.assert_called_once()
    assert result == "match line"
    assert "var/log/messages" in stats.files_touched
    assert stats.tool_calls == 1


def test_unknown_tool_returns_error_string(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    stats = _stats()
    result = dispatch_tool("nonexistent_tool", {}, manifest, tmp_path, stats)
    assert "Unknown tool" in result
    assert "nonexistent_tool" in result
    # Must return a string, not raise
    assert isinstance(result, str)
    assert stats.tool_calls == 1


def test_read_sos_command_tries_both_prefixes(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    stats = _stats()
    # Neither prefix exists on disk — should get not-found message, not raise
    result = dispatch_tool(
        "read_sos_command",
        {"command_name": "df"},
        manifest,
        tmp_path,
        stats,
    )
    assert isinstance(result, str)
    assert "df" in result
    assert stats.tool_calls == 1


def test_read_sos_command_finds_sos_commands_prefix(tmp_path: Path) -> None:
    sos_dir = tmp_path / "sos_commands"
    sos_dir.mkdir()
    (sos_dir / "df").write_text("Filesystem 1K-blocks\n/dev/sda1 100000\n")

    manifest = _manifest(tmp_path)
    stats = _stats()
    result = dispatch_tool(
        "read_sos_command",
        {"command_name": "df"},
        manifest,
        tmp_path,
        stats,
    )
    assert "Filesystem" in result
    assert "sos_commands/df" in stats.files_touched
