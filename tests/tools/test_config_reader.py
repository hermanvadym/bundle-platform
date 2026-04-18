"""Tests for bundle_platform.tools.config_reader."""

from pathlib import Path

import pytest

from bundle_platform.tools.config_reader import read_config, read_sos_command


@pytest.fixture()
def bundle_root(tmp_path: Path) -> Path:
    return tmp_path


class TestReadConfig:
    def test_happy_path(self, bundle_root: Path) -> None:
        (bundle_root / "etc").mkdir()
        f = bundle_root / "etc" / "hosts"
        f.write_text("127.0.0.1 localhost\n::1 localhost\n")

        result = read_config(bundle_root, "etc/hosts")

        assert "127.0.0.1 localhost" in result
        assert "::1 localhost" in result
        # Lines are numbered
        assert result.startswith("1:")

    def test_missing_file_returns_error_string(self, bundle_root: Path) -> None:
        result = read_config(bundle_root, "etc/nonexistent.conf")

        assert "not found" in result.lower()
        # Must not raise — tools always return strings
        assert isinstance(result, str)

    def test_line_cap_truncates_at_150(self, bundle_root: Path) -> None:
        (bundle_root / "etc").mkdir()
        f = bundle_root / "etc" / "big.conf"
        f.write_text("\n".join(f"line {i}" for i in range(1, 201)))  # 200 lines

        result = read_config(bundle_root, "etc/big.conf")

        lines = result.splitlines()
        # Last line should be the truncation marker, not line 200
        assert "truncated" in lines[-1].lower()
        # Exactly 150 content lines plus the marker = 151 total
        assert len(lines) == 151

    def test_path_traversal_rejected(self, bundle_root: Path) -> None:
        result = read_config(bundle_root, "../../etc/passwd")

        assert "outside bundle" in result.lower() or "error" in result.lower()


class TestReadSosCommand:
    def test_rhel_reads_from_sos_commands(self, bundle_root: Path) -> None:
        (bundle_root / "sos_commands").mkdir()
        (bundle_root / "sos_commands" / "uname").write_text("Linux host 5.14.0\n")

        result = read_sos_command(bundle_root, "uname", "rhel")

        assert "Linux host 5.14.0" in result

    def test_esxi_reads_from_commands(self, bundle_root: Path) -> None:
        (bundle_root / "commands").mkdir()
        (bundle_root / "commands" / "uname").write_text("VMkernel 7.0.3\n")

        result = read_sos_command(bundle_root, "uname", "esxi")

        assert "VMkernel 7.0.3" in result

    def test_rhel_does_not_read_commands_dir(self, bundle_root: Path) -> None:
        # Ensure RHEL never accidentally falls through to the ESXi directory
        (bundle_root / "commands").mkdir()
        (bundle_root / "commands" / "uname").write_text("should not appear\n")

        result = read_sos_command(bundle_root, "uname", "rhel")

        assert "not found" in result.lower()

    def test_missing_command_returns_error_string(self, bundle_root: Path) -> None:
        result = read_sos_command(bundle_root, "nonexistent_cmd", "rhel")

        assert isinstance(result, str)
        assert "not found" in result.lower()
