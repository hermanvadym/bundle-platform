from __future__ import annotations

import sys
import tarfile
from unittest.mock import patch

import pytest

from bundle_platform.eval.cli import _extract_archive, _is_archive, main


def test_cli_requires_subcommand(capsys: pytest.CaptureFixture) -> None:
    with patch.object(sys, "argv", ["bundle-platform-eval"]):
        with pytest.raises(SystemExit):
            main()


def test_cli_run_with_empty_golden_returns_1(tmp_path: pytest.TempPathFactory) -> None:
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    result = main(["run", "--bundle", str(bundle_dir), "--golden", str(golden_dir)])
    assert result == 1


def test_is_archive_detects_tgz(tmp_path: pytest.TempPathFactory) -> None:
    assert _is_archive(tmp_path / "bundle.tgz")
    assert _is_archive(tmp_path / "bundle.tar.gz")
    assert _is_archive(tmp_path / "bundle.tar.xz")
    assert _is_archive(tmp_path / "bundle.zip")
    assert not _is_archive(tmp_path / "bundle")
    assert not _is_archive(tmp_path / "bundle.log")


def test_extract_archive_tgz(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Build a small .tgz with a single file inside a subdirectory (realistic bundle layout)
    bundle_src = tmp_path / "src"
    (bundle_src / "var" / "log").mkdir(parents=True)
    (bundle_src / "var" / "log" / "vmkernel.log").write_text("kernel log", encoding="utf-8")

    archive = tmp_path / "bundle.tgz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(bundle_src, arcname="bundle")

    cache_dir = tmp_path / "cache"
    monkeypatch.setattr("bundle_platform.eval.cli._CACHE_DIR", cache_dir)

    result = _extract_archive(archive)

    assert result.is_dir()
    assert (result / "var" / "log" / "vmkernel.log").read_text(encoding="utf-8") == "kernel log"


def test_extract_archive_uses_cache_on_second_call(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_src = tmp_path / "src"
    bundle_src.mkdir()
    (bundle_src / "file.log").write_text("data", encoding="utf-8")

    archive = tmp_path / "bundle.tgz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(bundle_src, arcname="bundle")

    cache_dir = tmp_path / "cache"
    monkeypatch.setattr("bundle_platform.eval.cli._CACHE_DIR", cache_dir)

    first = _extract_archive(archive)
    second = _extract_archive(archive)

    assert first == second
