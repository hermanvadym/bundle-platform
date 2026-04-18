from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from bundle_platform.eval.cli import main


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
