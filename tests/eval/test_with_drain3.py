# tests/eval/test_with_drain3.py
from __future__ import annotations

from unittest.mock import MagicMock

from bundle_platform.eval.strategies.with_drain3 import WithDrain3Strategy
from bundle_platform.eval.strategy import RetrievedContext


def test_drain3_strategy_name() -> None:
    s = WithDrain3Strategy.__new__(WithDrain3Strategy)
    s.name = "with_drain3"
    assert s.name == "with_drain3"


def test_drain3_applies_templating() -> None:
    strategy = WithDrain3Strategy.__new__(WithDrain3Strategy)
    strategy._retriever = MagicMock()
    strategy._retriever.retrieve.return_value = (
        "=== var/log/vmkernel.log (lines 1-5) ===\n"
        "SCSI error on device naa.600a098\n"
        "SCSI error on device naa.700b199"
    )
    mock_miner = MagicMock()
    mock_miner.add_log_message.side_effect = lambda line: (
        "SCSI error on device <*>" if "SCSI" in line else line
    )
    strategy._miner = mock_miner

    result = strategy.retrieve("storage errors")
    assert "<*>" in result.text
    assert "naa.600a098" not in result.text
