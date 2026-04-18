# tests/pipeline/test_template_miner.py
from __future__ import annotations

from unittest.mock import MagicMock

from bundle_platform.pipeline.template_miner import TemplateMinerWrapper


def test_add_log_message_returns_template() -> None:
    wrapper = TemplateMinerWrapper.__new__(TemplateMinerWrapper)
    mock_miner = MagicMock()
    mock_miner.add_log_message.return_value = {"template_mined": "SCSI error on device <*>"}
    wrapper._miner = mock_miner

    result = wrapper.add_log_message("SCSI error on device naa.600a098")
    assert result == "SCSI error on device <*>"


def test_add_log_message_falls_back_on_no_template() -> None:
    wrapper = TemplateMinerWrapper.__new__(TemplateMinerWrapper)
    mock_miner = MagicMock()
    mock_miner.add_log_message.return_value = {}
    wrapper._miner = mock_miner

    original = "unparseable line"
    assert wrapper.add_log_message(original) == original
