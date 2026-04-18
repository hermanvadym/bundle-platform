from unittest.mock import MagicMock

import anthropic

from bundle_platform.agent.accounting import SessionStats, price_for


def test_price_for_sonnet():
    p = price_for("claude-sonnet-4-6")
    assert p.input_per_mtok == 3.00
    assert p.output_per_mtok == 15.00
    assert p.cache_write_per_mtok == 3.75
    assert p.cache_read_per_mtok == 0.30


def test_price_for_unknown_raises():
    import pytest
    with pytest.raises(KeyError, match="no pricing configured"):
        price_for("not-a-model")


def test_session_stats_update():
    stats = SessionStats()
    usage = MagicMock(spec=anthropic.types.Usage)
    usage.input_tokens = 100
    usage.output_tokens = 50
    usage.cache_creation_input_tokens = 10
    usage.cache_read_input_tokens = 200
    stats.update(usage)
    assert stats.input_tokens == 100
    assert stats.output_tokens == 50
    assert stats.cache_creation_tokens == 10
    assert stats.cache_read_tokens == 200


def test_session_stats_total_cost():
    stats = SessionStats()
    stats.input_tokens = 1_000_000
    stats.output_tokens = 1_000_000
    cost = stats.total_cost("claude-sonnet-4-6")
    # 1M input @ $3/Mtok + 1M output @ $15/Mtok = $18
    assert abs(cost - 18.0) < 0.01
