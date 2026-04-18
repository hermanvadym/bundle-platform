from unittest.mock import MagicMock, patch

from bundle_platform.llm.anthropic_direct import AnthropicDirectClient
from bundle_platform.llm.client import LLMResponse


@patch("bundle_platform.llm.anthropic_direct.anthropic.Anthropic")
def test_complete_maps_response_shape(mock_cls):
    fake_resp = MagicMock()
    fake_resp.content = ["block1"]
    fake_resp.stop_reason = "end_turn"
    fake_resp.usage.input_tokens = 100
    fake_resp.usage.output_tokens = 50
    fake_resp.usage.cache_creation_input_tokens = 10
    fake_resp.usage.cache_read_input_tokens = 5
    mock_cls.return_value.messages.create.return_value = fake_resp

    client = AnthropicDirectClient(api_key="test", model_id="claude-sonnet-4-6")
    resp = client.complete(system=[], messages=[], tools=[])

    assert isinstance(resp, LLMResponse)
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 100
    assert resp.usage.cache_creation_tokens == 10
    assert resp.usage.cache_read_tokens == 5
