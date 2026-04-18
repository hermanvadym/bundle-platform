from unittest.mock import MagicMock, patch

from bundle_platform.llm.bedrock import BedrockClient


@patch("bundle_platform.llm.bedrock.AnthropicBedrock")
def test_bedrock_complete_maps_usage(mock_cls):
    fake_resp = MagicMock()
    fake_resp.content = []
    fake_resp.stop_reason = "end_turn"
    fake_resp.usage.input_tokens = 42
    fake_resp.usage.output_tokens = 7
    fake_resp.usage.cache_creation_input_tokens = 0
    fake_resp.usage.cache_read_input_tokens = 0
    mock_cls.return_value.messages.create.return_value = fake_resp

    client = BedrockClient(
        aws_region="us-east-1",
        model_id="anthropic.claude-sonnet-4-5-20251001-v1:0",
    )
    resp = client.complete(system=[], messages=[], tools=[])
    assert resp.usage.input_tokens == 42
    assert resp.stop_reason == "end_turn"


@patch("bundle_platform.llm.bedrock.AnthropicBedrock")
def test_bedrock_none_stop_reason(mock_cls):
    fake = MagicMock()
    fake.content = []
    fake.stop_reason = None
    fake.usage.input_tokens = 1
    fake.usage.output_tokens = 1
    fake.usage.cache_creation_input_tokens = 0
    fake.usage.cache_read_input_tokens = 0
    mock_cls.return_value.messages.create.return_value = fake
    client = BedrockClient(aws_region="us-east-1", model_id="m")
    assert client.complete(system=[], messages=[], tools=[]).stop_reason == "end_turn"
