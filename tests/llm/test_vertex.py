from unittest.mock import MagicMock, patch

from bundle_platform.llm.vertex import VertexClient


@patch("bundle_platform.llm.vertex.AnthropicVertex")
def test_vertex_complete_maps_usage(mock_cls):
    fake = MagicMock()
    fake.content = []
    fake.stop_reason = "end_turn"
    fake.usage.input_tokens = 3
    fake.usage.output_tokens = 1
    fake.usage.cache_creation_input_tokens = 0
    fake.usage.cache_read_input_tokens = 0
    mock_cls.return_value.messages.create.return_value = fake

    client = VertexClient(
        region="us-east5",
        project_id="my-proj",
        model_id="claude-sonnet-4-5@20251001",
    )
    resp = client.complete(system=[], messages=[], tools=[])
    assert resp.usage.input_tokens == 3
    assert resp.stop_reason == "end_turn"


@patch("bundle_platform.llm.vertex.AnthropicVertex")
def test_vertex_none_stop_reason(mock_cls):
    fake = MagicMock()
    fake.content = []
    fake.stop_reason = None
    fake.usage.input_tokens = 1
    fake.usage.output_tokens = 1
    fake.usage.cache_creation_input_tokens = 0
    fake.usage.cache_read_input_tokens = 0
    mock_cls.return_value.messages.create.return_value = fake
    client = VertexClient(region="us-east5", project_id="p", model_id="m")
    assert client.complete(system=[], messages=[], tools=[]).stop_reason == "end_turn"
