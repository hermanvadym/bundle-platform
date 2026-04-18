from unittest.mock import MagicMock, patch

from bundle_platform.llm.azure import AzureClient


@patch("bundle_platform.llm.azure.anthropic.Anthropic")
def test_azure_uses_custom_base_url(mock_cls):
    fake = MagicMock()
    fake.content = []
    fake.stop_reason = "end_turn"
    fake.usage.input_tokens = 1
    fake.usage.output_tokens = 1
    fake.usage.cache_creation_input_tokens = 0
    fake.usage.cache_read_input_tokens = 0
    mock_cls.return_value.messages.create.return_value = fake

    client = AzureClient(
        endpoint="https://example.openai.azure.com/",
        api_key="azure-key",
        model_id="claude-sonnet-4-6",
    )
    client.complete(system=[], messages=[], tools=[])
    mock_cls.assert_called_once()
    kwargs = mock_cls.call_args.kwargs
    assert kwargs["base_url"].startswith("https://example.openai.azure.com")
    assert kwargs["api_key"] == "azure-key"


@patch("bundle_platform.llm.azure.anthropic.Anthropic")
def test_azure_none_stop_reason(mock_cls):
    fake = MagicMock()
    fake.content = []
    fake.stop_reason = None
    fake.usage.input_tokens = 1
    fake.usage.output_tokens = 1
    fake.usage.cache_creation_input_tokens = 0
    fake.usage.cache_read_input_tokens = 0
    mock_cls.return_value.messages.create.return_value = fake
    client = AzureClient(endpoint="https://example.com/", api_key="k", model_id="m")
    assert client.complete(system=[], messages=[], tools=[]).stop_reason == "end_turn"
