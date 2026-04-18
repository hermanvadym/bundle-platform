import os
from unittest.mock import patch

import pytest

from bundle_platform.llm import get_client


@patch.dict(os.environ, {"BUNDLE_PLATFORM_LLM": "unknown"})
def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        get_client()


@patch.dict(os.environ, {"BUNDLE_PLATFORM_LLM": "anthropic", "ANTHROPIC_API_KEY": "test-key"})
def test_factory_returns_anthropic_client():
    client = get_client()
    assert hasattr(client, "model_id")


@patch.dict(os.environ, {"BUNDLE_PLATFORM_LLM": "bedrock"})
@patch("bundle_platform.llm.bedrock.AnthropicBedrock")
def test_factory_returns_bedrock_client(mock_bedrock):
    client = get_client()
    assert client.model_id


@patch.dict(os.environ, {
    "BUNDLE_PLATFORM_LLM": "vertex",
    "BUNDLE_PLATFORM_VERTEX_REGION": "us-east5",
    "BUNDLE_PLATFORM_VERTEX_PROJECT": "proj",
})
@patch("bundle_platform.llm.vertex.AnthropicVertex")
def test_factory_returns_vertex_client(mock_vertex):
    client = get_client()
    assert client.model_id


@patch.dict(os.environ, {
    "BUNDLE_PLATFORM_LLM": "azure",
    "BUNDLE_PLATFORM_AZURE_ENDPOINT": "https://example.com/",
    "BUNDLE_PLATFORM_AZURE_API_KEY": "key",
})
def test_factory_returns_azure_client():
    client = get_client()
    assert client.model_id
