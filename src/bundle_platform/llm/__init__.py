"""LLM backend abstraction layer.

Use ``get_client()`` to obtain the configured backend. The concrete backend is
selected at runtime via the ``BUNDLE_PLATFORM_LLM`` environment variable so that
callers remain backend-agnostic.
"""
import os

from bundle_platform.llm.client import LLMClient, LLMResponse, LLMUsage

__all__ = ["LLMClient", "LLMResponse", "LLMUsage", "get_client"]


def get_client() -> LLMClient:
    """Return the configured LLMClient based on BUNDLE_PLATFORM_LLM env.

    Defaults to ``"anthropic"`` when the variable is unset.
    Raises ``ValueError`` for unknown backend names so misconfiguration fails fast.
    """
    backend = os.environ.get("BUNDLE_PLATFORM_LLM", "anthropic")
    match backend:
        case "anthropic":
            from bundle_platform.llm.anthropic_direct import AnthropicDirectClient
            return AnthropicDirectClient()
        case "bedrock":
            from bundle_platform.llm.bedrock import BedrockClient
            return BedrockClient()
        case "vertex":
            from bundle_platform.llm.vertex import VertexClient
            return VertexClient()
        case "azure":
            from bundle_platform.llm.azure import AzureClient
            return AzureClient()
        case _:
            raise ValueError(f"Unknown LLM backend: {backend!r}")
