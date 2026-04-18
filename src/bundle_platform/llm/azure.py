"""Azure-hosted Anthropic backend for LLMClient.

Uses the standard ``anthropic`` SDK with a custom ``base_url`` pointing at the
Azure endpoint — no extra SDK package needed beyond the base ``anthropic``
dependency already in pyproject.toml.
"""
import os
from typing import Any, cast

import anthropic

from bundle_platform.llm.client import LLMResponse, LLMUsage


class AzureClient:
    """LLMClient backed by an Azure-hosted Anthropic endpoint.

    Endpoint and API key are read from kwargs or ``BUNDLE_PLATFORM_AZURE_ENDPOINT``
    / ``BUNDLE_PLATFORM_AZURE_API_KEY`` env vars (both required when not passed).
    Model is read from ``model_id`` kwarg or ``BUNDLE_PLATFORM_MODEL`` env var.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        model_id: str | None = None,
    ) -> None:
        base_url = endpoint or os.environ["BUNDLE_PLATFORM_AZURE_ENDPOINT"]
        key = api_key or os.environ["BUNDLE_PLATFORM_AZURE_API_KEY"]
        self._client = anthropic.Anthropic(base_url=base_url, api_key=key)
        self.model_id = model_id or os.environ.get(
            "BUNDLE_PLATFORM_MODEL", "claude-sonnet-4-6"
        )

    def complete(
        self,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Call the Azure Anthropic endpoint and return a normalised LLMResponse."""
        resp = self._client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            system=cast(Any, system),
            messages=cast(Any, messages),
            tools=cast(Any, tools),
        )
        return LLMResponse(
            content=list(resp.content),
            stop_reason=resp.stop_reason or "end_turn",
            usage=LLMUsage(
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                cache_creation_tokens=getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
                cache_read_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            ),
            raw=resp,
        )
