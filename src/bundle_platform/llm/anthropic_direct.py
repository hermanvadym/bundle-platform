"""Anthropic direct API backend for LLMClient.

Wraps the ``anthropic`` SDK and maps its response shape to the shared
``LLMResponse`` / ``LLMUsage`` dataclasses so the rest of the codebase
stays backend-agnostic.
"""
import os
from typing import Any, cast

import anthropic

from bundle_platform.llm.client import LLMResponse, LLMUsage


class AnthropicDirectClient:
    """LLMClient backed by the Anthropic messages API.

    API key is read from ``api_key`` kwarg or ``ANTHROPIC_API_KEY`` env var.
    Model is read from ``model_id`` kwarg or ``BUNDLE_PLATFORM_MODEL`` env var,
    defaulting to ``claude-sonnet-4-6``.
    """

    def __init__(self, api_key: str | None = None, model_id: str | None = None) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=key)
        self.model_id = model_id or os.environ.get(
            "BUNDLE_PLATFORM_MODEL", "claude-sonnet-4-6"
        )

    def complete(
        self,
        system: list[dict],
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Call the Anthropic messages endpoint and return a normalised LLMResponse."""
        # Cast list[dict] to Any: the SDK accepts these shapes at runtime, but its
        # typed overloads expect vendor-specific TypedDicts. Casting here keeps the
        # Protocol's generic list[dict] interface while satisfying the type checker.
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
                # cache_creation_input_tokens / cache_read_input_tokens are only
                # present when prompt caching is active; default to 0 otherwise.
                cache_creation_tokens=getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
                cache_read_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            ),
            raw=resp,
        )
