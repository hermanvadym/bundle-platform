"""Google Cloud Vertex AI backend for LLMClient.

Wraps ``anthropic.AnthropicVertex`` and maps its response shape to the shared
``LLMResponse`` / ``LLMUsage`` dataclasses so the rest of the codebase stays
backend-agnostic.

Requires the ``vertex`` optional dependency group:
    uv sync --extra vertex
"""
import os
from typing import Any, cast

try:
    from anthropic import AnthropicVertex
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Vertex backend requires 'anthropic[vertex]'. "
        "Install with: uv sync --extra vertex"
    ) from exc

from bundle_platform.llm.client import LLMResponse, LLMUsage


class VertexClient:
    """LLMClient backed by the Anthropic Vertex AI messages API.

    Region and project are read from kwargs or ``BUNDLE_PLATFORM_VERTEX_REGION``
    / ``BUNDLE_PLATFORM_VERTEX_PROJECT`` env vars (both required when not passed).
    Model is read from ``model_id`` kwarg or ``BUNDLE_PLATFORM_MODEL`` env var.
    """

    def __init__(
        self,
        region: str | None = None,
        project_id: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self._client = AnthropicVertex(
            region=region or os.environ["BUNDLE_PLATFORM_VERTEX_REGION"],
            project_id=project_id or os.environ["BUNDLE_PLATFORM_VERTEX_PROJECT"],
        )
        self.model_id = model_id or os.environ.get(
            "BUNDLE_PLATFORM_MODEL", "claude-sonnet-4-5@20251001"
        )

    def complete(
        self,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Call the Vertex AI messages endpoint and return a normalised LLMResponse."""
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
