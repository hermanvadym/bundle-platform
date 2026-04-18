"""AWS Bedrock backend for LLMClient.

Wraps ``anthropic.AnthropicBedrock`` and maps its response shape to the shared
``LLMResponse`` / ``LLMUsage`` dataclasses so the rest of the codebase stays
backend-agnostic.

Requires the ``bedrock`` optional dependency group:
    uv sync --extra bedrock
"""
import os
from typing import Any, cast

try:
    from anthropic import AnthropicBedrock
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Bedrock backend requires 'anthropic[bedrock]'. "
        "Install with: uv sync --extra bedrock"
    ) from exc

from bundle_platform.llm.client import LLMResponse, LLMUsage


class BedrockClient:
    """LLMClient backed by the Anthropic Bedrock messages API.

    AWS region is read from ``aws_region`` kwarg or ``AWS_REGION`` env var,
    defaulting to ``us-east-1``.  Model is read from ``model_id`` kwarg or
    ``BUNDLE_PLATFORM_MODEL`` env var.
    """

    def __init__(self, aws_region: str | None = None, model_id: str | None = None) -> None:
        region = aws_region or os.environ.get("AWS_REGION", "us-east-1")
        self._client = AnthropicBedrock(aws_region=region)
        self.model_id = model_id or os.environ.get(
            "BUNDLE_PLATFORM_MODEL", "anthropic.claude-sonnet-4-5-20251001-v1:0"
        )

    def complete(
        self,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Call the Bedrock messages endpoint and return a normalised LLMResponse."""
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
