"""LLMClient Protocol and shared response dataclasses.

All LLM backends must implement LLMClient. Callers depend only on this module —
never on a concrete backend — so backends can be swapped without touching agent code.
"""
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class LLMUsage:
    """Token-usage counters returned by every LLM call.

    Cache fields default to 0 for backends that do not support prompt caching.
    """

    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass
class LLMResponse:
    """Normalised response returned by every LLMClient backend."""

    content: list[Any]  # list of Anthropic-compatible content blocks
    stop_reason: str    # "end_turn" | "tool_use" | "max_tokens"
    usage: LLMUsage
    # Backend-native response kept for debugging; excluded from repr to avoid noise.
    raw: Any = field(default=None, repr=False)


@runtime_checkable
class LLMClient(Protocol):
    """Protocol that all LLM backend clients must satisfy.

    Implementations must expose ``model_id`` as an instance attribute and
    implement ``complete()`` with the signature below.
    """

    model_id: str

    def complete(
        self,
        system: list[dict],
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a request to the LLM and return a normalised LLMResponse."""
        ...
