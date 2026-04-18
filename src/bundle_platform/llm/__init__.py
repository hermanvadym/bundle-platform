"""LLM backend abstraction layer.

Use ``get_client()`` to obtain the configured backend. The concrete backend is
selected at runtime via the ``BUNDLE_PLATFORM_LLM`` environment variable so that
callers remain backend-agnostic.
"""
from bundle_platform.llm.client import LLMClient, LLMResponse, LLMUsage

__all__ = ["LLMClient", "LLMResponse", "LLMUsage", "get_client"]


def get_client() -> LLMClient:
    """Return the configured LLMClient based on BUNDLE_PLATFORM_LLM env.

    Defaults to ``"anthropic"`` when the variable is unset.
    Raises ``ValueError`` for unknown backend names so misconfiguration fails fast.
    """
    import os

    backend = os.environ.get("BUNDLE_PLATFORM_LLM", "anthropic")
    if backend == "anthropic":
        from bundle_platform.llm.anthropic_direct import AnthropicDirectClient

        return AnthropicDirectClient()
    raise ValueError(f"Unknown LLM backend: {backend!r}")
