"""Typed exceptions for the RAG pipeline.

Callers catch RagUnavailable to degrade gracefully to tool-only mode.
"""


class RagUnavailable(RuntimeError):
    """RAG pipeline cannot serve the current request.

    Raised by embedder or store when the required infrastructure is not
    reachable or initialized. Callers should fall back to tool-only mode.
    """
