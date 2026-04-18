"""Shared line-cap helper. All tool functions return capped strings so
the agent's context budget cannot be blown out by a single tool call.
"""

TRUNCATION_MARKER = "[truncated: showing {shown} of {total} lines]"


def cap_lines(text: str, limit: int) -> str:
    """Return `text` truncated to at most `limit` lines.

    When truncation occurs, appends a trailing TRUNCATION_MARKER line so the
    agent sees the cap was hit. No-op when the input is at or under the limit.
    """
    if not text:
        return text
    lines = text.splitlines()
    if len(lines) <= limit:
        return text
    kept = lines[:limit]
    kept.append(TRUNCATION_MARKER.format(shown=limit, total=len(lines)))
    return "\n".join(kept)
