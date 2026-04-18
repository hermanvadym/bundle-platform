"""
Screenshot image describer for the RAG preprocessing pipeline.

Sends each screenshot in the bundle to the Claude vision API at preprocessing
time. The returned text description is stored as a LogChunk in Qdrant, making
screenshot content retrievable via semantic search alongside log evidence.

Why at preprocessing time: screenshots placed in a bundle folder are always
relevant (engineers don't include irrelevant screenshots). Pre-describing them
once pays ~$0.01-0.02 per image and produces zero cost per conversation turn.
"""

import base64
import os
from pathlib import Path
from typing import Literal

import anthropic
from anthropic.types import Base64ImageSourceParam, ImageBlockParam, MessageParam, TextBlockParam

from bundle_platform.pipeline.chunker import LogChunk, _detect_severity
from bundle_platform.tools.generic import FileManifest

_SupportedMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]

_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})
_MIME_TYPES: dict[str, _SupportedMediaType] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_VISION_PROMPT = (
    "You are analysing a diagnostic screenshot from a systems engineer.\n"
    "Extract ALL of the following that are visible:\n"
    "- Error codes, error messages, exception types, fault addresses\n"
    "- Dialog/window titles and application names\n"
    "- Software versions, patch levels, update KB numbers\n"
    "- Stack traces or technical detail text\n"
    "- Any timestamps or clocks visible on screen\n"
    "- System state indicators (progress bars, % complete, status text)\n"
    "- IP addresses, hostnames, service names mentioned\n"
    "Return a concise structured description. "
    "If you see none of the above, describe what is shown in one sentence."
)


def describe_images(
    manifest: FileManifest,
    bundle_root: Path,
    bundle_type: str = "unknown",
) -> list[LogChunk]:
    """
    Describe all screenshot-category files via the Claude vision API.

    Returns an empty list if no API key is set or no screenshots are present.
    Individual image failures are skipped silently so one bad image does not
    abort the entire preprocessing run.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return []

    image_entries = [e for e in manifest.entries if e.category == "screenshots"]
    if not image_entries:
        return []

    client = anthropic.Anthropic(api_key=api_key)
    chunks: list[LogChunk] = []

    print(f"  Describing {len(image_entries)} screenshot(s)...", flush=True)

    for entry in image_entries:
        path = bundle_root / entry.path
        suffix = Path(entry.path).suffix.lower()
        media_type = _MIME_TYPES.get(suffix)
        if not media_type:
            continue

        try:
            image_bytes = path.read_bytes()
        except OSError:
            continue

        encoded = base64.standard_b64encode(image_bytes).decode("ascii")

        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=400,
                messages=[
                    MessageParam(
                        role="user",
                        content=[
                            ImageBlockParam(
                                type="image",
                                source=Base64ImageSourceParam(
                                    type="base64",
                                    media_type=media_type,
                                    data=encoded,
                                ),
                            ),
                            TextBlockParam(type="text", text=_VISION_PROMPT),
                        ],
                    )
                ],
            )
        except anthropic.APIError:
            continue

        description = "".join(
            str(block.text) for block in response.content if hasattr(block, "text")
        )
        if not description:
            continue

        text = f"Screenshot: {Path(entry.path).name}\n{description}"
        chunks.append(
            LogChunk(
                bundle_id=bundle_root.name,
                file_path=entry.path,
                category="screenshots",
                start_line=1,
                end_line=1,
                text=text,
                severity=_detect_severity(text),
                bundle_type=bundle_type,
                timestamp_start=None,
                timestamp_end=None,
            )
        )

    return chunks
