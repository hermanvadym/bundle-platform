"""
Agent loop and tool dispatch for bundle-platform.

This module is the core of the application. It contains three main pieces:

1. TOOLS + dispatch_tool() — the tool definitions that Claude receives and the
   dispatcher that maps tool names to Python function calls.

2. run_session() — the interactive agent loop that sends messages to Claude,
   handles tool calls, and collects stats.

3. run_rag_session() — like run_session() but seeds each turn with RAG context.
"""

import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Protocol

import anthropic
from anthropic.types import MessageParam, ToolParam

from bundle_platform.agent.accounting import SessionStats
from bundle_platform.pipeline.config import load_api_key
from bundle_platform.pipeline.exceptions import RagUnavailable
from bundle_platform.tools import analysis as tools_logs
from bundle_platform.tools import config_reader as tools_config
from bundle_platform.tools import generic as tools_bundle
from bundle_platform.tools.generic import FileManifest


class _RetrieverProtocol(Protocol):
    def retrieve(
        self, question: str, time_window: tuple[datetime | None, datetime | None] | None = None
    ) -> str: ...


# The tool definitions sent to Claude with every API call.
# Each entry is a JSON Schema description of one callable tool.
# The "description" field is what Claude reads to decide WHICH tool to call —
# it must be precise enough to guide correct selection.
TOOLS: list[ToolParam] = [
    {
        "name": "list_files",
        "description": (
            "Filter the bundle file index by glob pattern or category. "
            "Call this first to orient yourself before accessing file content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match file paths (e.g. 'var/log/*')",
                },
                "category": {
                    "type": "string",
                    "description": (
                        "RHEL categories: system_logs, audit, sos_commands, "
                        "kernel, storage, network, config. "
                        "ESXi categories: system_logs, host_agent, storage, "
                        "network, vm_logs, commands, config."
                    ),
                    "enum": [
                        "system_logs", "audit", "sos_commands",
                        "kernel", "storage", "network", "config", "other",
                        "host_agent", "vm_logs", "commands",
                    ],
                },
            },
        },
    },
    {
        "name": "grep_log",
        "description": "Search a file for a regex pattern, returning matching lines with context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Relative path within the bundle"},
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "context_lines": {
                    "type": "integer",
                    "description": "Lines of context around each match (default 5)",
                },
            },
            "required": ["file_path", "pattern"],
        },
    },
    {
        "name": "read_section",
        "description": "Read a slice of a file (1-indexed lines). Always capped at 150 lines.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {
                    "type": "integer",
                    "description": "Optional end line (default: start_line + 150)",
                },
            },
            "required": ["file_path", "start_line"],
        },
    },
    {
        "name": "find_errors",
        "description": "Cross-file sweep for error or warning entries across all log files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "severity": {
                    "type": "string",
                    "enum": ["error", "warning"],
                    "description": "default: error",
                },
                "since": {"type": "string", "description": "Timestamp prefix to filter from"},
                "until": {"type": "string", "description": "Timestamp prefix to filter to"},
            },
        },
    },
    {
        "name": "correlate_timestamps",
        "description": "Find events near a timestamp string across multiple files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Relative file paths to search",
                },
                "timestamp": {
                    "type": "string",
                    "description": "Timestamp string to search for (partial match)",
                },
                "window_seconds": {
                    "type": "integer",
                    "description": "Context window in seconds (default 60)",
                },
            },
            "required": ["file_paths", "timestamp"],
        },
    },
    {
        "name": "read_sos_command",
        "description": (
            "Read the captured output of a sos command from sos_commands/ directory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command_name": {
                    "type": "string",
                    "description": (
                        "Command name as it appears in sos_commands/ (e.g. 'df', 'lsblk')"
                    ),
                },
            },
            "required": ["command_name"],
        },
    },
    {
        "name": "find_mentions",
        "description": (
            "Search multiple files for a keyword and aggregate matching lines. "
            "Use when hunting for a PID, IP address, hostname, or identifier "
            "across several log files at once."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Keyword or regex to search for (case-insensitive)",
                },
                "file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Relative file paths to search within the bundle",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Lines of context around each match (default: 3)",
                },
            },
            "required": ["keyword", "file_paths"],
        },
    },
]


def dispatch_tool(
    name: str,
    inputs: dict,
    manifest: FileManifest,
    bundle_root: Path,
    stats: SessionStats,
    time_window: tuple[datetime | None, datetime | None] | None = None,
    reference_date: date | None = None,
) -> str:
    """
    Route a Claude tool call to the corresponding Python function.

    Why: Claude returns tool calls as (name, inputs) pairs. This function is the
    single dispatch point that maps those to actual function calls, tracks which
    files were read, and provides a safe fallback for unknown tool names.

    Why match statement: Python 3.10+ structural matching is more readable than
    a dict of callables here because each arm needs different argument assembly —
    not just a different function name.

    Args:
        name:        Tool name as returned by Claude (must match a TOOLS entry).
        inputs:      Dict of tool arguments as returned by Claude.
        manifest:    FileManifest for list_files() and find_errors().
        bundle_root: Bundle root path for all file-reading tools.
        stats:       SessionStats to update (tool_calls + files_touched).

    Returns:
        String result from the tool. Claude receives this as the tool_result
        content and uses it to continue reasoning.
    """
    # Count every dispatch attempt, including unknown tools and errors
    stats.tool_calls += 1
    stats.turn_tool_calls += 1

    match name:
        case "list_files":
            return tools_bundle.list_files(
                manifest,
                pattern=inputs.get("pattern"),
                category=inputs.get("category"),
            )
        case "grep_log":
            path = inputs["file_path"]
            stats.files_touched.add(path)
            stats.turn_files.add(path)
            return tools_logs.grep_log(
                bundle_root, path, inputs["pattern"], inputs.get("context_lines", 5)
            )
        case "read_section":
            path = inputs["file_path"]
            stats.files_touched.add(path)
            stats.turn_files.add(path)
            return tools_logs.read_section(
                bundle_root, path, inputs["start_line"], inputs.get("end_line")
            )
        case "find_errors":
            since: datetime | None = None
            until: datetime | None = None
            if time_window:
                since, until = time_window
            return tools_logs.find_errors(
                bundle_root,
                manifest,
                inputs.get("severity", "error"),
                since,
                until,
                reference_date=reference_date,
            )
        case "correlate_timestamps":
            paths = inputs["file_paths"]
            stats.files_touched.update(paths)
            stats.turn_files.update(paths)
            return tools_logs.correlate_timestamps(
                bundle_root,
                paths,
                inputs["timestamp"],
                inputs.get("window_seconds", 60),
                reference_date=reference_date,
            )
        case "read_sos_command":
            command_name = inputs["command_name"]
            # RHEL stores command output under sos_commands/, ESXi under commands/.
            # Try both so the tool works transparently for either bundle type.
            for prefix in ("sos_commands", "commands"):
                if (bundle_root / prefix / command_name).exists():
                    path = f"{prefix}/{command_name}"
                    stats.files_touched.add(path)
                    stats.turn_files.add(path)
                    return tools_config.read_config(bundle_root, path)
            return (
                f"Command output not found: '{command_name}' "
                "(checked sos_commands/ and commands/)"
            )
        case "find_mentions":
            paths = inputs["file_paths"]
            stats.files_touched.update(paths)
            stats.turn_files.update(paths)
            return tools_logs.find_mentions(
                bundle_root,
                inputs["keyword"],
                paths,
                inputs.get("context_lines", 3),
            )
        case _:
            # Return a message (not raise) so Claude sees the error and can correct itself
            return f"Unknown tool: {name}"


def _format_index(manifest: FileManifest) -> str:
    """
    Format the file manifest as a structured text block for the system context.

    Why: The agent needs a complete map of the bundle before asking questions.
    Sending the index as a cached system context block means it counts toward
    the prompt cache hit rate, reducing cost on every follow-up question in the
    session.

    How: Groups files by category, lists up to 50 per category (truncating with
    a count for larger categories) to keep the index readable without being huge.

    Args:
        manifest: The bundle's FileManifest.

    Returns:
        Formatted string for inclusion as a cached system context block.
    """
    by_category: dict[str, list[str]] = {}
    for entry in manifest.entries:
        by_category.setdefault(entry.category, []).append(
            f"  {entry.path} ({entry.size_bytes} bytes)"
        )

    lines = [f"Bundle contains {len(manifest.entries)} files:\n"]
    for cat in sorted(by_category):
        paths = by_category[cat]
        lines.append(f"[{cat}] ({len(paths)} files)")
        # Cap at 50 per category — sos_commands/ can have 200+ entries
        lines.extend(paths[:50])
        if len(paths) > 50:
            lines.append(f"  ... and {len(paths) - 50} more")
        lines.append("")

    return "\n".join(lines)


def _build_system_prompt(
    time_window: tuple[datetime | None, datetime | None] | None,
    bundle_subtype: str | None = None,
) -> str:
    """Load the system prompt and append time-window context when set.

    Why extracted: run_session and run_rag_session both inject the same
    time-window nudge. A single function eliminates the copy-paste and
    ensures both modes stay in sync if the wording changes.
    """
    system_prompt = (_PROMPTS_DIR / "system.md").read_text()
    if bundle_subtype == "kvm":
        system_prompt += (
            "\n\n**Bundle contains KVM/libvirt — check kvm_logs and kvm_commands "
            "categories for VM failure evidence.**"
        )
    if time_window and any(time_window):
        start, end = time_window
        parts = []
        if start:
            parts.append(f"from {start.isoformat()}")
        if end:
            parts.append(f"to {end.isoformat()}")
        system_prompt += (
            f"\n\n**Active time window: {' '.join(parts)}. "
            "Focus your analysis on events and errors within this range. "
            "When grepping logs, prioritise lines with timestamps in this window.**"
        )
    return system_prompt


# Model identifier — update this string when a newer Sonnet version is available
_MODEL = "claude-sonnet-4-6"

# Path to the prompts directory, relative to this file.
# Using __file__ makes this work regardless of where the CLI is invoked from.
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# Commands the user can type to end the session
_EXIT_COMMANDS = {"exit", "quit", "q"}

# Maximum number of Q+A turns to keep in run_session history.
# Each turn is 2 messages (user + assistant). Older pairs are dropped so the
# context window doesn't grow unboundedly over long sessions.
_MAX_HISTORY_TURNS = 20


def _run_turn(
    *,
    client: anthropic.Anthropic,
    messages: list[MessageParam],
    system_prompt: str,
    index_text: str,
    manifest: FileManifest,
    bundle_root: Path,
    stats: SessionStats,
    turn_start: int,
    time_window: tuple[datetime | None, datetime | None] | None = None,
    reference_date: date | None = None,
) -> str:
    """
    Execute one agent turn: send messages to Claude, handle tool round-trips,
    trim tool history, and return the final answer text.

    Mutates ``messages`` in-place: tool call/result pairs from this turn are
    removed and replaced with a single assistant-answer message.
    """
    while True:
        for _attempt in range(3):
            try:
                response = client.messages.create(
                    model=_MODEL,
                    max_tokens=4096,
                    system=[
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": index_text,
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                    tools=TOOLS,
                    messages=messages,
                )
                break
            except (anthropic.RateLimitError, anthropic.APIConnectionError):
                if _attempt == 2:
                    raise
                wait = 60 - _attempt * 10
                print(f"API error — waiting {wait}s before retry...")
                time.sleep(wait)

        stats.update(response.usage)

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if isinstance(block, anthropic.types.ToolUseBlock):
                    try:
                        result = dispatch_tool(
                            block.name, block.input, manifest, bundle_root, stats,
                            time_window=time_window,
                            reference_date=reference_date,
                        )
                    except Exception as exc:
                        result = f"Tool '{block.name}' raised an unexpected error: {exc}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            if response.stop_reason == "max_tokens":
                print("[Note: response was truncated — try a more specific question]")
            answer_text: str = ""
            for block in response.content:
                if hasattr(block, "text"):
                    print(block.text)
                    answer_text += str(block.text)

            del messages[turn_start:]
            messages.append({"role": "assistant", "content": answer_text})
            print(stats.compact_line(_MODEL))
            print()
            return answer_text


def run_session(
    manifest: FileManifest,
    bundle_root: Path,
    stats: SessionStats,
    time_window: tuple[datetime | None, datetime | None] | None = None,
    max_cost_usd: float | None = None,
    bundle_subtype: str | None = None,
    reference_date: date | None = None,
) -> None:
    """
    Run the interactive agent session until the user exits.

    This is the main loop of the application. It:
    1. Reads the system prompt and file index from disk / memory
    2. Accepts user questions from stdin
    3. Sends each question to Claude with the tool definitions
    4. Handles tool call round-trips (Claude calls a tool → we execute it → Claude continues)
    5. Prints Claude's final answer plus token stats after each question
    6. Prints the full session summary on exit

    Why prompt caching: The system prompt and file index are the same for every
    message in a session. Marking them with cache_control means Anthropic caches
    them server-side, and subsequent turns pay only for the user message tokens
    rather than re-sending the full context each time. On a 10-question session
    this typically saves 80-90% of input token costs.

    Why an inner loop: Claude's tool_use stop reason means "I want to call a tool,
    please give me the result and let me continue". The inner while loop handles
    this: send → check stop_reason → if tool_use, execute tools and send results
    → repeat until stop_reason == "end_turn" (Claude is done thinking).

    Args:
        manifest:    FileManifest from index_files() — passed to dispatch_tool().
        bundle_root: Path to the unpacked bundle directory.
        stats:       SessionStats initialized with naive_baseline_tokens set.
                     This function accumulates token usage into it throughout the session.
    """
    api_key = load_api_key()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set.\n"
            "Run 'bundle-platform setup' to configure it, or set the environment variable."
        )

    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = _build_system_prompt(time_window, bundle_subtype=bundle_subtype)

    # Format the file index once — it doesn't change during the session either.
    # Both the system prompt and index will be sent as cached blocks.
    index_text = _format_index(manifest)

    # Conversation history — grows as the session progresses.
    # The Anthropic API requires the full message history on every call
    # (it's stateless), so we accumulate it here.
    messages: list[MessageParam] = []

    print("Type your question, or 'exit' to quit.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl+D and Ctrl+C gracefully — print a newline and exit cleanly
            print()
            break

        if not question:
            continue  # ignore empty input, re-prompt

        if question.lower() in _EXIT_COMMANDS:
            break

        stats.begin_turn()
        # Add the user's question to the conversation history
        messages.append({"role": "user", "content": question})
        # Mark where this turn begins so we can trim tool call/result
        # messages after Claude produces its final answer.
        turn_start = len(messages)

        _run_turn(
            client=client,
            messages=messages,
            system_prompt=system_prompt,
            index_text=index_text,
            manifest=manifest,
            bundle_root=bundle_root,
            stats=stats,
            turn_start=turn_start,
            time_window=time_window,
            reference_date=reference_date,
        )

        if max_cost_usd is not None and stats._compute_cost(_MODEL) >= max_cost_usd:
            cost = stats._compute_cost(_MODEL)
            print(
                f"\n⚠ Session cost ${cost:.2f} reached limit "
                f"${max_cost_usd:.2f}. Use --max-cost to raise.",
                file=sys.stderr,
            )
            break

        # Drop oldest Q+A pairs when history exceeds the sliding window.
        # Each turn = 2 messages (user question + assistant answer).
        if len(messages) > _MAX_HISTORY_TURNS * 2:
            del messages[:-_MAX_HISTORY_TURNS * 2]

    # Session ended — print the full summary
    print(stats.full_report(total_files=len(manifest.entries), model_id=_MODEL))


def run_rag_session(
    retriever: _RetrieverProtocol,
    manifest: FileManifest,
    bundle_root: Path,
    stats: SessionStats,
    time_window: tuple[datetime | None, datetime | None] | None = None,
    max_cost_usd: float | None = None,
    bundle_subtype: str | None = None,
    reference_date: date | None = None,
) -> None:
    """
    Run an interactive session using RAG context + tool loop.

    Each user question triggers RAG retrieval to seed a <context> block, then
    Claude enters the full tool loop (_run_turn) and may call tools to gather
    additional evidence. Setting turn_start before appending the user message
    causes _run_turn to strip the context-heavy user message from history after each
    turn, keeping only clean assistant answers — the same token-efficient
    rolling history as before.
    """
    api_key = load_api_key()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set.\n"
            "Run 'bundle-platform setup' to configure it, or set the environment variable."
        )

    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = _build_system_prompt(time_window, bundle_subtype=bundle_subtype)
    index_text = _format_index(manifest)
    messages: list[MessageParam] = []

    print("Type your question, or 'exit' to quit.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in _EXIT_COMMANDS:
            break

        stats.begin_turn()
        try:
            context = retriever.retrieve(question, time_window=time_window)
        except RagUnavailable as exc:
            print(
                f"[warning] RAG unavailable ({exc}); falling back to tool-only mode",
                flush=True,
            )
            run_session(
                manifest,
                bundle_root,
                stats,
                time_window=time_window,
                max_cost_usd=max_cost_usd,
                bundle_subtype=bundle_subtype,
                reference_date=reference_date,
            )
            return
        user_content = (
            f"<context>\n{context}\n</context>\n\n{question}" if context else question
        )

        # Set turn_start BEFORE appending: _run_turn will del messages[turn_start:]
        # which strips the context-heavy user message from history when done.
        turn_start = len(messages)
        messages.append({"role": "user", "content": user_content})

        _run_turn(
            client=client,
            messages=messages,
            system_prompt=system_prompt,
            index_text=index_text,
            manifest=manifest,
            bundle_root=bundle_root,
            stats=stats,
            turn_start=turn_start,
            time_window=time_window,
            reference_date=reference_date,
        )

        if max_cost_usd is not None and stats._compute_cost(_MODEL) >= max_cost_usd:
            cost = stats._compute_cost(_MODEL)
            print(
                f"\n⚠ Session cost ${cost:.2f} reached limit "
                f"${max_cost_usd:.2f}. Use --max-cost to raise.",
                file=sys.stderr,
            )
            break

        if len(messages) > _MAX_HISTORY_TURNS * 2:
            del messages[: -_MAX_HISTORY_TURNS * 2]

    print(stats.full_report(total_files=len(manifest.entries), model_id=_MODEL))
