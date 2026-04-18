"""Model pricing table and session token accounting.

Anthropic updates pricing periodically. Keep values here — loop.py reads
them via price_for(model_id). Sources: console.anthropic.com pricing page.
"""

from dataclasses import dataclass, field

import anthropic


@dataclass(frozen=True)
class Pricing:
    input_per_mtok: float
    output_per_mtok: float
    cache_write_per_mtok: float  # 1.25x base input -- write penalty
    cache_read_per_mtok: float   # 0.10x base input -- read discount


PRICING: dict[str, Pricing] = {
    "claude-sonnet-4-6": Pricing(3.00, 15.00, 3.75, 0.30),
    "claude-opus-4-7": Pricing(15.00, 75.00, 18.75, 1.50),
    "claude-haiku-4-5-20251001": Pricing(1.00, 5.00, 1.25, 0.10),
}


def price_for(model_id: str) -> Pricing:
    try:
        return PRICING[model_id]
    except KeyError:
        raise KeyError(
            f"no pricing configured for model {model_id!r}; "
            f"add it to accounting.PRICING"
        ) from None


@dataclass
class SessionStats:
    """
    Tracks token usage and tool activity across an entire interactive session.

    Why: The agent's value proposition is using far fewer tokens than loading the
    entire bundle. SessionStats makes this concrete — after each answer the user
    sees exactly how much they saved vs. the naive "paste everything" approach.

    How the naive baseline works: When the bundle is indexed, total_chars is set
    to the sum of all text file character counts. Dividing by 4 gives an approximate
    token count (industry standard: ~4 chars per token). This is what it would cost
    to load every file into a single context window.

    Attributes:
        input_tokens:           Cumulative input tokens sent to Claude this session.
        output_tokens:          Cumulative output tokens received from Claude.
        tool_calls:             Total number of tool invocations made by Claude.
        files_touched:          Set of file paths actually read by tools.
                                A set automatically deduplicates repeated reads.
        naive_baseline_tokens:  Estimated tokens to load the entire bundle naively.
                                Set once from manifest.total_chars // 4 in main.py.
        cache_creation_tokens:  Tokens that were written to Anthropic's prompt cache this session.
                                Billed at 1.25x base input rate. Occurs on the first call after the
                                cached blocks change (first message in a session).
        cache_read_tokens:      Tokens served from cache on subsequent calls.
                                Billed at 0.10x base input rate. This is where the cost savings
                                come from on multi-turn sessions.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    files_touched: set[str] = field(default_factory=set)
    naive_baseline_tokens: int = 0
    cache_creation_tokens: int = 0  # tokens written to prompt cache (billed at 1.25x)
    cache_read_tokens: int = 0      # tokens read from prompt cache (billed at 0.10x)
    # Per-turn counters — reset by begin_turn() before each user question.
    # Used in compact_line() to show activity for the current turn only.
    turn_tool_calls: int = 0
    turn_files: set[str] = field(default_factory=set)

    def update(self, usage: anthropic.types.Usage) -> None:
        """
        Accumulate token counts from one API response.

        Called after every Claude API call. Tracks both regular and cached token
        counts separately because they are billed at different rates.

        Cache creation tokens (first call per session): billed at 1.25x input rate.
        Cache read tokens (subsequent calls): billed at 0.10x input rate.
        Regular input tokens: standard rate.

        Args:
            usage: The usage object from a Claude API response.
        """
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        # getattr with default 0 handles API versions that don't return these fields
        self.cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0

    def begin_turn(self) -> None:
        """Reset per-turn counters at the start of each question."""
        self.turn_tool_calls = 0
        self.turn_files = set()

    def total_cost(self, model_id: str) -> float:
        """Calculate total session cost for the given model."""
        p = price_for(model_id)
        return (
            self.input_tokens / 1_000_000 * p.input_per_mtok
            + self.output_tokens / 1_000_000 * p.output_per_mtok
            + self.cache_creation_tokens / 1_000_000 * p.cache_write_per_mtok
            + self.cache_read_tokens / 1_000_000 * p.cache_read_per_mtok
        )

    def compact_line(self, model_id: str) -> str:
        """
        Return a short one-line summary shown after each agent answer.

        Why compact: The user sees this after every answer, so it must be
        glanceable without interrupting the session flow. The full report
        is reserved for exit.

        Format example:
            [tokens: 4,210 used / ~580K baseline | cost: $0.01 | 99.3% saved]
        """
        agent_tokens = self.input_tokens + self.output_tokens
        naive = self.naive_baseline_tokens

        # Guard against division by zero when naive_baseline is 0 (e.g. in tests)
        saved_pct = (1 - agent_tokens / naive) * 100 if naive else 0.0

        cost = self.total_cost(model_id)
        # Show baseline in thousands (e.g. "~580K") — the exact count is not useful
        naive_k = naive // 1000
        return (
            f"[tokens: {agent_tokens:,} used / ~{naive_k}K baseline"
            f" | cost: ${cost:.2f} | {saved_pct:.1f}% saved"
            f" | tools: {self.turn_tool_calls} | files: {len(self.turn_files)}]"
        )

    def full_report(self, total_files: int, model_id: str) -> str:
        """
        Return a formatted multi-line session summary shown at exit.

        Why separate from compact_line: At session end the user wants the complete
        picture — files touched vs. total, absolute costs, and efficiency percentage.
        This is the "receipt" for the session.

        Args:
            total_files: Total number of files in the bundle (len(manifest.entries)).
                         Shows "N touched / M total" to demonstrate how little
                         of the bundle was actually read.
            model_id:    Model identifier for pricing lookup.

        Returns:
            Multi-line string with decorative border for visual separation.
        """
        agent_tokens = self.input_tokens + self.output_tokens
        naive = self.naive_baseline_tokens
        saved_pct = (1 - agent_tokens / naive) * 100 if naive else 0.0
        total_cost = self.total_cost(model_id)
        # Naive scenario has no output tokens — you'd paste files, not chat
        naive_cost = naive / 1_000_000 * price_for(model_id).input_per_mtok

        return (
            "\n─── Session Stats ──────────────────────────────\n"
            f"  Agent tokens used:     {self.input_tokens:,} input"
            f" / {self.output_tokens:,} output\n"
            f"  Tool calls made:       {self.tool_calls}\n"
            f"  Files touched:         {len(self.files_touched)} / {total_files} in bundle\n"
            "\n"
            f"  Naive baseline:        ~{naive:,} tokens (all text files in bundle)\n"
            f"  Efficiency:            {saved_pct:.1f}% reduction\n"
            "\n"
            f"  Estimated cost:        ${total_cost:.2f}  (agent)\n"
            f"  vs naive cost:         ${naive_cost:.2f}  (naive)\n"
            "────────────────────────────────────────────────"
        )
