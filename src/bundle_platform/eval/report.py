from __future__ import annotations

from collections import defaultdict

from bundle_platform.eval.runner import Scorecard

_METRICS = ("evidence_file_recall", "evidence_regex_match", "answer_keyword_match", "seconds")


def render_markdown(card: Scorecard) -> str:
    """Render a Scorecard as a markdown table, averaging metrics per strategy."""
    by_strategy: dict[str, list[dict]] = defaultdict(list)
    for row in card.rows:
        by_strategy[row["strategy"]].append(row)

    lines = ["# Scorecard", ""]
    header = "| Strategy | " + " | ".join(_METRICS) + " |"
    sep = "|" + "---|" * (len(_METRICS) + 1)
    lines += [header, sep]
    for strategy, rows in by_strategy.items():
        cells = [strategy]
        for metric in _METRICS:
            values = [r[metric] for r in rows if metric in r]
            avg = sum(values) / len(values) if values else 0.0
            cells.append(f"{avg:.2f}s" if metric == "seconds" else f"{avg:.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
