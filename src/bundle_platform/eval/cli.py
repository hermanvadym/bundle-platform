from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bundle_platform.eval.golden import load_golden_set
from bundle_platform.eval.report import render_markdown
from bundle_platform.eval.runner import run_scorecard
from bundle_platform.eval.strategies.baseline import BaselineStrategy
from bundle_platform.eval.strategies.combined import CombinedStrategy
from bundle_platform.eval.strategies.with_dedup import WithDedupStrategy
from bundle_platform.eval.strategies.with_drain3 import WithDrain3Strategy
from bundle_platform.eval.strategies.with_rerank import WithRerankStrategy
from bundle_platform.eval.strategy import Strategy

_STRATEGIES: dict[str, type[Strategy]] = {
    "baseline": BaselineStrategy,
    "with_dedup": WithDedupStrategy,
    "with_drain3": WithDrain3Strategy,
    "with_rerank": WithRerankStrategy,
    "combined": CombinedStrategy,
}


def _null_answerer(question: str, context: str) -> str:
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bundle-platform-eval")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run eval scorecard")
    run_p.add_argument("--bundle", type=Path, required=True, help="Path to bundle root")
    run_p.add_argument("--golden", type=Path, required=True, help="Directory of golden YAML files")
    run_p.add_argument(
        "--strategies",
        default="baseline",
        help="Comma-separated strategy names (default: baseline)",
    )
    run_p.add_argument("--output", type=Path, default=None, help="Write markdown report to file")

    args = parser.parse_args(argv)

    questions = load_golden_set(args.golden)
    if not questions:
        print(f"No golden questions found in {args.golden}", file=sys.stderr)
        return 1

    strategy_names = [s.strip() for s in args.strategies.split(",")]
    strategies = []
    for name in strategy_names:
        if name not in _STRATEGIES:
            print(f"Unknown strategy: {name!r}. Available: {list(_STRATEGIES)}", file=sys.stderr)
            return 1
        strategies.append(_STRATEGIES[name]())

    card = run_scorecard(
        bundle_root=args.bundle,
        questions=questions,
        strategies=strategies,
        answerer=_null_answerer,
    )
    report = render_markdown(card)

    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
