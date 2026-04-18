from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from bundle_platform.eval.golden import GoldenQuestion
from bundle_platform.eval.metrics import score_deterministic
from bundle_platform.eval.strategy import Strategy


@dataclass
class Scorecard:
    rows: list[dict] = field(default_factory=list)


def run_scorecard(
    *,
    bundle_root: Path,
    questions: list[GoldenQuestion],
    strategies: list[Strategy],
    answerer,
    seeds: int = 1,
) -> Scorecard:
    """Run all strategies against all questions, return a Scorecard.

    For each strategy:
      1. Call strategy.preprocess(bundle_root)
      2. For each question, call strategy.retrieve(question.question)
      3. Call answerer(question.question, context.text) to get an answer string
      4. Call score_deterministic and record row with strategy name, question id,
         metrics, elapsed seconds
    """
    card = Scorecard()
    for strategy in strategies:
        strategy.preprocess(bundle_root)
        for question in questions:
            t0 = time.monotonic()
            context = strategy.retrieve(question.question)
            answer = answerer(question.question, context.text)
            elapsed = time.monotonic() - t0
            scores = score_deterministic(question, context, answer)
            card.rows.append({
                "strategy": strategy.name,
                "question_id": question.id,
                **scores,
                "seconds": elapsed,
            })
    return card
