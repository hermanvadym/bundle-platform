# Design: Phase 2a — Eval Framework + LLM Abstraction

**Date:** 2026-04-18
**Status:** Approved — supersedes `2026-04-18-phase2-ingestion-engine-kvm.md`
**Scope:** `llm/`, `eval/`, refactor of `agent/loop.py`

---

## Why This Replaces the Drain3-First Plan

The original Phase 2 plan shipped Drain3 template mining + deduplication without any way to verify these techniques actually help retrieval. On a sosreport where log lines are already specific (device IDs, PIDs), Drain3 can *hurt* recall by collapsing distinguishing tokens.

We reverse the order:

1. **First build a measurement harness** that can score any preprocessing/retrieval strategy against a fixed golden set of real user questions.
2. **Each candidate technology** (Drain3, BM25, spaCy, TextFSM, …) becomes a plug-in **strategy**, scored against a baseline.
3. **Ship only what wins** on the scorecard.

This also solves the work-laptop deployment problem: the LLM backend becomes a config flag so the same code runs against Anthropic direct, AWS Bedrock, GCP Vertex, or Azure without edits.

---

## Goal

Ship three tightly-scoped units:

1. **`llm/` package** — a thin `LLMClient` Protocol with four backend implementations (Anthropic direct, Bedrock, Vertex, Azure). Existing agent code is refactored to call the protocol instead of `anthropic.Anthropic` directly.

2. **`eval/` package** — golden dataset loader, `Strategy` protocol, `BaselineStrategy` (wraps current chunker+embedder+retriever), RAGAS-based metrics, runner, and scorecard reporter.

3. **First comparison strategy** — `WithDedupStrategy`: adds simple consecutive-identical-line collapsing. Cheapest candidate; if even this doesn't help, the whole dedup idea is suspect before we touch Drain3.

---

## Architecture

```
bundle_platform/
├── llm/
│   ├── __init__.py          # get_client() factory reads BUNDLE_PLATFORM_LLM env
│   ├── client.py            # LLMClient Protocol + Response/Message types
│   ├── anthropic_direct.py  # AnthropicDirectClient
│   ├── bedrock.py           # BedrockClient (AWS)
│   ├── vertex.py            # VertexClient (GCP)
│   └── azure.py             # AzureClient
├── agent/
│   └── loop.py              # refactored: takes LLMClient, not raw anthropic
└── eval/
    ├── __init__.py
    ├── golden.py            # load_golden_set(dir) -> list[GoldenQuestion]
    ├── strategy.py          # Strategy Protocol + BaselineStrategy
    ├── metrics.py           # ragas_score(question, retrieved, answer) -> Metrics
    ├── runner.py            # run_scorecard(bundle, golden, strategies) -> Scorecard
    ├── report.py            # render_markdown(scorecard) -> str
    ├── cli.py               # bundle-platform-eval entrypoint
    └── strategies/
        └── with_dedup.py    # first comparison strategy
```

---

## Unit 1: `llm/client.py`

A minimal Protocol that covers what `agent/loop.py` actually uses: `messages.create` with tools and prompt caching.

```python
from typing import Protocol
from dataclasses import dataclass

@dataclass
class LLMUsage:
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

@dataclass
class LLMResponse:
    content: list  # list of content blocks — text or tool_use
    stop_reason: str
    usage: LLMUsage

class LLMClient(Protocol):
    model_id: str
    def complete(
        self,
        system: list[dict],      # cached system blocks
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 4096,
    ) -> LLMResponse: ...
```

All four backends return the same `LLMResponse` shape so `agent/loop.py` doesn't branch. Each backend owns its SDK import so users only install what they need (`pip install bundle-platform[bedrock]`).

**Backend selection** — one env var:

```bash
BUNDLE_PLATFORM_LLM=anthropic|bedrock|vertex|azure
BUNDLE_PLATFORM_MODEL=<model id for that backend>
```

Model ID defaults per backend are documented in `docs/MODELS.md`.

---

## Unit 2: `eval/golden.py` — Dataset Format

One YAML file per question. Human-editable, git-diffable, grep-friendly.

```yaml
# eval/golden/prod-20260115/001_mysqld_oom.yaml
id: 001_mysqld_oom
bundle: sosreport-prod-20260115.tar.xz
question: "What process was killed by OOM around 03:14 on Jan 15?"
expected_files:
  - var/log/messages
expected_evidence_regex: "oom_kill.*mysqld"
expected_answer_contains:
  - mysqld
  - OOM
  - "03:14"
notes: "Live-confirmed: mysqld at 14GB, kernel killed at 03:14:22"
```

`load_golden_set(dir: Path) -> list[GoldenQuestion]` reads the directory, validates every file has the required fields, and returns a list. Missing fields fail loudly — silent skips would corrupt scores.

---

## Unit 3: `eval/strategy.py` — Strategy Protocol

A strategy owns the full retrieval pipeline for a bundle: preprocessing, indexing, and retrieval. Baseline wraps the current `chunker + fastembed + qdrant + retriever` stack.

```python
class Strategy(Protocol):
    name: str  # unique identifier, shows up in scorecard

    def preprocess(self, bundle_root: Path) -> None:
        """Build whatever index the strategy needs. Idempotent — may reuse cache."""

    def retrieve(self, question: str) -> RetrievedContext:
        """Return context string + source file paths actually surfaced."""
```

`RetrievedContext` carries both the text shown to the LLM *and* the list of files it came from — so we can score "did the strategy surface the expected files?" independently of "did the LLM then produce the right answer?"

---

## Unit 4: `eval/metrics.py` — RAGAS + Evidence Match

Two scoring layers:

| Metric | What it measures | Source |
|---|---|---|
| `evidence_file_recall` | Of expected_files, how many did the strategy retrieve? | deterministic, no LLM |
| `evidence_regex_match` | Did retrieved context match expected_evidence_regex? | deterministic |
| `answer_keyword_match` | Did answer contain all expected_answer_contains terms? | deterministic |
| `ragas_context_precision` | RAGAS: is retrieved context relevant? | LLM-judged |
| `ragas_context_recall` | RAGAS: did context contain the answer? | LLM-judged |
| `ragas_faithfulness` | RAGAS: is answer grounded in context? | LLM-judged |

Deterministic metrics run always and give a reliable floor. RAGAS metrics add nuance but cost API calls and have noise — weighted into the final score lower than deterministic ones.

RAGAS backend honours the same `BUNDLE_PLATFORM_LLM` setting so eval scoring uses the same Claude model as the agent.

---

## Unit 5: `eval/runner.py` + `eval/report.py`

`run_scorecard(bundle, golden_dir, strategies, seeds=3)`:

1. For each strategy: preprocess bundle once.
2. For each (strategy, question) pair: run retrieval → agent → answer. Repeat `seeds=3` times, average.
3. Collect metrics → `Scorecard` dataframe.
4. Render markdown table + per-question diff against the first strategy (treated as baseline).

Scorecard shape:

```
Strategy       | Evidence Recall | Precision | Faithfulness | Keyword Match | Cost   | Time
baseline       |      0.72       |   0.61    |    0.84      |    14/20      | $0.18  |  42s
with_dedup     |      0.74       |   0.68    |    0.85      |    15/20      | $0.12  |  38s
```

Per-question diff section flags regressions: every question where a non-baseline strategy scored worse than baseline is listed with the offending metric highlighted.

---

## Unit 6: `eval/strategies/with_dedup.py` — First Comparison

Simple consecutive-line collapse: if line N equals line N-1, increment a counter on the previous chunk instead of emitting a new one. No Drain3, no templates — just raw equality.

```python
class WithDedupStrategy(BaselineStrategy):
    name = "with_dedup"
    def preprocess(self, bundle_root: Path) -> None:
        # Same as baseline, but insert _collapse_consecutive before chunking
```

If this loses to baseline, ingesting Drain3 (10× the complexity) is a bad bet.

---

## What This Phase Does NOT Include

- Drain3 template mining (becomes Phase 2b strategy if baseline loses)
- BM25 / hybrid search (Phase 2c strategy)
- spaCy entity extraction (Phase 2d strategy)
- TextFSM structured CLI parsing (Phase 2e strategy)
- KVM parser (parked — revisit after eval harness validates the core stack)
- Incremental ingestion engine (parked — same reason)

---

## Testing Strategy

- `tests/llm/test_client_protocol.py` — a fake LLMClient + assertion that `agent/loop.py` works against it
- `tests/llm/test_bedrock.py` — mocked boto3 client, assert request shape + response parsing
- `tests/eval/test_golden.py` — load fixture golden YAMLs; assert missing-field files raise
- `tests/eval/test_strategy.py` — BaselineStrategy preprocess+retrieve against a tiny fixture bundle
- `tests/eval/test_metrics.py` — deterministic metrics with handcrafted inputs/expected scores
- `tests/eval/test_runner.py` — 2-strategy × 2-question run with a stub LLM; assert scorecard shape
- `tests/eval/test_with_dedup.py` — collapse logic on synthetic repeated lines

No live API calls in the test suite. RAGAS is mocked.

---

## Deployment Considerations

Documented in `docs/DEPLOYMENT.md` (new, created by this phase):

- Which backend to use on which machine (personal vs work laptop)
- Required IAM / auth for each cloud backend
- Corporate proxy / TLS-inspection workaround (`REQUESTS_CA_BUNDLE`)
- Offline fallback path via Ollama (not implemented in this phase, but documented as escape hatch)
- Data-egress warning: bundle chunks leave the laptop for the LLM backend
