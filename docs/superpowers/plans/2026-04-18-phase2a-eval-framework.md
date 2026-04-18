# Phase 2a Eval Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a measurement-first harness so every retrieval-improvement candidate (Drain3, BM25, spaCy, …) is scored against a baseline on real bundles before being shipped. Also abstract the LLM backend so the same code runs against Anthropic direct, AWS Bedrock, GCP Vertex, or Azure.

**Architecture:** Three new packages — `llm/` (backend abstraction), `eval/` (golden dataset + strategy protocol + runner + scorecard), and `eval/strategies/` (plug-in retrieval configurations). `agent/loop.py` is refactored once to call the LLMClient protocol, then everything else slots in without touching the agent.

**Tech Stack:** Python 3.13+, uv, anthropic SDK (+bedrock, +vertex extras), ragas, pyyaml, pytest, ruff, ty.

**Supersedes:** `2026-04-18-phase2-ingestion-engine-kvm.md` (Drain3 + engine + KVM parser). Those candidates re-enter as *strategies* in future phases if (and only if) they win on the scorecard.

---

## Task 1: LLMClient Protocol + Anthropic Direct Backend

**Files:**
- Create: `src/bundle_platform/llm/__init__.py`
- Create: `src/bundle_platform/llm/client.py`
- Create: `src/bundle_platform/llm/anthropic_direct.py`
- Test: `tests/llm/test_client.py`
- Test: `tests/llm/test_anthropic_direct.py`

- [ ] **Step 1: Write the failing protocol test**

```python
# tests/llm/test_client.py
from dataclasses import dataclass

from bundle_platform.llm.client import LLMClient, LLMResponse, LLMUsage


def test_llm_response_dataclass_fields():
    usage = LLMUsage(input_tokens=10, output_tokens=5)
    resp = LLMResponse(content=[], stop_reason="end_turn", usage=usage)
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 10
    assert resp.usage.cache_creation_tokens == 0


def test_llm_client_is_runtime_checkable_protocol():
    @dataclass
    class Fake:
        model_id: str = "fake"
        def complete(self, system, messages, tools, max_tokens=4096):
            return LLMResponse(content=[], stop_reason="end_turn",
                               usage=LLMUsage(input_tokens=0, output_tokens=0))
    assert isinstance(Fake(), LLMClient)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm/test_client.py -v`
Expected: FAIL with `ModuleNotFoundError: bundle_platform.llm`

- [ ] **Step 3: Implement the protocol and dataclasses**

```python
# src/bundle_platform/llm/__init__.py
from bundle_platform.llm.client import LLMClient, LLMResponse, LLMUsage

__all__ = ["LLMClient", "LLMResponse", "LLMUsage", "get_client"]


def get_client() -> LLMClient:
    """Return the configured LLMClient based on BUNDLE_PLATFORM_LLM env."""
    import os
    backend = os.environ.get("BUNDLE_PLATFORM_LLM", "anthropic")
    if backend == "anthropic":
        from bundle_platform.llm.anthropic_direct import AnthropicDirectClient
        return AnthropicDirectClient()
    raise ValueError(f"Unknown LLM backend: {backend!r}")
```

```python
# src/bundle_platform/llm/client.py
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class LLMUsage:
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass
class LLMResponse:
    content: list[Any]       # list of Anthropic-compatible content blocks
    stop_reason: str         # "end_turn" | "tool_use" | "max_tokens"
    usage: LLMUsage
    raw: Any = field(default=None, repr=False)  # backend-native response for debugging


@runtime_checkable
class LLMClient(Protocol):
    model_id: str

    def complete(
        self,
        system: list[dict],
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 4096,
    ) -> LLMResponse: ...
```

- [ ] **Step 4: Write the Anthropic direct backend test**

```python
# tests/llm/test_anthropic_direct.py
from unittest.mock import MagicMock, patch

from bundle_platform.llm.anthropic_direct import AnthropicDirectClient
from bundle_platform.llm.client import LLMResponse


@patch("bundle_platform.llm.anthropic_direct.anthropic.Anthropic")
def test_complete_maps_response_shape(mock_cls):
    fake_resp = MagicMock()
    fake_resp.content = ["block1"]
    fake_resp.stop_reason = "end_turn"
    fake_resp.usage.input_tokens = 100
    fake_resp.usage.output_tokens = 50
    fake_resp.usage.cache_creation_input_tokens = 10
    fake_resp.usage.cache_read_input_tokens = 5
    mock_cls.return_value.messages.create.return_value = fake_resp

    client = AnthropicDirectClient(api_key="test", model_id="claude-sonnet-4-6")
    resp = client.complete(system=[], messages=[], tools=[])

    assert isinstance(resp, LLMResponse)
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 100
    assert resp.usage.cache_creation_tokens == 10
    assert resp.usage.cache_read_tokens == 5
```

- [ ] **Step 5: Implement AnthropicDirectClient**

```python
# src/bundle_platform/llm/anthropic_direct.py
import os

import anthropic

from bundle_platform.llm.client import LLMResponse, LLMUsage


class AnthropicDirectClient:
    def __init__(self, api_key: str | None = None, model_id: str | None = None) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=key)
        self.model_id = model_id or os.environ.get(
            "BUNDLE_PLATFORM_MODEL", "claude-sonnet-4-6"
        )

    def complete(
        self,
        system: list[dict],
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 4096,
    ) -> LLMResponse:
        resp = self._client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
        )
        return LLMResponse(
            content=list(resp.content),
            stop_reason=resp.stop_reason or "end_turn",
            usage=LLMUsage(
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                cache_creation_tokens=getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
                cache_read_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            ),
            raw=resp,
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/llm/ -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/bundle_platform/llm/ tests/llm/
git commit -m "feat(llm): add LLMClient protocol + Anthropic direct backend"
```

---

## Task 2: Bedrock, Vertex, Azure Backends

**Files:**
- Create: `src/bundle_platform/llm/bedrock.py`
- Create: `src/bundle_platform/llm/vertex.py`
- Create: `src/bundle_platform/llm/azure.py`
- Modify: `src/bundle_platform/llm/__init__.py` — extend `get_client()` factory
- Modify: `pyproject.toml` — add optional dependency groups
- Test: `tests/llm/test_bedrock.py`
- Test: `tests/llm/test_vertex.py`
- Test: `tests/llm/test_azure.py`
- Test: `tests/llm/test_factory.py`

- [ ] **Step 1: Add optional dependencies**

Edit `pyproject.toml`, add after `[project]` `dependencies`:

```toml
[project.optional-dependencies]
bedrock = ["anthropic[bedrock]==0.94.0", "boto3>=1.34.0"]
vertex = ["anthropic[vertex]==0.94.0"]
azure = ["anthropic[bedrock]==0.94.0"]  # Azure AI Foundry uses Bedrock-compatible path
```

Run: `uv sync --extra bedrock --extra vertex`

- [ ] **Step 2: Write the Bedrock test**

```python
# tests/llm/test_bedrock.py
from unittest.mock import MagicMock, patch

from bundle_platform.llm.bedrock import BedrockClient


@patch("bundle_platform.llm.bedrock.AnthropicBedrock")
def test_bedrock_complete_maps_usage(mock_cls):
    fake_resp = MagicMock()
    fake_resp.content = []
    fake_resp.stop_reason = "end_turn"
    fake_resp.usage.input_tokens = 42
    fake_resp.usage.output_tokens = 7
    fake_resp.usage.cache_creation_input_tokens = 0
    fake_resp.usage.cache_read_input_tokens = 0
    mock_cls.return_value.messages.create.return_value = fake_resp

    client = BedrockClient(
        aws_region="us-east-1",
        model_id="anthropic.claude-sonnet-4-5-20251001-v1:0",
    )
    resp = client.complete(system=[], messages=[], tools=[])
    assert resp.usage.input_tokens == 42
    assert resp.stop_reason == "end_turn"
```

- [ ] **Step 3: Implement BedrockClient**

```python
# src/bundle_platform/llm/bedrock.py
import os

try:
    from anthropic import AnthropicBedrock
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Bedrock backend requires 'anthropic[bedrock]'. "
        "Install with: uv sync --extra bedrock"
    ) from exc

from bundle_platform.llm.client import LLMResponse, LLMUsage


class BedrockClient:
    def __init__(self, aws_region: str | None = None, model_id: str | None = None) -> None:
        region = aws_region or os.environ.get("AWS_REGION", "us-east-1")
        self._client = AnthropicBedrock(aws_region=region)
        self.model_id = model_id or os.environ.get(
            "BUNDLE_PLATFORM_MODEL", "anthropic.claude-sonnet-4-5-20251001-v1:0"
        )

    def complete(self, system, messages, tools, max_tokens=4096) -> LLMResponse:
        resp = self._client.messages.create(
            model=self.model_id, max_tokens=max_tokens,
            system=system, messages=messages, tools=tools,
        )
        return LLMResponse(
            content=list(resp.content),
            stop_reason=resp.stop_reason or "end_turn",
            usage=LLMUsage(
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                cache_creation_tokens=getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
                cache_read_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            ),
            raw=resp,
        )
```

- [ ] **Step 4: Write the Vertex test + implementation**

```python
# tests/llm/test_vertex.py
from unittest.mock import MagicMock, patch
from bundle_platform.llm.vertex import VertexClient


@patch("bundle_platform.llm.vertex.AnthropicVertex")
def test_vertex_complete_maps_usage(mock_cls):
    fake = MagicMock()
    fake.content = []
    fake.stop_reason = "end_turn"
    fake.usage.input_tokens = 3
    fake.usage.output_tokens = 1
    fake.usage.cache_creation_input_tokens = 0
    fake.usage.cache_read_input_tokens = 0
    mock_cls.return_value.messages.create.return_value = fake

    client = VertexClient(region="us-east5", project_id="my-proj",
                          model_id="claude-sonnet-4-5@20251001")
    resp = client.complete(system=[], messages=[], tools=[])
    assert resp.usage.input_tokens == 3
```

```python
# src/bundle_platform/llm/vertex.py
import os

try:
    from anthropic import AnthropicVertex
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Vertex backend requires 'anthropic[vertex]'. Install with: uv sync --extra vertex"
    ) from exc

from bundle_platform.llm.client import LLMResponse, LLMUsage


class VertexClient:
    def __init__(self, region: str | None = None, project_id: str | None = None,
                 model_id: str | None = None) -> None:
        self._client = AnthropicVertex(
            region=region or os.environ["BUNDLE_PLATFORM_VERTEX_REGION"],
            project_id=project_id or os.environ["BUNDLE_PLATFORM_VERTEX_PROJECT"],
        )
        self.model_id = model_id or os.environ.get(
            "BUNDLE_PLATFORM_MODEL", "claude-sonnet-4-5@20251001"
        )

    def complete(self, system, messages, tools, max_tokens=4096) -> LLMResponse:
        resp = self._client.messages.create(
            model=self.model_id, max_tokens=max_tokens,
            system=system, messages=messages, tools=tools,
        )
        return LLMResponse(
            content=list(resp.content),
            stop_reason=resp.stop_reason or "end_turn",
            usage=LLMUsage(
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                cache_creation_tokens=getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
                cache_read_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            ),
            raw=resp,
        )
```

- [ ] **Step 5: Write the Azure test + implementation**

Azure AI Foundry exposes Claude through a compatibility endpoint. Use the raw `anthropic.Anthropic` client with a custom `base_url` per Microsoft's docs.

```python
# tests/llm/test_azure.py
from unittest.mock import MagicMock, patch
from bundle_platform.llm.azure import AzureClient


@patch("bundle_platform.llm.azure.anthropic.Anthropic")
def test_azure_uses_custom_base_url(mock_cls):
    fake = MagicMock()
    fake.content = []
    fake.stop_reason = "end_turn"
    fake.usage.input_tokens = 1
    fake.usage.output_tokens = 1
    fake.usage.cache_creation_input_tokens = 0
    fake.usage.cache_read_input_tokens = 0
    mock_cls.return_value.messages.create.return_value = fake

    client = AzureClient(
        endpoint="https://example.openai.azure.com/",
        api_key="azure-key",
        model_id="claude-sonnet-4-6",
    )
    client.complete(system=[], messages=[], tools=[])
    mock_cls.assert_called_once()
    kwargs = mock_cls.call_args.kwargs
    assert kwargs["base_url"].startswith("https://example.openai.azure.com")
    assert kwargs["api_key"] == "azure-key"
```

```python
# src/bundle_platform/llm/azure.py
import os

import anthropic

from bundle_platform.llm.client import LLMResponse, LLMUsage


class AzureClient:
    def __init__(self, endpoint: str | None = None, api_key: str | None = None,
                 model_id: str | None = None) -> None:
        base_url = endpoint or os.environ["BUNDLE_PLATFORM_AZURE_ENDPOINT"]
        key = api_key or os.environ["BUNDLE_PLATFORM_AZURE_API_KEY"]
        self._client = anthropic.Anthropic(base_url=base_url, api_key=key)
        self.model_id = model_id or os.environ.get(
            "BUNDLE_PLATFORM_MODEL", "claude-sonnet-4-6"
        )

    def complete(self, system, messages, tools, max_tokens=4096) -> LLMResponse:
        resp = self._client.messages.create(
            model=self.model_id, max_tokens=max_tokens,
            system=system, messages=messages, tools=tools,
        )
        return LLMResponse(
            content=list(resp.content),
            stop_reason=resp.stop_reason or "end_turn",
            usage=LLMUsage(
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                cache_creation_tokens=getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
                cache_read_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            ),
            raw=resp,
        )
```

- [ ] **Step 6: Extend the factory**

Replace `get_client()` body in `src/bundle_platform/llm/__init__.py`:

```python
def get_client() -> LLMClient:
    import os
    backend = os.environ.get("BUNDLE_PLATFORM_LLM", "anthropic")
    match backend:
        case "anthropic":
            from bundle_platform.llm.anthropic_direct import AnthropicDirectClient
            return AnthropicDirectClient()
        case "bedrock":
            from bundle_platform.llm.bedrock import BedrockClient
            return BedrockClient()
        case "vertex":
            from bundle_platform.llm.vertex import VertexClient
            return VertexClient()
        case "azure":
            from bundle_platform.llm.azure import AzureClient
            return AzureClient()
        case _:
            raise ValueError(f"Unknown LLM backend: {backend!r}")
```

- [ ] **Step 7: Factory test**

```python
# tests/llm/test_factory.py
from unittest.mock import patch
import pytest
from bundle_platform.llm import get_client


@patch.dict("os.environ", {"BUNDLE_PLATFORM_LLM": "unknown"})
def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        get_client()


@patch.dict("os.environ", {"BUNDLE_PLATFORM_LLM": "anthropic",
                            "ANTHROPIC_API_KEY": "test-key"})
def test_factory_returns_anthropic_client():
    client = get_client()
    assert client.model_id  # has model_id attribute
```

- [ ] **Step 8: Run and commit**

```bash
uv run pytest tests/llm/ -v
uv run ruff check src/bundle_platform/llm/ tests/llm/
uv run ty check src/bundle_platform/llm/
git add src/bundle_platform/llm/ tests/llm/ pyproject.toml
git commit -m "feat(llm): add Bedrock, Vertex, Azure backends + factory"
```

---

## Task 3: Refactor `agent/loop.py` to Use LLMClient

**Files:**
- Modify: `src/bundle_platform/agent/loop.py` — replace `anthropic.Anthropic` construction with `get_client()`
- Modify: `src/bundle_platform/agent/accounting.py` — `update()` accepts `LLMUsage` instead of `anthropic.types.Usage`
- Test: `tests/agent/test_loop_llm_abstraction.py`

- [ ] **Step 1: Write the failing test for a fake LLM client**

```python
# tests/agent/test_loop_llm_abstraction.py
from bundle_platform.agent.loop import _run_turn
from bundle_platform.agent.accounting import SessionStats
from bundle_platform.llm.client import LLMResponse, LLMUsage
from bundle_platform.tools.generic import FileManifest


class FakeClient:
    model_id = "fake-model"
    def __init__(self):
        self.calls = 0
    def complete(self, system, messages, tools, max_tokens=4096):
        self.calls += 1
        class _TextBlock:
            text = "answer"
        return LLMResponse(
            content=[_TextBlock()], stop_reason="end_turn",
            usage=LLMUsage(input_tokens=10, output_tokens=5),
        )


def test_run_turn_uses_llm_client(tmp_path):
    manifest = FileManifest(entries=[])
    stats = SessionStats()
    messages: list = [{"role": "user", "content": "hello"}]
    fake = FakeClient()

    answer = _run_turn(
        client=fake, messages=messages, system_prompt="sys",
        index_text="idx", manifest=manifest, bundle_root=tmp_path,
        stats=stats, turn_start=1,
    )
    assert answer == "answer"
    assert fake.calls == 1
    assert stats.input_tokens == 10
    assert stats.output_tokens == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agent/test_loop_llm_abstraction.py -v`
Expected: FAIL — `_run_turn` still takes `anthropic.Anthropic`, not an abstract client.

- [ ] **Step 3: Refactor `_run_turn` signature**

In `src/bundle_platform/agent/loop.py`:
- Change the `client: anthropic.Anthropic` parameter to `client: LLMClient` (import from `bundle_platform.llm.client`)
- Replace the `client.messages.create(model=_MODEL, ...)` call with `client.complete(system=..., messages=messages, tools=TOOLS, max_tokens=4096)`
- Remove the `model=_MODEL` arg (client owns its model_id)
- Replace `stats.update(response.usage)` — `response.usage` is now `LLMUsage`
- Replace `anthropic.types.ToolUseBlock` isinstance check with structural check: `getattr(block, "type", None) == "tool_use"` so the loop doesn't care which backend produced the block

- [ ] **Step 4: Update `SessionStats.update`**

In `src/bundle_platform/agent/accounting.py`, change `update(usage)` to accept `LLMUsage` (duck-typed is fine — it has the same four attribute names). Add a local import and type annotation.

- [ ] **Step 5: Update `run_session` and `run_rag_session`**

Replace:
```python
api_key = load_api_key()
client = anthropic.Anthropic(api_key=api_key)
```
with:
```python
from bundle_platform.llm import get_client
client = get_client()
```

Also replace the retry-loop `except (anthropic.RateLimitError, anthropic.APIConnectionError)` with a broad `except Exception as exc` that only retries if `exc.__class__.__name__` is in `{"RateLimitError", "APIConnectionError", "ServiceUnavailableError"}` — keeps the retry logic backend-agnostic.

- [ ] **Step 6: Run full agent test suite**

Run: `uv run pytest tests/agent/ -v`
Expected: all PASS (including the new test and existing loop tests)

- [ ] **Step 7: Commit**

```bash
git add src/bundle_platform/agent/ tests/agent/test_loop_llm_abstraction.py
git commit -m "refactor(agent): use LLMClient protocol instead of raw anthropic SDK"
```

---

## Task 4: Golden Dataset Loader

**Files:**
- Create: `src/bundle_platform/eval/__init__.py`
- Create: `src/bundle_platform/eval/golden.py`
- Create: `tests/eval/__init__.py`
- Create: `tests/eval/test_golden.py`
- Create: `tests/eval/fixtures/golden/valid/001_oom.yaml`
- Create: `tests/eval/fixtures/golden/invalid/002_missing_question.yaml`

- [ ] **Step 1: Add pyyaml dependency**

```bash
cd /home/kai/bundle-platform && uv add pyyaml
```

- [ ] **Step 2: Write the failing test**

```python
# tests/eval/test_golden.py
from pathlib import Path
import pytest
from bundle_platform.eval.golden import GoldenQuestion, load_golden_set

FIXTURES = Path(__file__).parent / "fixtures" / "golden"


def test_load_valid_set_parses_all_fields():
    questions = load_golden_set(FIXTURES / "valid")
    assert len(questions) == 1
    q = questions[0]
    assert q.id == "001_oom"
    assert q.bundle == "sosreport.tar.xz"
    assert q.question.startswith("What process")
    assert q.expected_files == ["var/log/messages"]
    assert q.expected_evidence_regex == "oom_kill.*mysqld"
    assert "mysqld" in q.expected_answer_contains


def test_missing_required_field_raises():
    with pytest.raises(ValueError, match="missing required field 'question'"):
        load_golden_set(FIXTURES / "invalid")


def test_empty_dir_returns_empty_list(tmp_path):
    assert load_golden_set(tmp_path) == []
```

- [ ] **Step 3: Create fixture files**

```yaml
# tests/eval/fixtures/golden/valid/001_oom.yaml
id: 001_oom
bundle: sosreport.tar.xz
question: "What process was killed by OOM?"
expected_files:
  - var/log/messages
expected_evidence_regex: "oom_kill.*mysqld"
expected_answer_contains:
  - mysqld
  - OOM
```

```yaml
# tests/eval/fixtures/golden/invalid/002_missing_question.yaml
id: 002_broken
bundle: sosreport.tar.xz
expected_files: [var/log/messages]
```

- [ ] **Step 4: Implement the loader**

```python
# src/bundle_platform/eval/__init__.py
from bundle_platform.eval.golden import GoldenQuestion, load_golden_set

__all__ = ["GoldenQuestion", "load_golden_set"]
```

```python
# src/bundle_platform/eval/golden.py
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_REQUIRED = ("id", "bundle", "question", "expected_files")


@dataclass
class GoldenQuestion:
    id: str
    bundle: str
    question: str
    expected_files: list[str]
    expected_evidence_regex: str | None = None
    expected_answer_contains: list[str] = field(default_factory=list)
    notes: str = ""


def load_golden_set(directory: Path) -> list[GoldenQuestion]:
    """Load every *.yaml in directory, validate, return list."""
    if not directory.exists():
        return []
    questions: list[GoldenQuestion] = []
    for path in sorted(directory.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text()) or {}
        for field_name in _REQUIRED:
            if field_name not in raw:
                raise ValueError(
                    f"{path}: missing required field {field_name!r}"
                )
        questions.append(GoldenQuestion(
            id=raw["id"],
            bundle=raw["bundle"],
            question=raw["question"],
            expected_files=list(raw["expected_files"]),
            expected_evidence_regex=raw.get("expected_evidence_regex"),
            expected_answer_contains=list(raw.get("expected_answer_contains", [])),
            notes=raw.get("notes", ""),
        ))
    return questions
```

- [ ] **Step 5: Run and commit**

```bash
uv run pytest tests/eval/ -v
git add src/bundle_platform/eval/ tests/eval/ pyproject.toml uv.lock
git commit -m "feat(eval): add golden dataset loader"
```

---

## Task 5: Strategy Protocol + BaselineStrategy

**Files:**
- Create: `src/bundle_platform/eval/strategy.py`
- Create: `src/bundle_platform/eval/strategies/__init__.py`
- Create: `src/bundle_platform/eval/strategies/baseline.py`
- Test: `tests/eval/test_strategy.py`
- Test: `tests/eval/test_baseline_strategy.py`

- [ ] **Step 1: Write the protocol test**

```python
# tests/eval/test_strategy.py
from pathlib import Path
from bundle_platform.eval.strategy import RetrievedContext, Strategy


class FakeStrategy:
    name = "fake"
    def preprocess(self, bundle_root: Path) -> None: pass
    def retrieve(self, question: str) -> RetrievedContext:
        return RetrievedContext(text="ctx", source_files=["a.log"])


def test_fake_matches_protocol(tmp_path):
    s: Strategy = FakeStrategy()
    s.preprocess(tmp_path)
    r = s.retrieve("q")
    assert r.text == "ctx"
    assert r.source_files == ["a.log"]
```

- [ ] **Step 2: Implement the protocol + dataclass**

```python
# src/bundle_platform/eval/strategy.py
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class RetrievedContext:
    text: str
    source_files: list[str]


@runtime_checkable
class Strategy(Protocol):
    name: str
    def preprocess(self, bundle_root: Path) -> None: ...
    def retrieve(self, question: str) -> RetrievedContext: ...
```

- [ ] **Step 3: Write the baseline strategy test (smoke-level)**

```python
# tests/eval/test_baseline_strategy.py
from pathlib import Path
from bundle_platform.eval.strategies.baseline import BaselineStrategy


def test_baseline_preprocess_and_retrieve(tmp_path: Path):
    bundle = tmp_path / "bundle"
    log = bundle / "var" / "log"
    log.mkdir(parents=True)
    (log / "messages").write_text(
        "Jan 15 03:14:22 host kernel: Out of memory: Killed process 9841 (mysqld)\n"
    )

    strat = BaselineStrategy(cache_root=tmp_path / "cache")
    strat.preprocess(bundle)
    result = strat.retrieve("What was killed by OOM?")

    assert strat.name == "baseline"
    assert result.text  # non-empty
    assert any("messages" in f for f in result.source_files)
```

- [ ] **Step 4: Implement BaselineStrategy**

```python
# src/bundle_platform/eval/strategies/__init__.py
from bundle_platform.eval.strategies.baseline import BaselineStrategy

__all__ = ["BaselineStrategy"]
```

```python
# src/bundle_platform/eval/strategies/baseline.py
from pathlib import Path

from bundle_platform.eval.strategy import RetrievedContext
from bundle_platform.pipeline.chunker import chunk_manifest
from bundle_platform.pipeline.embedder import Embedder
from bundle_platform.pipeline.retriever import Retriever
from bundle_platform.pipeline.store import VectorStore
from bundle_platform.tools.generic import index_files


class BaselineStrategy:
    """Current chunker + fastembed + qdrant + retriever stack."""

    name = "baseline"

    def __init__(self, cache_root: Path | None = None) -> None:
        self._cache_root = cache_root
        self._retriever: Retriever | None = None
        self._bundle_root: Path | None = None

    def preprocess(self, bundle_root: Path) -> None:
        self._bundle_root = bundle_root
        manifest = index_files(bundle_root)
        chunks = chunk_manifest(bundle_root, manifest)
        embedder = Embedder()
        store = VectorStore.in_memory()
        texts = [c.text for c in chunks]
        vectors = embedder.embed_texts(texts) if texts else []
        store.upsert(chunks, vectors)
        self._retriever = Retriever(
            store=store, embedder=embedder, bundle_root=bundle_root
        )

    def retrieve(self, question: str) -> RetrievedContext:
        if self._retriever is None:
            raise RuntimeError("preprocess() must be called before retrieve()")
        text = self._retriever.retrieve(question)
        # Extract file paths from the retriever's context block.
        # Retriever output format: "## <path>:<start>-<end>\n<text>\n\n..."
        files = sorted({
            line.split(":")[0].removeprefix("## ").strip()
            for line in text.splitlines()
            if line.startswith("## ")
        })
        return RetrievedContext(text=text, source_files=files)
```

> **Note:** If `Retriever.retrieve` has a different return format, adjust the file-extraction parser here. Check `src/bundle_platform/pipeline/retriever.py` before implementing — use whichever delimiter the retriever actually emits.

- [ ] **Step 5: Run and commit**

```bash
uv run pytest tests/eval/ -v
git add src/bundle_platform/eval/ tests/eval/
git commit -m "feat(eval): add Strategy protocol + BaselineStrategy"
```

---

## Task 6: Deterministic Metrics

**Files:**
- Create: `src/bundle_platform/eval/metrics.py`
- Test: `tests/eval/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_metrics.py
from bundle_platform.eval.golden import GoldenQuestion
from bundle_platform.eval.metrics import score_deterministic
from bundle_platform.eval.strategy import RetrievedContext


def _q(**kw):
    defaults = dict(
        id="q1", bundle="b.tar", question="?",
        expected_files=["var/log/messages"],
        expected_evidence_regex="oom_kill",
        expected_answer_contains=["mysqld", "OOM"],
    )
    defaults.update(kw)
    return GoldenQuestion(**defaults)


def test_all_expected_files_surfaced():
    q = _q()
    ctx = RetrievedContext(text="", source_files=["var/log/messages", "other"])
    s = score_deterministic(q, ctx, answer="")
    assert s["evidence_file_recall"] == 1.0


def test_partial_file_recall():
    q = _q(expected_files=["a.log", "b.log"])
    ctx = RetrievedContext(text="", source_files=["a.log"])
    assert score_deterministic(q, ctx, answer="")["evidence_file_recall"] == 0.5


def test_evidence_regex_match():
    q = _q()
    ctx = RetrievedContext(text="kernel: oom_kill process 9841", source_files=[])
    assert score_deterministic(q, ctx, answer="")["evidence_regex_match"] == 1.0


def test_evidence_regex_miss():
    q = _q()
    ctx = RetrievedContext(text="nothing here", source_files=[])
    assert score_deterministic(q, ctx, answer="")["evidence_regex_match"] == 0.0


def test_answer_keyword_match_all():
    q = _q()
    assert score_deterministic(q, RetrievedContext(text="", source_files=[]),
                                answer="mysqld killed by OOM")["answer_keyword_match"] == 1.0


def test_answer_keyword_match_partial():
    q = _q()
    assert score_deterministic(q, RetrievedContext(text="", source_files=[]),
                                answer="mysqld only")["answer_keyword_match"] == 0.5
```

- [ ] **Step 2: Implement the metrics**

```python
# src/bundle_platform/eval/metrics.py
import re

from bundle_platform.eval.golden import GoldenQuestion
from bundle_platform.eval.strategy import RetrievedContext


def score_deterministic(
    question: GoldenQuestion,
    context: RetrievedContext,
    answer: str,
) -> dict[str, float]:
    """Return deterministic metrics; no LLM calls."""
    expected = set(question.expected_files)
    retrieved = set(context.source_files)
    file_recall = (
        len(expected & retrieved) / len(expected) if expected else 1.0
    )

    if question.expected_evidence_regex:
        regex_match = 1.0 if re.search(
            question.expected_evidence_regex, context.text
        ) else 0.0
    else:
        regex_match = 1.0  # nothing to check

    if question.expected_answer_contains:
        found = sum(
            1 for kw in question.expected_answer_contains
            if kw.lower() in answer.lower()
        )
        keyword_match = found / len(question.expected_answer_contains)
    else:
        keyword_match = 1.0

    return {
        "evidence_file_recall": file_recall,
        "evidence_regex_match": regex_match,
        "answer_keyword_match": keyword_match,
    }
```

- [ ] **Step 3: Run and commit**

```bash
uv run pytest tests/eval/test_metrics.py -v
git add src/bundle_platform/eval/metrics.py tests/eval/test_metrics.py
git commit -m "feat(eval): add deterministic metrics (file recall, regex, keyword)"
```

---

## Task 7: Runner + Scorecard Report

**Files:**
- Create: `src/bundle_platform/eval/runner.py`
- Create: `src/bundle_platform/eval/report.py`
- Test: `tests/eval/test_runner.py`
- Test: `tests/eval/test_report.py`

- [ ] **Step 1: Write the runner test with stub strategy + stub agent**

```python
# tests/eval/test_runner.py
from pathlib import Path
from bundle_platform.eval.golden import GoldenQuestion
from bundle_platform.eval.runner import run_scorecard
from bundle_platform.eval.strategy import RetrievedContext


class StubStrategy:
    name = "stub"
    def preprocess(self, bundle_root: Path) -> None: pass
    def retrieve(self, question: str) -> RetrievedContext:
        return RetrievedContext(
            text="kernel: oom_kill mysqld",
            source_files=["var/log/messages"],
        )


def _answerer(question, context):
    return "mysqld was killed by OOM at 03:14"


def test_run_scorecard_all_metrics_present(tmp_path):
    question = GoldenQuestion(
        id="q1", bundle="b.tar",
        question="What was OOM killed?",
        expected_files=["var/log/messages"],
        expected_evidence_regex="oom_kill",
        expected_answer_contains=["mysqld", "OOM"],
    )
    card = run_scorecard(
        bundle_root=tmp_path,
        questions=[question],
        strategies=[StubStrategy()],
        answerer=_answerer,
        seeds=1,
    )
    assert len(card.rows) == 1
    row = card.rows[0]
    assert row["strategy"] == "stub"
    assert row["question_id"] == "q1"
    assert row["evidence_file_recall"] == 1.0
    assert row["evidence_regex_match"] == 1.0
    assert row["answer_keyword_match"] == 1.0
```

- [ ] **Step 2: Implement the runner**

```python
# src/bundle_platform/eval/runner.py
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from bundle_platform.eval.golden import GoldenQuestion
from bundle_platform.eval.metrics import score_deterministic
from bundle_platform.eval.strategy import RetrievedContext, Strategy

Answerer = Callable[[GoldenQuestion, RetrievedContext], str]


@dataclass
class Scorecard:
    rows: list[dict] = field(default_factory=list)


def run_scorecard(
    bundle_root: Path,
    questions: list[GoldenQuestion],
    strategies: list[Strategy],
    answerer: Answerer,
    seeds: int = 3,
) -> Scorecard:
    """Run every (strategy, question) pair `seeds` times, average, return card."""
    card = Scorecard()
    for strategy in strategies:
        strategy.preprocess(bundle_root)
        for question in questions:
            metrics_accum: dict[str, list[float]] = {}
            elapsed = 0.0
            for _ in range(seeds):
                t0 = time.monotonic()
                ctx = strategy.retrieve(question.question)
                answer = answerer(question, ctx)
                elapsed += time.monotonic() - t0
                det = score_deterministic(question, ctx, answer)
                for k, v in det.items():
                    metrics_accum.setdefault(k, []).append(v)
            row = {
                "strategy": strategy.name,
                "question_id": question.id,
                "seconds": round(elapsed / seeds, 2),
            }
            for k, values in metrics_accum.items():
                row[k] = round(sum(values) / len(values), 3)
            card.rows.append(row)
    return card
```

- [ ] **Step 3: Write the report test**

```python
# tests/eval/test_report.py
from bundle_platform.eval.report import render_markdown
from bundle_platform.eval.runner import Scorecard


def test_render_aggregates_per_strategy():
    card = Scorecard(rows=[
        {"strategy": "baseline", "question_id": "q1",
         "evidence_file_recall": 1.0, "evidence_regex_match": 1.0,
         "answer_keyword_match": 1.0, "seconds": 0.5},
        {"strategy": "baseline", "question_id": "q2",
         "evidence_file_recall": 0.5, "evidence_regex_match": 0.0,
         "answer_keyword_match": 0.5, "seconds": 0.5},
    ])
    md = render_markdown(card)
    assert "baseline" in md
    assert "0.75" in md  # average file recall (1.0 + 0.5) / 2
```

- [ ] **Step 4: Implement the report renderer**

```python
# src/bundle_platform/eval/report.py
from collections import defaultdict

from bundle_platform.eval.runner import Scorecard

_METRICS = ("evidence_file_recall", "evidence_regex_match",
            "answer_keyword_match", "seconds")


def render_markdown(card: Scorecard) -> str:
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
            cells.append(f"{avg:.2f}" if metric != "seconds" else f"{avg:.2f}s")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
```

- [ ] **Step 5: Run and commit**

```bash
uv run pytest tests/eval/test_runner.py tests/eval/test_report.py -v
git add src/bundle_platform/eval/runner.py src/bundle_platform/eval/report.py \
    tests/eval/test_runner.py tests/eval/test_report.py
git commit -m "feat(eval): add runner + markdown scorecard"
```

---

## Task 8: First Comparison Strategy — WithDedup

**Files:**
- Create: `src/bundle_platform/eval/strategies/with_dedup.py`
- Test: `tests/eval/test_with_dedup.py`

- [ ] **Step 1: Write the dedup strategy test**

```python
# tests/eval/test_with_dedup.py
from pathlib import Path
from bundle_platform.eval.strategies.with_dedup import (
    WithDedupStrategy,
    collapse_consecutive_duplicates,
)


def test_collapse_identical_consecutive_lines():
    raw = ["a", "a", "a", "b", "c", "c"]
    collapsed = collapse_consecutive_duplicates(raw)
    assert collapsed == ["a", "b", "c"]


def test_collapse_preserves_non_duplicate_order():
    raw = ["a", "b", "a", "b"]
    assert collapse_consecutive_duplicates(raw) == ["a", "b", "a", "b"]


def test_with_dedup_strategy_name():
    assert WithDedupStrategy().name == "with_dedup"
```

- [ ] **Step 2: Implement WithDedupStrategy**

```python
# src/bundle_platform/eval/strategies/with_dedup.py
from pathlib import Path

from bundle_platform.eval.strategies.baseline import BaselineStrategy
from bundle_platform.eval.strategy import RetrievedContext
from bundle_platform.pipeline.chunker import chunk_file
from bundle_platform.pipeline.embedder import Embedder
from bundle_platform.pipeline.retriever import Retriever
from bundle_platform.pipeline.store import VectorStore
from bundle_platform.tools.generic import index_files


def collapse_consecutive_duplicates(lines: list[str]) -> list[str]:
    """Collapse runs of identical adjacent lines to a single occurrence."""
    out: list[str] = []
    for line in lines:
        if not out or out[-1] != line:
            out.append(line)
    return out


class WithDedupStrategy(BaselineStrategy):
    """Baseline plus consecutive-duplicate line collapsing before chunking."""

    name = "with_dedup"

    def preprocess(self, bundle_root: Path) -> None:
        self._bundle_root = bundle_root
        manifest = index_files(bundle_root)
        chunks = []
        for entry in manifest.entries:
            path = bundle_root / entry.path
            if not path.is_file():
                continue
            try:
                raw = path.read_text(errors="replace").splitlines()
            except OSError:
                continue
            deduped = collapse_consecutive_duplicates(raw)
            tmp = bundle_root / f".dedup_{entry.path.replace('/', '_')}"
            tmp.write_text("\n".join(deduped))
            try:
                chunks.extend(chunk_file(bundle_root, entry))
            finally:
                tmp.unlink(missing_ok=True)

        embedder = Embedder()
        store = VectorStore.in_memory()
        texts = [c.text for c in chunks]
        vectors = embedder.embed_texts(texts) if texts else []
        store.upsert(chunks, vectors)
        self._retriever = Retriever(store=store, embedder=embedder, bundle_root=bundle_root)
```

> **Note:** The above uses a placeholder file-rewrite approach for clarity. If the chunker supports passing pre-read lines directly (check `chunk_file` signature at `src/bundle_platform/pipeline/chunker.py`), prefer that — avoids touching the bundle. Refactor to `chunk_file(bundle_root, entry, lines=deduped)` if the signature allows, otherwise keep the temp-file approach inside a `tmp_path` outside `bundle_root`.

- [ ] **Step 3: Run and commit**

```bash
uv run pytest tests/eval/test_with_dedup.py -v
git add src/bundle_platform/eval/strategies/with_dedup.py tests/eval/test_with_dedup.py
git commit -m "feat(eval): add WithDedupStrategy as first comparison candidate"
```

---

## Task 9: CLI Entrypoint

**Files:**
- Create: `src/bundle_platform/eval/cli.py`
- Modify: `pyproject.toml` — add `[project.scripts]` entry
- Test: `tests/eval/test_cli.py`

- [ ] **Step 1: Add the script entry to pyproject.toml**

```toml
[project.scripts]
bundle-platform-eval = "bundle_platform.eval.cli:main"
```

- [ ] **Step 2: Write the CLI smoke test**

```python
# tests/eval/test_cli.py
import sys
from unittest.mock import patch
import pytest
from bundle_platform.eval.cli import main


def test_cli_requires_args(capsys):
    with patch.object(sys, "argv", ["bundle-platform-eval"]):
        with pytest.raises(SystemExit):
            main()
    out = capsys.readouterr()
    assert "bundle" in (out.err + out.out).lower()
```

- [ ] **Step 3: Implement the CLI**

```python
# src/bundle_platform/eval/cli.py
import argparse
import sys
from pathlib import Path

from bundle_platform.eval.golden import load_golden_set
from bundle_platform.eval.report import render_markdown
from bundle_platform.eval.runner import run_scorecard
from bundle_platform.eval.strategies.baseline import BaselineStrategy
from bundle_platform.eval.strategies.with_dedup import WithDedupStrategy
from bundle_platform.llm import get_client

_STRATEGIES = {
    "baseline": BaselineStrategy,
    "with_dedup": WithDedupStrategy,
}


def main() -> int:
    parser = argparse.ArgumentParser(prog="bundle-platform-eval")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run scorecard against a bundle")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--golden", type=Path, required=True)
    run.add_argument(
        "--strategies", default="baseline,with_dedup",
        help="Comma-separated strategy names"
    )
    run.add_argument("--seeds", type=int, default=3)
    run.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()
    if args.command != "run":
        parser.print_help()
        return 1

    questions = load_golden_set(args.golden)
    if not questions:
        print(f"No golden questions found in {args.golden}", file=sys.stderr)
        return 1

    strategies = [_STRATEGIES[n.strip()]() for n in args.strategies.split(",")]
    client = get_client()

    def answerer(question, context):
        # Minimal single-turn answer — agent loop is overkill for eval.
        resp = client.complete(
            system=[{"type": "text", "text": "You answer bundle diagnostic questions."}],
            messages=[{"role": "user",
                       "content": f"<context>\n{context.text}\n</context>\n\n{question.question}"}],
            tools=[],
        )
        return "".join(
            getattr(b, "text", "") for b in resp.content
            if getattr(b, "type", None) == "text"
        )

    card = run_scorecard(
        bundle_root=args.bundle,
        questions=questions,
        strategies=strategies,
        answerer=answerer,
        seeds=args.seeds,
    )
    md = render_markdown(card)
    if args.output:
        args.output.write_text(md)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests, lint, type-check, commit**

```bash
uv run pytest tests/eval/ -v
uv run ruff check src/bundle_platform/eval/ tests/eval/
uv run ty check src/bundle_platform/eval/
git add src/bundle_platform/eval/cli.py tests/eval/test_cli.py pyproject.toml
git commit -m "feat(eval): add bundle-platform-eval CLI"
```

---

## Task 10: Deployment Docs

**Files:**
- Create: `docs/DEPLOYMENT.md`
- Create: `docs/MODELS.md`

- [ ] **Step 1: Write `docs/MODELS.md`**

Document the model ID mapping per backend:

```markdown
# Model IDs by Backend

| Backend | Sonnet | Opus |
|---|---|---|
| Anthropic direct | claude-sonnet-4-6 | claude-opus-4-7 |
| AWS Bedrock | anthropic.claude-sonnet-4-5-20251001-v1:0 | anthropic.claude-opus-4-7-20260215-v1:0 |
| GCP Vertex | claude-sonnet-4-5@20251001 | claude-opus-4-7@20260215 |
| Azure AI Foundry | claude-sonnet-4-6 | claude-opus-4-7 |

Set `BUNDLE_PLATFORM_MODEL` to override the default for a given backend.
```

- [ ] **Step 2: Write `docs/DEPLOYMENT.md`**

Cover:
1. Personal machine — `BUNDLE_PLATFORM_LLM=anthropic` + `ANTHROPIC_API_KEY`
2. Work laptop via Bedrock — `BUNDLE_PLATFORM_LLM=bedrock`, IAM role or AWS keys, region
3. Work laptop via Vertex — `BUNDLE_PLATFORM_LLM=vertex`, `gcloud auth application-default login`
4. Work laptop via Azure — endpoint + API key
5. Corporate proxy / TLS inspection — `REQUESTS_CA_BUNDLE` and `SSL_CERT_FILE`
6. Data-egress warning — bundle chunks leave the laptop; check with security
7. Running an eval scorecard end-to-end

- [ ] **Step 3: Commit**

```bash
git add docs/DEPLOYMENT.md docs/MODELS.md
git commit -m "docs: add DEPLOYMENT and MODELS guides"
```

---

## Task 11: Supersede the Old Phase 2 Plan

**Files:**
- Modify: `docs/superpowers/plans/2026-04-18-phase2-ingestion-engine-kvm.md`
- Modify: `docs/superpowers/specs/2026-04-18-phase2-ingestion-engine-kvm.md`

- [ ] **Step 1: Prepend a superseded notice to both files**

Add this block at the top of each file (before the existing `# Design:` / `# Implementation Plan` heading):

```markdown
> **⚠ SUPERSEDED 2026-04-18** — this plan was replaced by
> `2026-04-18-phase2a-eval-framework.md`. Drain3, the incremental engine, and the
> KVM parser are deferred until the eval harness demonstrates that simpler
> preprocessing changes (e.g. consecutive-line dedup) produce measurable
> retrieval gains. If Drain3 ever returns, it will re-enter as a plug-in
> `Strategy` scored against baseline, not as a mandatory pipeline change.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/plans/2026-04-18-phase2-ingestion-engine-kvm.md \
    docs/superpowers/specs/2026-04-18-phase2-ingestion-engine-kvm.md
git commit -m "docs: mark phase2 Drain3/KVM plan as superseded by phase2a eval"
```

---

## Final Validation

- [ ] Run full test suite: `uv run pytest -q`
- [ ] Lint: `uv run ruff check .`
- [ ] Type-check: `uv run ty check src/bundle_platform/`
- [ ] Verify CLI loads: `uv run bundle-platform-eval run --help`

Expected: all green. Eval harness ready to accept golden questions against real bundles.
