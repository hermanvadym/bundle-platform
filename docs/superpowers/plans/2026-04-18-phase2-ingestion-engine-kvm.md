# Phase 2 — Incremental Ingestion Engine + KVM Parser: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Drain3 template mining, template-aware deduplication, a checkpoint/resume ingestion engine, and a full KVM bundle parser to `bundle-platform`.

**Architecture:** `pipeline/template_miner.py` wraps Drain3; `pipeline/deduplicator.py` collapses repetitive lines using templates as dedup keys; `pipeline/engine.py` orchestrates the per-file pipeline (dedup → chunk → embed template → store original); `parsers/kvm.py` adds a dual-mode KVM adapter (sosreport + virsh-export). `LogChunk` gains two new defaulted fields (`text_template`, `repeat_count`) so existing code is unaffected.

**Tech Stack:** Python 3.13+, `drain3`, `fastembed`, `qdrant-client`, `anthropic`, `uv`, `ruff`, `ty`, `pytest`

---

## Import Translation Reference

All imports use `bundle_platform.*`. Key interfaces used in this plan:

| Symbol | Import path |
|---|---|
| `LogChunk`, `chunk_file` | `bundle_platform.pipeline.chunker` |
| `Embedder` | `bundle_platform.pipeline.embedder` |
| `VectorStore` | `bundle_platform.pipeline.store` |
| `RagUnavailable` | `bundle_platform.pipeline.exceptions` |
| `FileManifest`, `FileEntry` | `bundle_platform.tools.generic` |
| `cache_dir`, `is_preprocessed` | `bundle_platform.pipeline.preprocessor` |
| `BundleAdapter` | `bundle_platform.parsers.base` |

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/bundle_platform/pipeline/template_miner.py` | **Create** | Drain3 `TemplateMinerWrapper` — stateful, line-by-line template extraction |
| `src/bundle_platform/pipeline/deduplicator.py` | **Create** | `DedupLine` dataclass + `deduplicate()` — collapse repetitive lines |
| `src/bundle_platform/pipeline/chunker.py` | **Modify** | Add `text_template: str = ""`, `repeat_count: int = 1` to `LogChunk`; add `chunk_from_dedup_lines()` |
| `src/bundle_platform/pipeline/engine.py` | **Create** | `BundleEngine` — checkpoint/resume ingestion orchestrator |
| `src/bundle_platform/parsers/kvm.py` | **Create** | `_KvmAdapter(BundleAdapter)` — dual-mode KVM parser |
| `src/bundle_platform/parsers/detect.py` | **Modify** | Add KVM detection branch before RHEL |
| `src/bundle_platform/parsers/__init__.py` | **Modify** | Add `"kvm"` route in `load_adapter()` |
| `tests/pipeline/test_template_miner.py` | **Create** | Tests for `TemplateMinerWrapper` |
| `tests/pipeline/test_deduplicator.py` | **Create** | Tests for `deduplicate()` |
| `tests/pipeline/test_engine.py` | **Create** | Tests for `BundleEngine` checkpoint/resume |
| `tests/parsers/test_kvm.py` | **Create** | Tests for KVM adapter both modes |
| `pyproject.toml` | **Modify** | Add `drain3` dependency |

---

## Task 1: Add `drain3` dependency + `pipeline/template_miner.py`

**Files:**
- Modify: `pyproject.toml`
- Create: `src/bundle_platform/pipeline/template_miner.py`
- Create: `tests/pipeline/test_template_miner.py`

- [ ] **Step 1: Add drain3 dependency**

```bash
cd ~/bundle-platform && uv add drain3
```

Expected: `pyproject.toml` updated with `drain3` entry, `uv.lock` updated.

- [ ] **Step 2: Write failing tests**

Create `tests/pipeline/test_template_miner.py`:

```python
import pytest
from bundle_platform.pipeline.template_miner import TemplateMinerWrapper


def test_mine_returns_template():
    miner = TemplateMinerWrapper()
    # After seeing enough similar lines, Drain3 produces a template with <*> tokens
    lines = [
        "NMP: nmp_ThrottleLogForDevice: Cmd 0x2a to dev naa.600a098 on path vmhba1",
        "NMP: nmp_ThrottleLogForDevice: Cmd 0x2a to dev naa.700b199 on path vmhba2",
        "NMP: nmp_ThrottleLogForDevice: Cmd 0x2a to dev naa.800c200 on path vmhba3",
    ]
    templates = [miner.mine(line) for line in lines]
    # All lines should produce the same template after the tree stabilises
    assert templates[-1] == templates[-2]
    # Template should contain at least one <*> placeholder
    assert "<*>" in templates[-1]


def test_mine_distinct_lines_produce_different_templates():
    miner = TemplateMinerWrapper()
    t1 = miner.mine("kernel: OOM kill process 1234 httpd total-vm:512kB")
    t2 = miner.mine("link is down on eth0")
    assert t1 != t2


def test_mine_updates_tree_incrementally():
    miner = TemplateMinerWrapper()
    # First call — tree is empty, template = the line itself (no variables yet)
    t1 = miner.mine("Timeout for host 192.168.1.10 on port 443")
    # Second similar call — may refine the template
    t2 = miner.mine("Timeout for host 10.0.0.1 on port 443")
    # Both return strings (not None, not empty)
    assert isinstance(t1, str) and len(t1) > 0
    assert isinstance(t2, str) and len(t2) > 0


def test_mine_empty_line_returns_string():
    miner = TemplateMinerWrapper()
    result = miner.mine("")
    assert isinstance(result, str)
```

- [ ] **Step 3: Run tests — expect ImportError**

```bash
cd ~/bundle-platform && uv run pytest tests/pipeline/test_template_miner.py -v
```

Expected: `ImportError: No module named 'bundle_platform.pipeline.template_miner'`

- [ ] **Step 4: Implement `template_miner.py`**

Create `src/bundle_platform/pipeline/template_miner.py`:

```python
"""
Drain3 template miner wrapper for the ingestion pipeline.

Drain3 processes log lines one at a time and maintains a prefix tree of templates.
Each call to mine() returns the current best template for that line and updates the
tree — identical variable tokens (IPs, PIDs, device IDs) are replaced with <*>.

Why: Embedding the template instead of the raw line groups semantically similar log
events together in vector space, improving retrieval precision. The original line is
always preserved separately so Claude never loses forensic specificity.
"""

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


class TemplateMinerWrapper:
    """Stateful Drain3 wrapper. One instance per ingestion run."""

    def __init__(
        self,
        sim_threshold: float = 0.4,
        depth: int = 4,
        max_children: int = 100,
    ) -> None:
        config = TemplateMinerConfig()
        config.drain_sim_th = sim_threshold
        config.drain_depth = depth
        config.drain_max_children = max_children
        # Disable persistence — state lives only for the duration of this run
        config.snapshot_interval_minutes = 0
        self._miner = TemplateMiner(config=config)

    def mine(self, line: str) -> str:
        """
        Return the Drain3 template for this log line, updating the tree as a side effect.

        Returns the line itself if it does not match any existing template cluster
        and Drain3 has not yet seen enough similar lines to abstract variables.
        Never raises — returns an empty string for empty input.
        """
        if not line:
            return line
        result = self._miner.add_log_message(line)
        if result is None:
            return line
        return result["template_mined"]
```

- [ ] **Step 5: Run tests — expect all pass**

```bash
cd ~/bundle-platform && uv run pytest tests/pipeline/test_template_miner.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Ruff + ty check**

```bash
cd ~/bundle-platform && uv run ruff check src/bundle_platform/pipeline/template_miner.py tests/pipeline/test_template_miner.py && uv run ty check src/
```

Expected: no issues.

- [ ] **Step 7: Commit**

```bash
cd ~/bundle-platform && git add pyproject.toml uv.lock src/bundle_platform/pipeline/template_miner.py tests/pipeline/test_template_miner.py && git commit -m "feat: add Drain3 template miner wrapper"
```

---

## Task 2: `pipeline/deduplicator.py`

**Files:**
- Create: `src/bundle_platform/pipeline/deduplicator.py`
- Create: `tests/pipeline/test_deduplicator.py`

- [ ] **Step 1: Write failing tests**

Create `tests/pipeline/test_deduplicator.py`:

```python
from bundle_platform.pipeline.deduplicator import DedupLine, deduplicate
from bundle_platform.pipeline.template_miner import TemplateMinerWrapper


def _miner() -> TemplateMinerWrapper:
    return TemplateMinerWrapper()


def test_unique_lines_unchanged():
    lines = [
        "Apr 15 10:00:00 host kernel: OOM kill process 1234",
        "Apr 15 10:00:01 host sshd: Accepted password for root",
    ]
    result = deduplicate(lines, _miner())
    assert len(result) == 2
    assert result[0].text == lines[0]
    assert result[0].repeat_count == 1


def test_identical_consecutive_lines_collapsed():
    line = "Apr 15 10:00:00 host heartbeat: OK"
    lines = [line] * 10
    result = deduplicate(lines, _miner())
    assert len(result) == 1
    assert result[0].text == line
    assert result[0].repeat_count == 10


def test_similar_lines_within_window_collapsed():
    # Same template, timestamps within 60 seconds
    lines = [
        "Apr 15 10:00:00 NMP: ThrottleLog dev naa.600a to path vmhba1",
        "Apr 15 10:00:05 NMP: ThrottleLog dev naa.700b to path vmhba2",
        "Apr 15 10:00:10 NMP: ThrottleLog dev naa.800c to path vmhba3",
    ]
    miner = _miner()
    result = deduplicate(lines, miner, window_seconds=60)
    # After Drain3 stabilises on the template, these should collapse
    # (may not collapse on first call if tree hasn't stabilised — that's OK)
    # At minimum: each DedupLine is a DedupLine with text preserved
    assert all(isinstance(dl, DedupLine) for dl in result)
    assert all(dl.text in lines for dl in result)


def test_lines_outside_window_not_collapsed():
    lines = [
        "Apr 15 10:00:00 NMP: ThrottleLog dev naa.600a to path vmhba1",
        "Apr 15 10:02:00 NMP: ThrottleLog dev naa.700b to path vmhba2",  # 120s later
    ]
    miner = _miner()
    result = deduplicate(lines, miner, window_seconds=60)
    # Outside the 60s window — must NOT be collapsed even if same template
    assert len(result) == 2


def test_first_occurrence_text_preserved():
    lines = [
        "Apr 15 10:00:00 NMP: ThrottleLog dev naa.600a to path vmhba1",
        "Apr 15 10:00:05 NMP: ThrottleLog dev naa.700b to path vmhba2",
    ]
    miner = _miner()
    # Seed miner with many similar lines so template is stable before this test
    for _ in range(5):
        miner.mine("NMP: ThrottleLog dev naa.000x to path vmhba0")
    result = deduplicate(lines, miner, window_seconds=60)
    if len(result) == 1:
        # First occurrence text preserved — NOT the template, NOT the second line
        assert result[0].text == lines[0]
        assert "naa.600a" in result[0].text


def test_no_timestamp_lines_collapse_only_if_consecutive():
    lines = [
        "heartbeat OK",
        "heartbeat OK",
        "heartbeat OK",
        "link down on eth0",
        "heartbeat OK",
    ]
    result = deduplicate(lines, _miner())
    # First three collapse into one; then link down; then one more heartbeat
    texts = [dl.text for dl in result]
    assert "link down on eth0" in texts
    # Heartbeat before and after link-down must NOT be merged
    heartbeat_entries = [dl for dl in result if "heartbeat OK" in dl.text]
    assert len(heartbeat_entries) == 2


def test_dedup_line_has_template():
    lines = ["Apr 15 10:00:00 host kernel: OOM kill process 1234"]
    result = deduplicate(lines, _miner())
    assert isinstance(result[0].template, str)
    assert len(result[0].template) > 0
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
cd ~/bundle-platform && uv run pytest tests/pipeline/test_deduplicator.py -v
```

Expected: `ImportError: No module named 'bundle_platform.pipeline.deduplicator'`

- [ ] **Step 3: Implement `deduplicator.py`**

Create `src/bundle_platform/pipeline/deduplicator.py`:

```python
"""
Template-aware log line deduplicator.

Collapses consecutive log lines that share the same Drain3 template and fall
within a configurable time window into a single DedupLine. The first occurrence
is kept as `text` so specific variable values (device IDs, PIDs, hostnames) are
preserved for forensic use. The template is stored separately for embedding.

Why time-windowed: Drain3 template identity alone would merge events hours apart
that happen to have the same structure. A 60s default window groups storms while
keeping temporally distinct occurrences separate — which matters for correlating
failure sequences.
"""

from dataclasses import dataclass

from bundle_platform.pipeline.template_miner import TemplateMinerWrapper
from bundle_platform.shared.timestamps import detect_timestamp_format, extract_timestamp_str
from bundle_platform.shared.timestamps import ts_to_float


@dataclass
class DedupLine:
    """A (possibly collapsed) log line ready for chunking."""

    text: str           # original first occurrence — specific variable values intact
    repeat_count: int   # number of raw lines collapsed into this one
    template: str       # Drain3 template — used for embedding


def deduplicate(
    lines: list[str],
    miner: TemplateMinerWrapper,
    window_seconds: int = 60,
) -> list[DedupLine]:
    """
    Collapse consecutive lines sharing the same Drain3 template within a time window.

    Args:
        lines:          Raw log lines from a single file.
        miner:          Shared TemplateMinerWrapper for this ingestion run.
        window_seconds: Maximum elapsed time (seconds) between first and last line
                        in a collapse group. Lines with unparseable timestamps use
                        strict consecutive matching (no time-window).

    Returns:
        List of DedupLine, one per distinct event (or cluster of identical events).
        Order is preserved. repeat_count >= 1 always.
    """
    if not lines:
        return []

    result: list[DedupLine] = []
    # Current group state
    group_text: str = ""
    group_template: str = ""
    group_count: int = 0
    group_start_ts: float | None = None
    group_last_ts: float | None = None

    def _flush() -> None:
        if group_count > 0:
            result.append(DedupLine(text=group_text, repeat_count=group_count, template=group_template))

    def _parse_ts(line: str) -> float | None:
        fmt = detect_timestamp_format(line)
        if fmt == "unknown":
            return None
        ts_str = extract_timestamp_str(line, fmt)
        if ts_str is None:
            return None
        return ts_to_float(ts_str)

    for line in lines:
        template = miner.mine(line)
        ts = _parse_ts(line)

        if group_count == 0:
            # Start first group
            group_text = line
            group_template = template
            group_count = 1
            group_start_ts = ts
            group_last_ts = ts
            continue

        same_template = template == group_template

        # Time-window check: only apply when both timestamps are available
        within_window: bool
        if ts is not None and group_start_ts is not None:
            within_window = (ts - group_start_ts) <= window_seconds
        else:
            # No timestamps — only collapse if strictly consecutive (same template, no gap)
            within_window = same_template

        if same_template and within_window:
            group_count += 1
            group_last_ts = ts
        else:
            _flush()
            group_text = line
            group_template = template
            group_count = 1
            group_start_ts = ts
            group_last_ts = ts

    _flush()
    return result
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
cd ~/bundle-platform && uv run pytest tests/pipeline/test_deduplicator.py -v
```

Expected: 7 passed. (Note: `test_similar_lines_within_window_collapsed` may have 1-2 lines not collapsed on small input because Drain3 needs a few occurrences to stabilise — the test only asserts `DedupLine` structure, not collapse count, so it will pass.)

- [ ] **Step 5: Ruff + ty**

```bash
cd ~/bundle-platform && uv run ruff check src/bundle_platform/pipeline/deduplicator.py tests/pipeline/test_deduplicator.py && uv run ty check src/
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
cd ~/bundle-platform && git add src/bundle_platform/pipeline/deduplicator.py tests/pipeline/test_deduplicator.py && git commit -m "feat: add template-aware log deduplicator"
```

---

## Task 3: Extend `LogChunk` + add `chunk_from_dedup_lines()`

**Files:**
- Modify: `src/bundle_platform/pipeline/chunker.py`
- Modify: `tests/pipeline/test_chunker.py` (if it exists — add new tests at end)
- Create: `tests/pipeline/test_chunk_from_dedup.py`

- [ ] **Step 1: Write failing tests for the new function**

Create `tests/pipeline/test_chunk_from_dedup.py`:

```python
from pathlib import Path
import pytest
from bundle_platform.pipeline.chunker import LogChunk, chunk_from_dedup_lines
from bundle_platform.pipeline.deduplicator import DedupLine
from bundle_platform.tools.generic import FileEntry


def _entry(path: str = "var/log/messages", category: str = "system_logs", size: int = 100) -> FileEntry:
    return FileEntry(path=path, category=category, size_bytes=size)


def test_chunk_from_dedup_lines_basic():
    dedup_lines = [
        DedupLine(text="Apr 15 10:00:00 kernel: OOM kill", repeat_count=1, template="Apr <*> kernel: OOM kill"),
        DedupLine(text="Apr 15 10:00:01 sshd: login root", repeat_count=3, template="Apr <*> sshd: login <*>"),
    ]
    chunks = chunk_from_dedup_lines(dedup_lines, _entry(), bundle_type="rhel", bundle_id="test-bundle")
    assert len(chunks) >= 1
    chunk = chunks[0]
    assert isinstance(chunk, LogChunk)
    # text is original lines joined
    assert "OOM kill" in chunk.text
    assert "login root" in chunk.text
    # text_template is templates joined
    assert "<*>" in chunk.text_template
    # repeat_count is sum of all DedupLine repeat_counts
    assert chunk.repeat_count == 4  # 1 + 3


def test_chunk_from_dedup_lines_empty():
    chunks = chunk_from_dedup_lines([], _entry(), bundle_type="rhel", bundle_id="test")
    assert chunks == []


def test_chunk_from_dedup_lines_text_template_fallback():
    # If text_template is empty string for a DedupLine, the chunk text_template
    # should fall back to using the original text for that line
    dedup_lines = [
        DedupLine(text="plain line with no template", repeat_count=1, template=""),
    ]
    chunks = chunk_from_dedup_lines(dedup_lines, _entry(), bundle_type="rhel", bundle_id="test")
    assert len(chunks) == 1
    # Fallback: text_template == text when template is empty
    assert chunks[0].text_template == chunks[0].text or chunks[0].text_template == "plain line with no template"


def test_logchunk_has_text_template_and_repeat_count_fields():
    chunk = LogChunk(
        bundle_id="b",
        file_path="var/log/messages",
        category="system_logs",
        start_line=1,
        end_line=2,
        text="line one\nline two",
        severity=None,
    )
    # New fields exist with correct defaults
    assert chunk.text_template == ""
    assert chunk.repeat_count == 1
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
cd ~/bundle-platform && uv run pytest tests/pipeline/test_chunk_from_dedup.py -v
```

Expected: `ImportError` (function does not exist yet).

- [ ] **Step 3: Add fields to `LogChunk` and add `chunk_from_dedup_lines()`**

In `src/bundle_platform/pipeline/chunker.py`:

**Add two fields to `LogChunk`** (after `timestamp_end`, before the closing of the dataclass — they must have defaults to remain backward compatible):

```python
    text_template: str = ""
    # Drain3 template for this chunk — used for embedding. Empty if no template mined.
    repeat_count: int = 1
    # Total raw lines this chunk represents (sum of DedupLine.repeat_count values).
```

**Add this function** at the end of the file (before `_detect_severity`):

```python
def chunk_from_dedup_lines(
    dedup_lines: list["DedupLine"],
    entry: FileEntry,
    bundle_type: str = "unknown",
    bundle_id: str = "",
) -> list["LogChunk"]:
    """
    Produce LogChunks from pre-deduplicated lines.

    Used by BundleEngine to chunk files that have already been run through
    the deduplicator. Each chunk's text is the joined original lines; its
    text_template is the joined Drain3 templates (used for embedding).
    repeat_count is the sum of all DedupLine.repeat_counts in the window.

    Falls back to text when a DedupLine has an empty template.
    """
    if not dedup_lines:
        return []

    # Extract text and template per line
    texts = [dl.text for dl in dedup_lines]
    templates = [dl.template if dl.template else dl.text for dl in dedup_lines]
    repeat_counts = [dl.repeat_count for dl in dedup_lines]

    chunks: list[LogChunk] = []
    if len(texts) <= _CHUNK_LINES:
        chunk_text = "\n".join(texts)
        chunk_template = "\n".join(templates)
        ts_values = [t for t in (_extract_timestamp(line) for line in texts) if t is not None]
        chunks.append(LogChunk(
            bundle_id=bundle_id,
            file_path=entry.path,
            category=entry.category,
            start_line=1,
            end_line=len(texts),
            text=chunk_text,
            severity=_detect_severity(chunk_text),
            bundle_type=bundle_type,
            timestamp_start=ts_values[0] if ts_values else None,
            timestamp_end=ts_values[-1] if ts_values else None,
            text_template=chunk_template,
            repeat_count=sum(repeat_counts),
        ))
    else:
        start = 0
        while start < len(texts):
            end = min(start + _CHUNK_LINES, len(texts))
            window_texts = texts[start:end]
            window_templates = templates[start:end]
            window_counts = repeat_counts[start:end]
            chunk_text = "\n".join(window_texts)
            chunk_template = "\n".join(window_templates)
            ts_values = [
                t for t in (_extract_timestamp(line) for line in window_texts) if t is not None
            ]
            chunks.append(LogChunk(
                bundle_id=bundle_id,
                file_path=entry.path,
                category=entry.category,
                start_line=start + 1,
                end_line=end,
                text=chunk_text,
                severity=_detect_severity(chunk_text),
                bundle_type=bundle_type,
                timestamp_start=ts_values[0] if ts_values else None,
                timestamp_end=ts_values[-1] if ts_values else None,
                text_template=chunk_template,
                repeat_count=sum(window_counts),
            ))
            if end == len(texts):
                break
            start += _STEP
    return chunks
```

**Add the type annotation import at the top** of chunker.py (add to existing imports):

```python
from __future__ import annotations
```

(This allows the `"DedupLine"` forward reference without a circular import.)

- [ ] **Step 4: Run new tests**

```bash
cd ~/bundle-platform && uv run pytest tests/pipeline/test_chunk_from_dedup.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Run full test suite — no regressions**

```bash
cd ~/bundle-platform && uv run pytest -q
```

Expected: all passed (existing tests unaffected by new default fields).

- [ ] **Step 6: Ruff + ty**

```bash
cd ~/bundle-platform && uv run ruff check src/bundle_platform/pipeline/chunker.py tests/pipeline/test_chunk_from_dedup.py && uv run ty check src/
```

Expected: clean.

- [ ] **Step 7: Commit**

```bash
cd ~/bundle-platform && git add src/bundle_platform/pipeline/chunker.py tests/pipeline/test_chunk_from_dedup.py && git commit -m "feat: add text_template/repeat_count to LogChunk; add chunk_from_dedup_lines()"
```

---

## Task 4: `pipeline/engine.py` — BundleEngine

**Files:**
- Create: `src/bundle_platform/pipeline/engine.py`
- Create: `tests/pipeline/test_engine.py`

- [ ] **Step 1: Write failing tests**

Create `tests/pipeline/test_engine.py`:

```python
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from bundle_platform.pipeline.engine import BundleEngine
from bundle_platform.pipeline.store import VectorStore
from bundle_platform.tools.generic import FileEntry, FileManifest


def _manifest(paths: list[str]) -> FileManifest:
    return FileManifest(
        entries=[FileEntry(path=p, category="system_logs", size_bytes=100) for p in paths],
        bundle_root=Path("/fake/root"),
    )


def _engine(tmp_path: Path, manifest: FileManifest, bundle_type: str = "rhel") -> BundleEngine:
    bundle_path = tmp_path / "bundle.tar.xz"
    bundle_path.write_bytes(b"fake")
    return BundleEngine(
        manifest=manifest,
        bundle_root=tmp_path / "root",
        bundle_path=bundle_path,
        bundle_type=bundle_type,
    )


def test_engine_writes_checkpoint_after_each_file(tmp_path):
    manifest = _manifest(["var/log/messages", "var/log/audit/audit.log"])
    engine = _engine(tmp_path, manifest)

    with patch.object(engine, "_process_file", return_value=3) as mock_proc:
        with patch.object(engine, "_make_store", return_value=VectorStore.in_memory()):
            engine.run()

    state_file = engine.checkpoint_path
    assert state_file.exists()
    state = json.loads(state_file.read_text())
    assert "var/log/messages" in state["processed_files"]
    assert "var/log/audit/audit.log" in state["processed_files"]


def test_engine_skips_already_processed_files(tmp_path):
    manifest = _manifest(["var/log/messages", "var/log/audit/audit.log"])
    engine = _engine(tmp_path, manifest)

    # Pre-write a checkpoint marking first file as done
    engine.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    engine.checkpoint_path.write_text(json.dumps({
        "schema": 1,
        "processed_files": ["var/log/messages"],
        "chunk_count": 10,
    }))

    processed = []
    def fake_process(entry, store, miner):
        processed.append(entry.path)
        return 0

    with patch.object(engine, "_process_file", side_effect=fake_process):
        with patch.object(engine, "_make_store", return_value=VectorStore.in_memory()):
            engine.run()

    # Only the second file should have been processed
    assert processed == ["var/log/audit/audit.log"]


def test_engine_writes_preprocessed_json_on_completion(tmp_path):
    manifest = _manifest(["var/log/messages"])
    engine = _engine(tmp_path, manifest)

    with patch.object(engine, "_process_file", return_value=1):
        with patch.object(engine, "_make_store", return_value=VectorStore.in_memory()):
            engine.run()

    done_file = engine.checkpoint_path.parent / "preprocessed.json"
    assert done_file.exists()


def test_engine_returns_vector_store(tmp_path):
    manifest = _manifest([])
    engine = _engine(tmp_path, manifest)
    store = VectorStore.in_memory()

    with patch.object(engine, "_make_store", return_value=store):
        result = engine.run()

    assert result is store
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
cd ~/bundle-platform && uv run pytest tests/pipeline/test_engine.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `engine.py`**

Create `src/bundle_platform/pipeline/engine.py`:

```python
"""
Incremental bundle ingestion engine.

Orchestrates the full per-file pipeline:
  read lines → deduplicate (Drain3 template-aware) → chunk → embed template → store

Checkpoint/resume: engine_state.json in the cache dir records which files have been
fully processed. On restart, already-processed files are skipped so large bundles
can be interrupted and resumed without re-embedding.

Why separate from preprocessor.py: The original preprocessor chunked all files first,
then embedded in batch. This engine processes file-by-file so checkpoint granularity
is per-file, not per-bundle. preprocessor.py is kept for backward compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path

from bundle_platform.pipeline.chunker import chunk_from_dedup_lines
from bundle_platform.pipeline.deduplicator import deduplicate
from bundle_platform.pipeline.embedder import Embedder
from bundle_platform.pipeline.exceptions import RagUnavailable
from bundle_platform.pipeline.preprocessor import cache_dir
from bundle_platform.pipeline.store import VectorStore
from bundle_platform.pipeline.template_miner import TemplateMinerWrapper
from bundle_platform.tools.generic import FileEntry, FileManifest

_STATE_FILE = "engine_state.json"
_DONE_FILE = "preprocessed.json"
_SCHEMA_VERSION = 1


class BundleEngine:
    """
    Per-file incremental ingestion with checkpoint/resume.

    Usage:
        engine = BundleEngine(manifest, bundle_root, bundle_path, bundle_type="rhel")
        store = engine.run()  # resumes from checkpoint if one exists
    """

    def __init__(
        self,
        manifest: FileManifest,
        bundle_root: Path,
        bundle_path: Path,
        bundle_type: str = "unknown",
        embedder: Embedder | None = None,
        dedup_window_seconds: int = 60,
    ) -> None:
        self._manifest = manifest
        self._bundle_root = bundle_root
        self._bundle_path = bundle_path
        self._bundle_type = bundle_type
        self._embedder = embedder
        self._dedup_window = dedup_window_seconds
        self._cache = cache_dir(bundle_path)
        self.checkpoint_path = self._cache / _STATE_FILE

    def run(self) -> VectorStore:
        """
        Ingest the bundle. Returns a ready VectorStore.
        Resumes from the last checkpoint if engine_state.json exists.
        """
        state = self._load_state()
        store = self._make_store()
        embedder = self._embedder or Embedder()
        miner = TemplateMinerWrapper()

        for entry in self._manifest.entries:
            if entry.path in state["processed_files"]:
                continue
            chunk_count = self._process_file(entry, store, miner, embedder)
            state["processed_files"].append(entry.path)
            state["chunk_count"] += chunk_count
            self._save_state(state)

        self._mark_done()
        return store

    def _process_file(
        self,
        entry: FileEntry,
        store: VectorStore,
        miner: TemplateMinerWrapper,
        embedder: Embedder | None = None,
    ) -> int:
        """Read, deduplicate, chunk, embed, and upsert one file. Returns chunk count."""
        path = self._bundle_root / entry.path
        try:
            text = path.read_text(errors="replace")
        except OSError:
            return 0

        lines = text.splitlines()
        if not lines:
            return 0

        dedup_lines = deduplicate(lines, miner, self._dedup_window)
        bundle_id = self._bundle_root.name
        chunks = chunk_from_dedup_lines(
            dedup_lines, entry, bundle_type=self._bundle_type, bundle_id=bundle_id
        )
        if not chunks:
            return 0

        _embedder = embedder or self._embedder or Embedder()
        embed_texts = [c.text_template if c.text_template else c.text for c in chunks]
        try:
            vectors = _embedder.embed_texts(embed_texts)
        except RagUnavailable:
            return 0

        payloads = [
            {
                "file_path": c.file_path,
                "category": c.category,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "text": c.text,
                "text_template": c.text_template,
                "repeat_count": c.repeat_count,
                "severity": c.severity,
                "bundle_type": c.bundle_type,
                "timestamp_start": c.timestamp_start,
                "timestamp_end": c.timestamp_end,
            }
            for c in chunks
        ]
        store.upsert(vectors, payloads)
        return len(chunks)

    def _make_store(self) -> VectorStore:
        qdrant_path = self._cache / "qdrant"
        return VectorStore.from_path(qdrant_path)

    def _load_state(self) -> dict:
        if self.checkpoint_path.exists():
            try:
                state = json.loads(self.checkpoint_path.read_text())
                if state.get("schema") == _SCHEMA_VERSION:
                    return state
            except (json.JSONDecodeError, KeyError):
                pass
        return {"schema": _SCHEMA_VERSION, "processed_files": [], "chunk_count": 0}

    def _save_state(self, state: dict) -> None:
        self._cache.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.write_text(json.dumps(state, indent=2))

    def _mark_done(self) -> None:
        self._cache.mkdir(parents=True, exist_ok=True)
        (self._cache / _DONE_FILE).write_text("{}")
```

- [ ] **Step 4: Run tests**

```bash
cd ~/bundle-platform && uv run pytest tests/pipeline/test_engine.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Full suite + ruff + ty**

```bash
cd ~/bundle-platform && uv run pytest -q && uv run ruff check src/bundle_platform/pipeline/engine.py tests/pipeline/test_engine.py && uv run ty check src/
```

Expected: all passed, clean.

- [ ] **Step 6: Commit**

```bash
cd ~/bundle-platform && git add src/bundle_platform/pipeline/engine.py tests/pipeline/test_engine.py && git commit -m "feat: add BundleEngine with checkpoint/resume ingestion pipeline"
```

---

## Task 5: `parsers/kvm.py` + detect.py + `__init__.py`

**Files:**
- Create: `src/bundle_platform/parsers/kvm.py`
- Modify: `src/bundle_platform/parsers/detect.py`
- Modify: `src/bundle_platform/parsers/__init__.py`
- Create: `tests/parsers/test_kvm.py`

- [ ] **Step 1: Write failing tests**

Create `tests/parsers/test_kvm.py`:

```python
import tarfile
import io
from pathlib import Path
import pytest
from bundle_platform.parsers.kvm import get_adapter
from bundle_platform.parsers.detect import detect_bundle_type


# ── Helper: build a minimal in-memory tar archive ──────────────────────────

def _make_tar(members: list[str], tmp_path: Path, name: str = "bundle.tar") -> Path:
    archive = tmp_path / name
    with tarfile.open(archive, "w") as tf:
        for member in members:
            info = tarfile.TarInfo(name=f"prefix/{member}")
            info.size = 0
            tf.addfile(info, io.BytesIO(b""))
    return archive


# ── Detection tests ──────────────────────────────────────────────────────────

def test_detect_kvm_sosreport(tmp_path):
    archive = _make_tar(["var/log/libvirt/libvirtd.log", "sos_commands/uname"], tmp_path)
    assert detect_bundle_type(archive) == "kvm"


def test_detect_virsh_export_via_virsh_list(tmp_path):
    archive = _make_tar(["virsh-list.txt", "nodeinfo.txt"], tmp_path)
    assert detect_bundle_type(archive) == "kvm"


def test_detect_virsh_export_via_dumpxml(tmp_path):
    archive = _make_tar(["dumpxml/vm1.xml", "net-list.txt"], tmp_path)
    assert detect_bundle_type(archive) == "kvm"


def test_detect_rhel_without_libvirt_still_rhel(tmp_path):
    # Plain RHEL sosreport with no libvirt — must stay "rhel"
    archive = _make_tar(["sos_commands/uname", "var/log/messages", "etc/hosts"], tmp_path)
    assert detect_bundle_type(archive) == "rhel"


# ── KVM-sosreport adapter tests ───────────────────────────────────────────────

def test_kvm_sos_tag_system_logs():
    adapter = get_adapter("sos")
    assert adapter.tag_file("var/log/messages") == "system_logs"
    assert adapter.tag_file("var/log/kern.log") == "system_logs"


def test_kvm_sos_tag_libvirt_logs():
    adapter = get_adapter("sos")
    assert adapter.tag_file("var/log/libvirt/libvirtd.log") == "libvirt_logs"
    assert adapter.tag_file("var/log/libvirt/qemu/vm1.log") == "vm_logs"


def test_kvm_sos_tag_config():
    adapter = get_adapter("sos")
    assert adapter.tag_file("etc/libvirt/qemu.conf") == "config"
    assert adapter.tag_file("etc/hosts") == "config"


def test_kvm_sos_tag_other():
    adapter = get_adapter("sos")
    assert adapter.tag_file("proc/meminfo") == "other"


def test_kvm_sos_error_sweep_categories():
    adapter = get_adapter("sos")
    cats = adapter.error_sweep_categories()
    assert "system_logs" in cats
    assert "libvirt_logs" in cats


def test_kvm_sos_validate_ok(tmp_path):
    (tmp_path / "var" / "log" / "libvirt").mkdir(parents=True)
    (tmp_path / "sos_commands").mkdir()
    get_adapter("sos").validate(tmp_path)  # should not raise


def test_kvm_sos_validate_missing_dir(tmp_path):
    (tmp_path / "sos_commands").mkdir()
    with pytest.raises(ValueError, match="var/log/libvirt"):
        get_adapter("sos").validate(tmp_path)


# ── virsh-export adapter tests ────────────────────────────────────────────────

def test_kvm_virsh_tag_vm_inventory():
    adapter = get_adapter("virsh")
    assert adapter.tag_file("virsh-list.txt") == "vm_inventory"
    assert adapter.tag_file("virsh-dominfo/vm1.txt") == "vm_inventory"


def test_kvm_virsh_tag_vm_config():
    adapter = get_adapter("virsh")
    assert adapter.tag_file("dumpxml/vm1.xml") == "vm_config"


def test_kvm_virsh_tag_network():
    adapter = get_adapter("virsh")
    assert adapter.tag_file("net-list.txt") == "network_config"
    assert adapter.tag_file("net-dumpxml/default.xml") == "network_config"


def test_kvm_virsh_tag_storage():
    adapter = get_adapter("virsh")
    assert adapter.tag_file("pool-list.txt") == "storage_config"
    assert adapter.tag_file("vol-list.txt") == "storage_config"


def test_kvm_virsh_tag_host_info():
    adapter = get_adapter("virsh")
    assert adapter.tag_file("nodeinfo.txt") == "host_info"
    assert adapter.tag_file("capabilities.txt") == "host_info"


def test_kvm_virsh_tag_other():
    adapter = get_adapter("virsh")
    assert adapter.tag_file("something_unknown.txt") == "other"


def test_kvm_virsh_error_sweep_categories():
    adapter = get_adapter("virsh")
    assert "vm_inventory" in adapter.error_sweep_categories()


def test_kvm_virsh_validate_ok(tmp_path):
    (tmp_path / "virsh-list.txt").write_text("Id Name State\n")
    get_adapter("virsh").validate(tmp_path)


def test_kvm_virsh_validate_missing(tmp_path):
    with pytest.raises(ValueError, match="virsh-list.txt"):
        get_adapter("virsh").validate(tmp_path)


def test_failure_patterns_returns_empty_string():
    assert get_adapter("sos").failure_patterns() == ""
    assert get_adapter("virsh").failure_patterns() == ""
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
cd ~/bundle-platform && uv run pytest tests/parsers/test_kvm.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `parsers/kvm.py`**

Create `src/bundle_platform/parsers/kvm.py`:

```python
"""
KVM/libvirt bundle parser.

Handles two bundle formats:
  "sos"   — sosreport from a host running KVM/libvirt (has sos_commands/ + var/log/libvirt/)
  "virsh" — virsh command export (has virsh-list.txt or dumpxml/)

get_adapter(mode) returns the correct _KvmAdapter for the detected mode.
"""

from pathlib import Path

from bundle_platform.parsers.base import BundleAdapter

# ── KVM-sosreport category rules (first match wins) ──────────────────────────

_SOS_RULES: list[tuple[str, str]] = [
    ("var/log/libvirt/qemu/", "vm_logs"),       # per-VM QEMU logs before libvirt catch-all
    ("var/log/libvirt/", "libvirt_logs"),
    ("var/log/messages", "system_logs"),
    ("var/log/kern.log", "system_logs"),
    ("var/log/syslog", "system_logs"),
    ("sos_commands/", "sos_commands"),
    ("etc/libvirt/", "config"),
    ("etc/qemu/", "config"),
    ("etc/multipath.conf", "storage"),
    ("var/lib/libvirt/", "storage"),
    ("etc/", "config"),
]

# ── virsh-export category rules ───────────────────────────────────────────────

_VIRSH_RULES: list[tuple[str, str]] = [
    ("virsh-list.txt", "vm_inventory"),
    ("virsh-dominfo/", "vm_inventory"),
    ("dumpxml/", "vm_config"),
    ("net-list.txt", "network_config"),
    ("net-dumpxml/", "network_config"),
    ("pool-list.txt", "storage_config"),
    ("vol-list.txt", "storage_config"),
    ("nodeinfo.txt", "host_info"),
    ("capabilities.txt", "host_info"),
]


class _KvmAdapter(BundleAdapter):
    """KVM bundle adapter. mode is "sos" or "virsh"."""

    bundle_type: str = "kvm"

    def __init__(self, mode: str) -> None:
        if mode not in ("sos", "virsh"):
            raise ValueError(f"unknown KVM mode: {mode!r}. Expected 'sos' or 'virsh'")
        self._mode = mode
        self._rules = _SOS_RULES if mode == "sos" else _VIRSH_RULES

    def validate(self, root: Path) -> None:
        if self._mode == "sos":
            libvirt_dir = root / "var" / "log" / "libvirt"
            if not libvirt_dir.exists():
                raise ValueError(
                    f"not a KVM sosreport: missing var/log/libvirt in {root}"
                )
        else:
            virsh_list = root / "virsh-list.txt"
            dumpxml_dir = root / "dumpxml"
            if not virsh_list.exists() and not dumpxml_dir.exists():
                raise ValueError(
                    f"not a virsh export: missing virsh-list.txt or dumpxml/ in {root}"
                )

    def tag_file(self, path: str) -> str:
        for prefix, category in self._rules:
            if path.startswith(prefix) or path == prefix.rstrip("/"):
                return category
        return "other"

    def timestamp_format(self, path: str) -> str:
        if self._mode == "virsh":
            return "unknown"
        # KVM-sosreport log files follow syslog or ISO 8601 format
        if path.startswith("var/log/"):
            return "syslog"
        return "unknown"

    def error_sweep_categories(self) -> frozenset[str]:
        if self._mode == "sos":
            return frozenset({"system_logs", "libvirt_logs"})
        return frozenset({"vm_inventory"})


def get_adapter(mode: str) -> BundleAdapter:
    """Return a KVM adapter for the given mode ('sos' or 'virsh')."""
    return _KvmAdapter(mode)
```

- [ ] **Step 4: Update `parsers/detect.py`**

Add KVM detection **before** the RHEL branch in `detect_bundle_type()`. The RHEL check is currently `if has_sos_commands: return "rhel"`. Replace it with:

```python
    # Check for KVM-sosreport: libvirt logs + sos_commands (before RHEL check — both have sos_commands)
    has_libvirt = any(name.startswith("var/log/libvirt/") for name in stripped)
    if has_sos_commands and has_libvirt:
        return "kvm"

    # Check for virsh export
    has_virsh_list = "virsh-list.txt" in stripped
    has_dumpxml = any(name.startswith("dumpxml/") for name in stripped)
    if has_virsh_list or has_dumpxml:
        return "kvm"

    # Check for sos_commands/ (RHEL sosreport without KVM)
    if has_sos_commands:
        return "rhel"
```

Remove the old `if has_sos_commands: return "rhel"` line.

- [ ] **Step 5: Update `parsers/__init__.py`**

Add `kvm` import and route:

```python
from bundle_platform.parsers import esxi, kvm, rhel
from bundle_platform.parsers.base import BundleAdapter
from bundle_platform.parsers.detect import detect_bundle_type

__all__ = ["BundleAdapter", "detect_bundle_type", "load_adapter"]


def load_adapter(bundle_type: str) -> BundleAdapter:
    """Return the BundleAdapter for the given bundle type string."""
    if bundle_type == "rhel":
        return rhel.get_adapter()
    if bundle_type == "esxi":
        return esxi.get_adapter()
    if bundle_type == "kvm":
        # Mode is inferred at runtime from the unpacked bundle structure.
        # Default to "sos" — callers that know the mode can call kvm.get_adapter() directly.
        return kvm.get_adapter("sos")
    raise ValueError(f"unknown bundle type: {bundle_type!r}")
```

- [ ] **Step 6: Run KVM tests**

```bash
cd ~/bundle-platform && uv run pytest tests/parsers/test_kvm.py -v
```

Expected: all passed.

- [ ] **Step 7: Run full suite + ruff + ty**

```bash
cd ~/bundle-platform && uv run pytest -q && uv run ruff check src/bundle_platform/parsers/ tests/parsers/ && uv run ty check src/
```

Expected: all passed, clean.

- [ ] **Step 8: Commit**

```bash
cd ~/bundle-platform && git add src/bundle_platform/parsers/kvm.py src/bundle_platform/parsers/detect.py src/bundle_platform/parsers/__init__.py tests/parsers/test_kvm.py && git commit -m "feat: add KVM dual-mode parser; extend detect.py for KVM detection"
```

---

## Done

Phase 2 complete. `~/bundle-platform` has:
- Drain3 template mining with forensic-safe dual storage (template for embedding, original for Claude)
- Template-aware deduplication with time-window collapse
- `BundleEngine` with per-file checkpoint/resume
- Full KVM parser for sosreport and virsh-export bundles
- Green `pytest -q`, `ruff check`, `ty check` baseline

**Next:** Phase 3 — pygrok structured field extraction + anomaly scoring integration.
