# Design: Phase 2 — Incremental Ingestion Engine + KVM Parser

**Date:** 2026-04-18
**Status:** Approved
**Scope:** `pipeline/engine.py`, `pipeline/template_miner.py`, `pipeline/deduplicator.py`, `parsers/kvm.py`

---

## Goal

Ship two independent units:

1. **Incremental ingestion engine** — replaces `preprocessor.py` as the orchestrator. Processes bundles file-by-file with checkpoint/resume, Drain3 template mining, and template-aware deduplication. Preserves original log text so forensic specificity is never lost.

2. **KVM parser** — full `BundleAdapter` implementation for KVM/libvirt bundles in two modes: KVM-sosreport (RHEL host running KVM) and virsh-export (structured virsh command output).

---

## Why Drain3 Does Not Hurt Root Cause Analysis

A key design decision: Drain3 strips variable tokens (device IDs, PIDs, hostnames) from log lines for embedding. This would destroy forensic specificity if the stripped text were returned to Claude. It is not.

The split:

| Field | Used for | Contains |
|---|---|---|
| `text_template` | `fastembed` → Qdrant vector | `"NMP: ... dev <*> on path <*>"` |
| `text` | Qdrant payload → Claude's tools | `"NMP: ... dev \"naa.600...\" on path \"vmhba...\""` |

Retrieval uses `text_template` vectors (better semantic clustering). Claude reads `text` (original, with all specific values intact). Drain3 only affects which chunks get surfaced — never what Claude reads.

`repeat_count` in the payload gives Claude frequency context: "this NMP error was collapsed from 847 identical lines in 10 minutes" is a meaningful RCA signal.

---

## Architecture

```
BundleEngine.run()
  └── for each file not in checkpoint:
        raw_lines = read(file)
        dedup_lines = deduplicator.deduplicate(raw_lines, miner)   # uses Drain3 template as key
        chunks = chunker.chunk_lines(dedup_lines, file_meta)       # LogChunk gains text_template + repeat_count
        vectors = embedder.embed([c.text_template for c in chunks])
        store.upsert(chunks, vectors)
        checkpoint.mark_done(file)
  └── write preprocessed.json  # existing done marker
```

---

## Unit 1: `pipeline/template_miner.py`

Thin stateful wrapper around Drain3's `TemplateMiner`. One instance per ingestion run; its tree updates incrementally as lines are fed in.

```python
class TemplateMinerWrapper:
    def mine(self, line: str) -> str: ...
    # Returns template string, e.g. "NMP: nmp_ThrottleLogForDevice: Cmd <*> to dev <*>"
    # Updates Drain3's internal tree as a side effect.
```

**Configuration:** Drain3 defaults (depth=4, max_children=100, sim_threshold=0.4). No custom config file needed for Phase 2.

---

## Unit 2: `pipeline/deduplicator.py`

Collapses repetitive log lines using the Drain3 template as the dedup key.

```python
@dataclass
class DedupLine:
    text: str          # original first occurrence — specific values preserved
    repeat_count: int  # how many lines were collapsed into this one
    template: str      # Drain3 template — used for embedding

def deduplicate(
    lines: list[str],
    miner: TemplateMinerWrapper,
    window_seconds: int = 60,
) -> list[DedupLine]: ...
```

**Collapse strategy:**
- Consecutive lines sharing the same Drain3 template AND within `window_seconds` of each other → collapsed into one `DedupLine(text=first_occurrence, repeat_count=N, template=T)`
- Lines with no parseable timestamp → collapse only if strictly consecutive (no time-window applied)
- `window_seconds=60` default; callers can override per file type

---

## Unit 3: `pipeline/engine.py`

`BundleEngine` replaces `preprocessor.py` as the ingestion orchestrator. `preprocessor.py` is kept for backward compatibility but marked deprecated.

### Checkpoint

State file: `~/.cache/bundle-platform/<bundle-id>/engine_state.json`

```json
{
  "schema": 1,
  "processed_files": ["var/log/messages", "var/log/audit/audit.log"],
  "chunk_count": 1842
}
```

On startup: files already in `processed_files` are skipped. On each file completion: state is written to disk before moving to the next file (crash-safe). On full completion: writes `preprocessed.json` so `is_preprocessed()` still works.

### Interface

```python
class BundleEngine:
    def __init__(
        self,
        manifest: FileManifest,
        bundle_root: Path,
        bundle_path: Path,
        bundle_type: str = "unknown",
        embedder: Embedder | None = None,
        dedup_window_seconds: int = 60,
    ) -> None: ...

    def run(self) -> VectorStore: ...
    # Returns a ready VectorStore. Resumes from checkpoint if one exists.
```

---

## Unit 4: `LogChunk` changes

Two new fields added to `LogChunk` in `pipeline/chunker.py` with defaults (backward compatible):

```python
@dataclass
class LogChunk:
    # ... existing fields ...
    text_template: str = ""   # Drain3 template; empty string = no template mined
    repeat_count: int = 1     # lines collapsed into this chunk by deduplicator
```

`text_template = ""` signals that no template was mined (e.g. non-log files, structured text). In that case, `engine.py` falls back to embedding `text` directly.

---

## Unit 5: `parsers/kvm.py`

Single `_KvmAdapter(BundleAdapter)` handling two modes detected at instantiation.

### Detection (added to `parsers/detect.py`)

KVM branch is checked **before** RHEL — a KVM-sosreport also has `sos_commands/`, so order matters.

| Signal | Means |
|---|---|
| `var/log/libvirt/` AND `sos_commands/` | KVM-sosreport mode |
| `virsh-list.txt` OR `dumpxml/` | virsh-export mode |

### KVM-sosreport categories

| Category | Paths matched |
|---|---|
| `system_logs` | `var/log/messages`, `var/log/kern.log`, `var/log/syslog` |
| `libvirt_logs` | `var/log/libvirt/libvirtd.log`, `var/log/libvirt/qemu/*.log` |
| `vm_logs` | `var/log/libvirt/qemu/` (per-VM logs) |
| `sos_commands` | `sos_commands/` |
| `config` | `etc/libvirt/`, `etc/qemu/` |
| `storage` | `etc/multipath.conf`, `var/lib/libvirt/` |
| `other` | everything else |

### virsh-export categories

| Category | Paths matched |
|---|---|
| `vm_inventory` | `virsh-list.txt`, `virsh-dominfo/` |
| `vm_config` | `dumpxml/` |
| `network_config` | `net-list.txt`, `net-dumpxml/` |
| `storage_config` | `pool-list.txt`, `vol-list.txt` |
| `host_info` | `nodeinfo.txt`, `capabilities.txt` |
| `other` | everything else |

### Other methods

- `error_sweep_categories()`: `frozenset({"system_logs", "libvirt_logs"})` for sosreport; `frozenset({"vm_inventory"})` for virsh
- `timestamp_format(path)`: delegates to RHEL syslog/ISO detection for sosreport log files; `"unknown"` for virsh-export files
- `validate(root)`: checks for `var/log/libvirt/` (sosreport) or `virsh-list.txt`/`dumpxml/` (virsh); raises `ValueError` if neither found
- `failure_patterns()`: returns `""` (stubbed until Phase 7)

### Routing

`parsers/__init__.py`: add `"kvm"` branch in `load_adapter()` → `kvm.get_adapter(mode)`.
`parsers/detect.py`: add KVM branch before RHEL branch.

---

## New Dependencies

| Package | Why | Where |
|---|---|---|
| `drain3` | Template mining | `pipeline/template_miner.py` |

Add to `pyproject.toml` via `uv add drain3`. No other new dependencies.

---

## Testing Strategy

- `tests/pipeline/test_template_miner.py` — mine known log lines, assert templates match expected patterns; verify tree updates across multiple calls
- `tests/pipeline/test_deduplicator.py` — collapse identical-template lines within window; preserve distinct templates; handle no-timestamp lines; verify `repeat_count` and `text` (first occurrence) correctness
- `tests/pipeline/test_engine.py` — checkpoint written after each file; resume skips processed files; completion writes `preprocessed.json`; uses in-memory VectorStore and mock embedder
- `tests/parsers/test_kvm.py` — sosreport mode detection and category tagging; virsh-export mode detection and category tagging; `validate()` raises on unrecognized structure; `error_sweep_categories()` returns correct set per mode

---

## What This Phase Does NOT Include

- pygrok structured extraction (Phase 3)
- Loglizer anomaly scoring (Phase 3+)
- KVM failure patterns / detectors (Phase 7)
- Multi-bundle batch ingestion (not needed — single-bundle incremental covers the use case)
