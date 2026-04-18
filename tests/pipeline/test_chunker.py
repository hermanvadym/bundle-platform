from bundle_platform.pipeline.chunker import LogChunk, chunk_file
from bundle_platform.tools.generic import FileEntry


def test_log_chunk_fields():
    chunk = LogChunk(
        bundle_id="test-bundle",
        file_path="var/log/messages",
        category="system_logs",
        severity="error",
        text="kernel: oom kill",
        bundle_type="rhel",
        start_line=1,
        end_line=1,
        timestamp_start=None,
        timestamp_end=None,
    )
    assert chunk.file_path == "var/log/messages"
    assert chunk.bundle_type == "rhel"


def test_chunk_file_returns_chunks(tmp_path):
    log = tmp_path / "messages"
    log.write_text("\n".join(f"Apr 15 02:31:0{i} host kernel: event {i}" for i in range(10)))
    entry = FileEntry(path="messages", size_bytes=log.stat().st_size, category="system_logs")
    chunks = chunk_file(tmp_path, entry, bundle_type="rhel")
    assert len(chunks) > 0
    assert all(isinstance(c, LogChunk) for c in chunks)
