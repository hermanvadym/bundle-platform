import tarfile
from pathlib import Path

from bundle_platform.tools.generic import FileEntry, FileManifest, index_files


def _make_bundle(tmp: Path) -> Path:
    bundle = tmp / "test.tar.gz"
    src = tmp / "src"
    src.mkdir()
    (src / "var" / "log").mkdir(parents=True)
    (src / "var" / "log" / "messages").write_text("line1\nline2\n")
    with tarfile.open(bundle, "w:gz") as tf:
        tf.add(src, arcname="sosreport-test")
    return bundle


def test_file_entry_fields():
    entry = FileEntry(path="var/log/messages", size_bytes=100, category="system_logs")
    assert entry.path == "var/log/messages"
    assert entry.size_bytes == 100
    assert entry.category == "system_logs"


def test_index_files_returns_manifest(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "var" / "log").mkdir(parents=True)
    (root / "var" / "log" / "messages").write_text("hello\n")

    def tagger(path: str) -> str:
        return "system_logs"

    manifest = index_files(root, tagger)
    assert isinstance(manifest, FileManifest)
    assert any("messages" in e.path for e in manifest.entries)
