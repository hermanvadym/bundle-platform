from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
import zipfile
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

_CACHE_DIR = Path.home() / ".cache" / "bundle-platform" / "bundles"

_ARCHIVE_SUFFIXES = {".tgz", ".tar.gz", ".tar.xz", ".tar.bz2", ".zip"}


def _is_archive(path: Path) -> bool:
    return "".join(path.suffixes[-2:]) in _ARCHIVE_SUFFIXES or path.suffix in _ARCHIVE_SUFFIXES


def _file_hash(path: Path) -> str:
    """SHA-256 of the first 4 MB — fast enough for cache keying without reading huge archives."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        h.update(fh.read(4 * 1024 * 1024))
    return h.hexdigest()[:16]


def _extract_archive(archive: Path) -> Path:
    """Extract archive to a cache directory, reusing the cached copy on repeat runs.

    Cache key is <stem>-<hash16> so different archives with the same name don't collide,
    and re-running on the same archive skips extraction entirely.
    """
    key = f"{archive.stem.split('.')[0]}-{_file_hash(archive)}"
    dest = _CACHE_DIR / key
    if dest.exists():
        print(f"[bundle-platform-eval] using cached extraction: {dest}")
        return dest

    print(f"[bundle-platform-eval] extracting {archive.name} → {dest}")
    tmp = Path(tempfile.mkdtemp(prefix="bpe-"))
    try:
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(tmp)
        else:
            with tarfile.open(archive) as tf:
                tf.extractall(tmp)

        # If the archive extracted into a single top-level directory, use that as the root.
        # This handles archives like `bundle.tgz` that contain `bundle/commands/`, `bundle/var/`, …
        children = [c for c in tmp.iterdir() if not c.name.startswith(".")]
        root = children[0] if len(children) == 1 and children[0].is_dir() else tmp

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(root), dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return dest


def _null_answerer(question: str, context: str) -> str:
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bundle-platform-eval")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run eval scorecard")
    run_p.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Path to unpacked bundle directory or a .tgz/.zip archive",
    )
    run_p.add_argument("--golden", type=Path, required=True, help="Directory of golden YAML files")
    run_p.add_argument(
        "--strategies",
        default="baseline",
        help="Comma-separated strategy names (default: baseline)",
    )
    run_p.add_argument("--output", type=Path, default=None, help="Write markdown report to file")

    args = parser.parse_args(argv)

    bundle_root = args.bundle
    if _is_archive(bundle_root):
        if not bundle_root.is_file():
            print(f"Archive not found: {bundle_root}", file=sys.stderr)
            return 1
        bundle_root = _extract_archive(bundle_root)
    elif not bundle_root.is_dir():
        print(f"Bundle path is not a directory or archive: {bundle_root}", file=sys.stderr)
        return 1

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
        bundle_root=bundle_root,
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
