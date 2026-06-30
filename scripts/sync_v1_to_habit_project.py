#!/usr/bin/env python3
"""Sync habit_project_v1 -> habit_project (exclude private graph_features plugin only)."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

SRC = Path(r"F:\work\habit_project_v1")
DST = Path(r"F:\work\habit_project")

SKIP_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    "outputs",
    "graph_features",
    "build_test_render",
    "build_bold_check",
    "build_html_verify",
    "build_html_audit",
    "build_html_check2",
    "build_html_check",
}

SKIP_FILES = {
    "test_habitat_graph_features.py",
}

SKIP_SUFFIXES = {".pyc", ".pyo"}

DOCS_GRAPH_PREFIX = "source/reference/features/graph/"


def should_skip(path: Path, base: Path) -> bool:
    """Return True when a source path must not be copied to the public tree."""
    rel = path.relative_to(base).as_posix()
    if rel.startswith(DOCS_GRAPH_PREFIX):
        return True
    if any(part in SKIP_DIR_NAMES for part in path.parts):
        return True
    if path.name in SKIP_FILES:
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    return False


def _file_hash(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_copy2(src_file: Path, dst_file: Path) -> bool:
    """Copy file; skip when identical; tolerate transient locks."""
    if dst_file.exists():
        try:
            if _file_hash(src_file) == _file_hash(dst_file):
                return False
        except OSError:
            pass
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src_file, dst_file)
        return True
    except PermissionError:
        tmp = dst_file.with_suffix(dst_file.suffix + ".sync_tmp")
        shutil.copy2(src_file, tmp)
        if dst_file.exists():
            dst_file.unlink()
        tmp.replace(dst_file)
        return True


def copy_tree(rel: str) -> int:
    """Copy one top-level subtree from v1 to public, honoring skip rules."""
    src_root = SRC / rel
    dst_root = DST / rel
    if not src_root.exists():
        return 0
    count = 0
    for src_file in src_root.rglob("*"):
        if not src_file.is_file():
            continue
        if should_skip(src_file, src_root):
            continue
        rel_file = src_file.relative_to(src_root)
        dst_file = dst_root / rel_file
        if safe_copy2(src_file, dst_file):
            count += 1
    return count


def main() -> None:
    totals = 0
    for rel in ("habit", "config", "tests", "scripts", "developer", "demo_data/results_config_test"):
        copied = copy_tree(rel)
        print(f"copied {copied} files from {rel}")
        totals += copied

    for rel in (
        "pyproject.toml",
        "requirements.txt",
        "requirements-gpu.txt",
        "poetry.lock",
        "MANIFEST.in",
        ".pre-commit-config.yaml",
        ".gitattributes",
        "count_lines.py",
        "del_some_dir.py",
        "dicomsort.py",
        "README.md",
        "README_en.md",
    ):
        src = SRC / rel
        if src.is_file():
            dst = DST / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            safe_copy2(src, dst)
            totals += 1

    docs_src = SRC / "docs"
    docs_dst = DST / "docs"
    doc_count = 0
    for src_file in docs_src.rglob("*"):
        if not src_file.is_file():
            continue
        rel_file = src_file.relative_to(docs_src)
        rel_posix = rel_file.as_posix()
        if rel_posix.startswith("build"):
            continue
        if rel_posix.startswith(DOCS_GRAPH_PREFIX):
            continue
        if "__pycache__" in rel_file.parts:
            continue
        dst_file = docs_dst / rel_file
        if safe_copy2(src_file, dst_file):
            doc_count += 1
    print(f"copied {doc_count} doc files")
    totals += doc_count

    print(f"total copied: {totals}")
    _strip_graph_dependency_from_public_pyproject()
    print("Run scripts/check_public_leak.py against habit_project to verify no graph leak.")


def _strip_graph_dependency_from_public_pyproject() -> None:
    """Remove optional graph/networkx entries from the public pyproject copy."""
    pyproject = DST / "pyproject.toml"
    if not pyproject.is_file():
        return
    lines = pyproject.read_text(encoding="utf-8").splitlines(keepends=True)
    out: list[str] = []
    skip_extras_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("networkx"):
            continue
        if stripped == "[tool.poetry.extras]":
            skip_extras_block = True
            continue
        if skip_extras_block:
            if stripped.startswith("["):
                skip_extras_block = False
                out.append(line)
            continue
        out.append(line)
    pyproject.write_text("".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
