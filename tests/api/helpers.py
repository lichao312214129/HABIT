# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Small helpers for comparing pipeline outputs in API tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Set


def _ignored_relative_paths(root: Path, ignore_globs: Iterable[str]) -> Set[Path]:
    ignored: Set[Path] = set()
    for pattern in ignore_globs:
        for path in root.rglob(pattern):
            if path.is_file():
                ignored.add(path.relative_to(root))
    return ignored


def collect_relative_files(
    root: Path,
    *,
    suffixes: tuple[str, ...] = (".nii.gz", ".nrrd", ".csv"),
    ignore_globs: Iterable[str] = ("*.log",),
) -> Set[Path]:
    """
    Collect relative file paths under ``root`` for parity comparison.

    Args:
        root: Directory tree to scan.
        suffixes: File suffixes to include.
        ignore_globs: Glob patterns (relative to ``root``) to skip.

    Returns:
        Set of paths relative to ``root``.
    """
    ignored = _ignored_relative_paths(root, ignore_globs)
    files: Set[Path] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel in ignored:
            continue
        if suffixes and not any(path.name.endswith(s) for s in suffixes):
            continue
        files.add(rel)
    return files


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_output_trees_equal(
    left_root: Path,
    right_root: Path,
    *,
    ignore_globs: Iterable[str] = ("*.log",),
) -> None:
    """
    Assert two output directories contain identical artifact files by hash.

    Raises:
        AssertionError: When file sets or contents differ.
    """
    left_files = collect_relative_files(left_root, ignore_globs=ignore_globs)
    right_files = collect_relative_files(right_root, ignore_globs=ignore_globs)
    assert left_files == right_files, (
        f"Output file sets differ.\nOnly left: {sorted(left_files - right_files)}\n"
        f"Only right: {sorted(right_files - left_files)}"
    )
    for rel in sorted(left_files):
        left_hash = file_sha256(left_root / rel)
        right_hash = file_sha256(right_root / rel)
        assert left_hash == right_hash, f"Content mismatch for {rel}"
