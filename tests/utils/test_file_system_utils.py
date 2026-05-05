"""
Unit tests for habit.utils.file_system_utils.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from habit.utils.file_system_utils import (
    ensure_dir,
    find_files,
    list_subdirectories,
)


class TestEnsureDir:
    def test_creates_directory(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "new" / "nested"
        ensure_dir(str(new_dir))
        assert new_dir.is_dir()

    def test_existing_directory_no_error(self, tmp_path: Path) -> None:
        existing = tmp_path / "exists"
        existing.mkdir()
        ensure_dir(str(existing))  # must not raise
        assert existing.is_dir()


class TestFindFiles:
    def test_finds_files_by_extension(self, tmp_path: Path) -> None:
        (tmp_path / "a.nii.gz").write_bytes(b"")
        (tmp_path / "b.nii.gz").write_bytes(b"")
        (tmp_path / "c.txt").write_bytes(b"")
        result = find_files(str(tmp_path), pattern="*.nii.gz")
        assert len(result) == 2

    def test_recursive_search(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.nii.gz").write_bytes(b"")
        result = find_files(str(tmp_path), pattern="*.nii.gz", recursive=True)
        assert len(result) >= 1

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        result = find_files(str(tmp_path), pattern="*.nii.gz")
        assert result == []


class TestListSubdirectories:
    def test_lists_immediate_subdirs(self, tmp_path: Path) -> None:
        (tmp_path / "subA").mkdir()
        (tmp_path / "subB").mkdir()
        (tmp_path / "file.txt").write_bytes(b"")
        subdirs = list_subdirectories(str(tmp_path))
        assert len(subdirs) == 2

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        assert list_subdirectories(str(tmp_path)) == []
