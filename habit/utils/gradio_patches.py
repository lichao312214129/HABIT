# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
Gradio runtime patches for HABIT GUI.

Gradio's ``FileExplorer.ls`` only catches ``FileNotFoundError`` and
``PermissionError`` around ``os.listdir``. On WSL/Docker bind mounts, some
directories raise ``OSError: [Errno 5] Input/output error`` and crash the ASGI
server with HTTP 500. These helpers return empty listings instead.
"""

from __future__ import annotations

import fnmatch
import os
from typing import List, Optional

from gradio.components.base import server
from gradio.components.file_explorer import FileExplorer

_PATCHED_ATTR = "_habit_ls_patched"


def safe_listdir(path: str) -> Optional[List[str]]:
    """
    List a directory without raising on common filesystem errors.

    Args:
        path: Absolute directory path to inspect.

    Returns:
        Optional[List[str]]: Sorted entry names when readable; ``[]`` when the
        path is missing or permission is denied; ``None`` when the path exists
        but ``os.listdir`` fails with ``OSError`` (e.g. WSL mount I/O error).
    """
    if not path:
        return []
    try:
        return sorted(os.listdir(path))
    except (FileNotFoundError, PermissionError):
        return []
    except OSError:
        return None


@server
def _habit_file_explorer_ls(
    self: FileExplorer,
    subdirectory: list[str] | None = None,
) -> list[dict[str, str]] | None:
    """
    Drop-in replacement for ``FileExplorer.ls`` with broader ``listdir`` handling.

    Args:
        self: Gradio ``FileExplorer`` instance.
        subdirectory: Relative path segments under ``root_dir``.

    Returns:
        list[dict[str, str]] | None: Folder/file metadata for the explorer UI.
    """
    if subdirectory is None:
        subdirectory = []

    full_subdir_path = self._safe_join(subdirectory)
    subdir_items = safe_listdir(full_subdir_path)
    if subdir_items is None:
        return []

    files: List[dict[str, str]] = []
    folders: List[dict[str, str]] = []
    for item in subdir_items:
        full_path = os.path.join(full_subdir_path, item)

        try:
            is_file = not os.path.isdir(full_path)
        except (PermissionError, OSError):
            continue

        valid_by_glob = fnmatch.fnmatch(full_path, self.glob)

        if is_file and not valid_by_glob:
            continue

        if self.ignore_glob and fnmatch.fnmatch(full_path, self.ignore_glob):
            continue

        target = files if is_file else folders
        target.append(
            {
                "name": item,
                "type": "file" if is_file else "folder",
                "valid": valid_by_glob,
            }
        )

    return folders + files


def apply_gradio_patches() -> None:
    """
    Patch Gradio ``FileExplorer.ls`` once at GUI startup.

    The patched method keeps the public name ``ls`` and ``_is_server_fn`` so
    Gradio 6 front-end RPC routing continues to work.
    """
    if getattr(FileExplorer, _PATCHED_ATTR, False):
        return

    FileExplorer.ls = _habit_file_explorer_ls  # type: ignore[method-assign]
    setattr(FileExplorer, _PATCHED_ATTR, True)
