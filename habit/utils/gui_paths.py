# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Resolve paths to the next-generation ``habit-gui/`` bundle shipped with HABIT.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_habit_gui_root() -> Path:
    """
    Locate the ``habit-gui/`` directory.

    Search order:
    1. ``HABIT_GUI_ROOT`` environment variable
    2. Sibling of the installed ``habit`` package (editable / source checkout)
    3. Bundled copy under ``habit/_gui_bundle`` (wheel installs)

    Returns:
        Path: Absolute path to ``habit-gui/``.

    Raises:
        FileNotFoundError: When no GUI bundle can be located.
    """
    env_root: str | None = os.environ.get("HABIT_GUI_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if _is_gui_root(root):
            return root

    import habit

    habit_pkg = Path(habit.__file__).resolve().parent
    sibling = habit_pkg.parent / "habit-gui"
    if _is_gui_root(sibling):
        return sibling

    bundled = habit_pkg / "_gui_bundle"
    if _is_gui_root(bundled):
        return bundled

    raise FileNotFoundError(
        "HABIT GUI bundle not found. Install from the repository root with "
        "'pip install -e .' or set HABIT_GUI_ROOT to your habit-gui directory."
    )


def _is_gui_root(path: Path) -> bool:
    """Return True when ``path`` looks like a valid habit-gui root."""
    return (path / "api" / "habit_gui_api" / "main.py").is_file()


def gui_api_dir(gui_root: Path) -> Path:
    """Return the FastAPI package directory (``habit-gui/api``)."""
    return gui_root / "api"


def gui_static_dir(gui_root: Path) -> Path:
    """Return the built web frontend directory (``habit-gui/web/dist``)."""
    return gui_root / "web" / "dist"


def gui_repo_root(gui_root: Path) -> Path:
    """Return the repository root (parent of ``habit-gui/``)."""
    return gui_root.parent
