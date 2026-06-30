# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
HABIT GUI project folder management (case-oriented workflow for clinicians).

A project root contains standard subfolders and ``habit_project.json`` metadata.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from habit.utils.docker_path_utils import to_runtime_path

PROJECT_FILENAME: str = "habit_project.json"

STANDARD_SUBDIRS: List[str] = [
    "01_raw",
    "02_preprocessed",
    "03_habitat",
    "04_features",
    "05_ml",
    "reports",
]


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_project_root(path: str) -> str:
    """
    Normalize a user-entered project root to an absolute runtime path.

    On WSL/Docker, Windows paths like ``F:\\work\\proj`` are converted to
    ``/mnt/f/work/proj`` before resolving so cwd is not wrongly prepended.

    Args:
        path: Project root from GUI textbox or recent-projects list.

    Returns:
        str: Absolute normalized path, or empty string when input is blank.
    """
    if not path or not str(path).strip():
        return ""
    runtime = to_runtime_path(str(path).strip())
    return str(Path(runtime).expanduser().resolve())


def default_project_dict(name: str, root: str) -> Dict[str, Any]:
    """
    Build a new project metadata document.

    Args:
        name: Human-readable project title.
        root: Absolute project root path.

    Returns:
        Dict[str, Any]: Serializable project metadata.
    """
    return {
        "name": name,
        "root": root,
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "template_id": "liver_dce_two_step",
        "wizard_step": 1,
        "paths": {
            "raw": "01_raw",
            "dicom_sorted": "01_raw/sorted_dicom",
            "preprocessed_root": "02_preprocessed",
            "preprocessed": "02_preprocessed/processed_images",
            "habitat_out": "03_habitat",
            "features": "04_features",
            "ml": "05_ml",
            "compare": "05_ml/comparison",
        },
        "workflow_state": {
            step: {"status": "pending"} for step in (
                "dicom_sort", "preprocess", "habitat", "extract", "ml", "compare"
            )
        },
        "last_run": {},
    }


def create_project(root: str, name: str) -> Dict[str, Any]:
    """
    Create project directory tree and write ``habit_project.json``.

    Args:
        root: Absolute path for the new project root.
        name: Display name shown in the GUI.

    Returns:
        Dict[str, Any]: Saved project metadata.

    Raises:
        OSError: When directories or the JSON file cannot be written.
    """
    root_norm = normalize_project_root(root)
    root_path = Path(root_norm)
    root_path.mkdir(parents=True, exist_ok=True)
    for sub in STANDARD_SUBDIRS:
        (root_path / sub).mkdir(parents=True, exist_ok=True)
    # Standard nested folders used by preprocessing and comparison outputs
    (root_path / "01_raw" / "sorted_dicom").mkdir(parents=True, exist_ok=True)
    (root_path / "02_preprocessed" / "processed_images").mkdir(parents=True, exist_ok=True)
    (root_path / "05_ml" / "comparison").mkdir(parents=True, exist_ok=True)
    meta = default_project_dict(name=name, root=root_norm)
    save_project(meta)
    return meta


def project_file_path(root: str) -> Path:
    """Return the path to ``habit_project.json`` under a project root."""
    return Path(normalize_project_root(root)) / PROJECT_FILENAME


def load_project(root: str) -> Optional[Dict[str, Any]]:
    """
    Load project metadata from disk.

    Args:
        root: Project root directory.

    Returns:
        Optional[Dict[str, Any]]: Parsed JSON or None when missing/invalid.
    """
    path = project_file_path(root)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        return None
    return None


def save_project(meta: Dict[str, Any]) -> None:
    """
    Persist project metadata and bump ``updated_at``.

    Args:
        meta: Project document containing a ``root`` key.

    Raises:
        ValueError: When ``root`` is missing.
        OSError: On write failure.
    """
    root = normalize_project_root(str(meta.get("root", "")).strip())
    if not root:
        raise ValueError("Project metadata must include 'root'.")
    meta = dict(meta)
    meta["root"] = root
    meta["updated_at"] = _utc_now_iso()
    path = project_file_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def resolve_project_path(root: str, relative_key: str, meta: Optional[Dict[str, Any]] = None) -> str:
    """
    Resolve a logical path key (e.g. ``preprocessed``) to an absolute path.

    Args:
        root: Project root directory.
        relative_key: Key inside ``paths`` or a literal relative segment.
        meta: Optional cached project metadata.

    Returns:
        str: Absolute filesystem path.
    """
    root_path = Path(normalize_project_root(root))
    document = meta or load_project(str(root_path)) or {}
    paths = document.get("paths", {}) or {}
    rel = paths.get(relative_key, relative_key)
    return str((root_path / rel).resolve())


def update_project_paths(root: str, updates: Dict[str, str]) -> Dict[str, Any]:
    """
    Merge path updates into project metadata and save.

    Args:
        root: Project root directory.
        updates: Mapping of logical keys to relative paths.

    Returns:
        Dict[str, Any]: Updated project metadata.
    """
    meta = load_project(root) or default_project_dict(
        name=Path(normalize_project_root(root)).name,
        root=normalize_project_root(root),
    )
    paths = dict(meta.get("paths", {}) or {})
    paths.update(updates)
    meta["paths"] = paths
    save_project(meta)
    return meta


def list_recent_projects(max_items: int = 8) -> List[str]:
    """
    List recently modified project roots known to the GUI draft store.

    Args:
        max_items: Maximum entries to return.

    Returns:
        List[str]: Absolute project root paths, newest first.
    """
    from habit.gui.utils import gui_draft_dir

    recent_file = gui_draft_dir() / "recent_projects.json"
    if not recent_file.is_file():
        return []
    try:
        with recent_file.open("r", encoding="utf-8") as handle:
            items = json.load(handle)
        if not isinstance(items, list):
            return []
        valid = [str(p) for p in items if p and os.path.isdir(str(p))]
        return valid[:max_items]
    except (OSError, json.JSONDecodeError):
        return []


def register_recent_project(root: str) -> None:
    """Push a project root to the recent-projects list."""
    from habit.gui.utils import gui_draft_dir

    root = normalize_project_root(root)
    items = list_recent_projects(max_items=20)
    items = [root] + [p for p in items if p != root]
    recent_file = gui_draft_dir() / "recent_projects.json"
    try:
        recent_file.parent.mkdir(parents=True, exist_ok=True)
        with recent_file.open("w", encoding="utf-8") as handle:
            json.dump(items[:20], handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except OSError:
        pass
