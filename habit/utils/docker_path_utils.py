# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
Host ↔ container path bridge for HABIT Docker on Windows.

Doctors enter or browse paths exactly as in File Explorer (``F:\\work\\data``).
Pipeline code always receives native runtime paths (``/mnt/f/work/data`` inside
the container). Display helpers reverse the mapping for text boxes and YAML.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

# Windows drive letter path: F:\foo, F:/foo, F:foo
_WIN_DRIVE_PATH_RE = re.compile(r"^([A-Za-z]):[/\\]?(.*)$")

# Drive letter marker anywhere in a string (used to recover wrongly joined paths).
_WIN_DRIVE_INFIX_RE = re.compile(r"([A-Za-z]:[/\\])")

# Known Docker volume shortcuts (docker-compose bind mounts).
_DOCKER_SHORTCUTS: Tuple[Tuple[str, str], ...] = (
    ("/data", "data"),
    ("/config", "config"),
    ("/output", "output"),
)


def is_docker_runtime() -> bool:
    """
    Return True when HABIT is executing inside a Docker container.

    Returns:
        bool: True if ``HABIT_DOCKER=1`` or ``/.dockerenv`` exists.
    """
    flag = os.environ.get("HABIT_DOCKER", "").strip().lower()
    if flag in ("1", "true", "yes"):
        return True
    return Path("/.dockerenv").exists()


def _is_wsl() -> bool:
    """
    Return True when the current process runs inside Windows Subsystem for Linux.

    Returns:
        bool: True if a WSL environment is detected.
    """
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        with open("/proc/version", encoding="utf-8", errors="ignore") as handle:
            return "microsoft" in handle.read().lower()
    except OSError:
        return False


def uses_windows_mount_path_bridge() -> bool:
    """
    Return True when GUI paths should map between ``F:\\\\...`` and ``/mnt/<drive>/...``.

    Docker on a Windows host and WSL both expose host drives under ``/mnt``; doctors
    should see and type Windows-style paths while pipeline code uses runtime mount paths.

    Returns:
        bool: True in Docker (Windows host) and WSL; False elsewhere.
    """
    if is_docker_runtime():
        return os.environ.get("HABIT_HOST_OS", "windows").strip().lower() == "windows"
    return _is_wsl()


def get_mount_root() -> str:
    """
    Return the container prefix where Windows drive letters are bind-mounted.

    Returns:
        str: Mount root, default ``/mnt`` (``HABIT_DOCKER_MOUNT_ROOT``).
    """
    root = os.environ.get("HABIT_DOCKER_MOUNT_ROOT", "/mnt").strip() or "/mnt"
    return root.rstrip("/")


def parse_windows_drive_path(path: str) -> Optional[Tuple[str, str]]:
    """
    Parse a Windows drive path into (drive_letter_lower, remainder_posix).

    Args:
        path: Raw user path string.

    Returns:
        Optional[Tuple[str, str]]: ``("f", "work/demo_data/dicom")`` or None.
    """
    if not path or not str(path).strip():
        return None
    raw = str(path).strip()
    match = _WIN_DRIVE_PATH_RE.match(raw)
    if not match:
        return None
    drive = match.group(1).lower()
    remainder = match.group(2).replace("\\", "/").strip("/")
    return drive, remainder


def recover_embedded_windows_drive_path(path: str) -> str:
    """
    Recover a Windows drive path after WSL ``Path.resolve()`` wrongly prefixed cwd.

    Examples::

        C:\\Users\\foo\\F:\\work\\proj  →  F:\\work\\proj
        /mnt/c/Users/foo/F:\\work\\proj  →  F:\\work\\proj

    Args:
        path: Raw path string from GUI textbox or project metadata.

    Returns:
        str: Best-effort Windows drive path, or the original string when unchanged.
    """
    raw = str(path).strip()
    if not raw:
        return raw
    matches = list(_WIN_DRIVE_INFIX_RE.finditer(raw))
    if len(matches) >= 2:
        return raw[matches[-1].start() :]
    if len(matches) == 1 and not raw.startswith(matches[0].group(0)):
        return raw[matches[0].start() :]
    return raw


def is_windows_drive_path(path_value: str) -> bool:
    """
    Return True when ``path_value`` looks like a Windows drive path.

    Args:
        path_value: Candidate path string.

    Returns:
        bool: True for ``C:\\...`` or ``F:/...`` style paths.
    """
    return parse_windows_drive_path(path_value or "") is not None


def windows_path_to_container_mount(path: str, mount_root: Optional[str] = None) -> Optional[str]:
    """
    Map a Windows drive path to a bind-mounted path inside the container.

    Args:
        path: Windows path such as ``F:\\work\\habit\\demo_data``.
        mount_root: Container mount prefix for host drives.

    Returns:
        Optional[str]: Container path such as ``/mnt/f/work/habit/demo_data``.
    """
    parsed = parse_windows_drive_path(path)
    if parsed is None:
        return None
    root = (mount_root or get_mount_root()).rstrip("/")
    drive, remainder = parsed
    base = f"{root}/{drive}"
    if remainder:
        return f"{base}/{remainder}"
    return base


def container_mount_to_windows_path(path: str, mount_root: Optional[str] = None) -> Optional[str]:
    """
    Map a bind-mounted container path back to a Windows drive path for display.

    Args:
        path: Container path such as ``/mnt/f/work/demo``.
        mount_root: Mount prefix used for drive letters.

    Returns:
        Optional[str]: Windows path such as ``F:\\work\\demo``, or None if not under a mount.
    """
    if not path:
        return None
    root = (mount_root or get_mount_root()).rstrip("/")
    norm = path.replace("\\", "/")
    prefix = f"{root}/"
    if not norm.startswith(prefix):
        return None
    tail = norm[len(prefix) :]
    if not tail:
        return None
    parts = tail.split("/", 1)
    drive = parts[0]
    if len(drive) != 1 or not drive.isalpha():
        return None
    remainder = parts[1] if len(parts) > 1 else ""
    if remainder:
        return f"{drive.upper()}:\\" + remainder.replace("/", "\\")
    return f"{drive.upper()}:\\"


def to_runtime_path(path: str, mount_root: Optional[str] = None) -> str:
    """
    Convert a user-facing path to a path usable by ``open()`` in the current process.

    On Docker + Windows host or WSL: ``F:\\data`` → ``/mnt/f/data``.
    On native Windows: normal ``Path.resolve()``.
    Relative paths: resolved against ``os.getcwd()``.

    Args:
        path: Path from GUI, CLI, or YAML.
        mount_root: Optional override for drive mount prefix.

    Returns:
        str: Runtime absolute path string.
    """
    if not path or not str(path).strip():
        return path

    raw = recover_embedded_windows_drive_path(str(path).strip())
    root = mount_root or get_mount_root()

    if uses_windows_mount_path_bridge():
        container_path = windows_path_to_container_mount(raw, mount_root=root)
        if container_path is not None:
            return container_path
        if raw.startswith(f"{root}/") and os.path.isabs(raw.replace("\\", "/")):
            return raw.replace("\\", "/")

    if os.path.isabs(raw):
        return os.path.normpath(raw)
    return os.path.abspath(raw)


def to_user_visible_path(path: str, mount_root: Optional[str] = None) -> str:
    """
    Convert an internal runtime path to what the doctor expects to see (Windows style).

    On Docker (Windows host) and WSL, ``/mnt/f/work/data`` is shown as ``F:\\work\\data``.

    Args:
        path: Runtime path (e.g. ``/mnt/f/work/data`` or ``F:\\work\\data``).
        mount_root: Optional mount prefix override.

    Returns:
        str: Display-friendly path (``F:\\work\\data`` when mappable).
    """
    if not path or not str(path).strip():
        return path

    raw = recover_embedded_windows_drive_path(str(path).strip())
    runtime = to_runtime_path(raw, mount_root=mount_root)
    if runtime != raw:
        raw = runtime

    if uses_windows_mount_path_bridge():
        win = container_mount_to_windows_path(raw, mount_root=mount_root)
        if win is not None:
            return win
        if is_windows_drive_path(raw):
            return format_windows_path(raw)

    if os.name == "nt" and is_windows_drive_path(raw):
        return format_windows_path(raw)

    return raw


def format_windows_path(path: str) -> str:
    """
    Normalize a Windows path to ``F:\\foo\\bar`` backslash form for display.

    Args:
        path: Windows or mixed-separator path.

    Returns:
        str: Normalized display path.
    """
    parsed = parse_windows_drive_path(path)
    if parsed is None:
        return path.replace("/", "\\") if os.name == "nt" else path
    drive, remainder = parsed
    if remainder:
        return f"{drive.upper()}:\\" + remainder.replace("/", "\\")
    return f"{drive.upper()}:\\"


def normalize_gui_path(path: str, mount_root: Optional[str] = None) -> str:
    """Backward-compatible alias for :func:`to_runtime_path`."""
    return to_runtime_path(path, mount_root=mount_root)


def list_docker_browse_roots() -> List[str]:
    """
    Return directory roots available for in-browser folder picking inside Docker.

    Returns:
        List[str]: Existing roots (drive mounts under ``/mnt``, plus ``/data`` etc.).
    """
    roots: List[str] = []
    mount = get_mount_root()
    if os.path.isdir(mount):
        try:
            for name in sorted(os.listdir(mount)):
                candidate = os.path.join(mount, name)
                if len(name) == 1 and name.isalpha() and os.path.isdir(candidate):
                    roots.append(candidate)
        except OSError:
            pass
    for shortcut, _ in _DOCKER_SHORTCUTS:
        if os.path.isdir(shortcut):
            roots.append(shortcut)
    return roots or ["/"]


def docker_browse_root() -> str:
    """
    Default root for Gradio FileExplorer in Docker (parent of all drive mounts).

    Returns:
        str: ``/mnt`` when present, else ``/``.
    """
    mount = get_mount_root()
    return mount if os.path.isdir(mount) else "/"


# Common YAML / GUI field names that hold filesystem paths.
DEFAULT_PATH_FIELD_NAMES: Tuple[str, ...] = (
    "data_dir",
    "out_dir",
    "output_dir",
    "raw_img_folder",
    "habitats_map_folder",
    "csv_path",
    "pipeline_path",
    "train_ids_path",
    "test_ids_path",
    "dcm2niix_path",
    "elastix_path",
    "transformix_path",
    "checkpoint_dir",
    "voxel_params_file",
    "supervoxel_params_file",
)


def display_path_value(field_name: str, value: object) -> object:
    """
    If ``value`` is a path field, convert runtime path to user-visible form for GUI widgets.

    Args:
        field_name: YAML / form field name.
        value: Field value from loaded config.

    Returns:
        object: Display value (paths converted; others unchanged).
    """
    if not isinstance(value, str) or not value.strip():
        return value
    if field_name in DEFAULT_PATH_FIELD_NAMES or field_name.endswith(
        ("_dir", "_path", "_folder", "_file")
    ):
        return to_user_visible_path(value)
    return value
