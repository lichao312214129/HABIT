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
"""Fail-fast writable probes and atomic replace helpers for result paths.

Long habitat / ML runs often discover ``Permission denied`` only at the final
``StudyResult.save`` step. Callers should probe the destination directory
(and any files that will be overwritten) before expensive work, and writers
should replace destinations atomically so a failed write cannot leave a
truncated final artefact.
"""

from __future__ import annotations

import errno
import os
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

from habit.exceptions import HABITAPIError

__all__ = [
    "is_filesystem_permission_error",
    "unwritable_destination_message",
    "probe_writable_directory",
    "raise_unwritable_destination",
    "write_via_temp_then_replace",
]

PathLike = Union[str, Path]

#: Marker bytes written by :func:`probe_writable_directory` (never kept).
_PROBE_PAYLOAD = b"habit-write-probe\n"


def is_filesystem_permission_error(exc: BaseException) -> bool:
    """
    Return whether ``exc`` indicates a filesystem permission / ACL failure.

    SimpleITK often surfaces access problems as ``RuntimeError`` whose message
    contains ``Permission denied`` rather than a native ``PermissionError``.

    Args:
        exc: Exception raised by open / replace / SimpleITK WriteImage / etc.

    Returns:
        ``True`` when the failure is permission-related and should be wrapped
        into an actionable :class:`~habit.exceptions.HABITAPIError`.
    """
    if isinstance(exc, PermissionError):
        return True
    if isinstance(exc, OSError):
        if getattr(exc, "errno", None) in (errno.EACCES, errno.EPERM):
            return True
        # Windows: winerror 5 == ERROR_ACCESS_DENIED
        if getattr(exc, "winerror", None) == 5:
            return True
    message = str(exc).lower()
    needles = (
        "permission denied",
        "access is denied",
        "access denied",
        "operation not permitted",
        "read-only file system",
    )
    return any(needle in message for needle in needles)


def unwritable_destination_message(path: PathLike) -> str:
    """
    Build an actionable English error for an unwritable save destination.

    Args:
        path: Directory or file that could not be written / overwritten.

    Returns:
        Multi-line message including the path and recovery hints (delete /
        rename / new ``out_dir`` / Windows ACL).
    """
    resolved = str(Path(path))
    return (
        f"Cannot write results to {resolved}: permission denied or the path "
        "is locked / read-only.\n"
        "Before re-running a long analysis, fix write access:\n"
        "  - delete or rename existing files that HABIT would overwrite;\n"
        "  - choose a new writable out_dir; or\n"
        "  - on Windows, check folder ACLs / clear the read-only attribute "
        "(Properties -> uncheck Read-only) and close programs that have the "
        "file open."
    )


def raise_unwritable_destination(
    path: PathLike, *, cause: Optional[BaseException] = None
) -> None:
    """
    Raise :class:`~habit.exceptions.HABITAPIError` for an unwritable path.

    Args:
        path: Directory or file that failed the write probe / write attempt.
        cause: Optional underlying OS / SimpleITK exception.

    Raises:
        HABITAPIError: Always.
    """
    message = unwritable_destination_message(path)
    if cause is None:
        raise HABITAPIError(message)
    raise HABITAPIError(message) from cause


def probe_writable_directory(
    directory: PathLike,
    *,
    existing_paths: Optional[Sequence[PathLike]] = None,
) -> Path:
    """
    Fail fast when ``directory`` cannot accept new files or needed overwrites.

    Creates ``directory`` when missing, writes a tiny temporary probe file and
    removes it. For each path in ``existing_paths`` that already exists, opens
    the file for update so a later atomic ``os.replace`` is not the first time
    the ACL / read-only problem appears. Existing artefacts are never deleted
    by the probe.

    Args:
        directory: Destination directory (e.g. recipe ``out_dir``).
        existing_paths: Optional files that will be overwritten by the save.
            Non-existent paths are ignored; only real files are probed.

    Returns:
        The resolved destination directory.

    Raises:
        HABITAPIError: When the directory or an existing destination file is
            not writable, with path and recovery guidance in the message.
    """
    root = Path(directory)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if is_filesystem_permission_error(exc):
            raise_unwritable_destination(root, cause=exc)
        raise HABITAPIError(
            f"Cannot create output directory {root}: {exc}"
        ) from exc

    probe_name = f".habit_write_probe_{os.getpid()}_{uuid.uuid4().hex}.tmp"
    probe_path = root / probe_name
    try:
        with open(probe_path, "wb") as handle:
            handle.write(_PROBE_PAYLOAD)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        if is_filesystem_permission_error(exc):
            raise_unwritable_destination(root, cause=exc)
        raise HABITAPIError(
            f"Cannot write probe file under {root}: {exc}"
        ) from exc
    finally:
        try:
            if probe_path.exists():
                probe_path.unlink()
        except OSError:
            # A leftover probe is harmless; the write itself already succeeded.
            pass

    for raw in existing_paths or ():
        candidate = Path(raw)
        if not candidate.exists() or not candidate.is_file():
            continue
        _probe_overwrite(candidate)

    return root.resolve(strict=False)


def _probe_overwrite(path: Path) -> None:
    """
    Verify an existing file can be overwritten without destroying it.

    Opens the file for update (``r+b``). That fails for read-only attributes
    and common Windows ACL / share locks — the same cases where a later
    ``os.replace`` or SimpleITK write would fail. Does **not** delete the
    caller's artefact: early YAML probes run before long fits and must not
    wipe previous results.

    Args:
        path: Existing file that a save would overwrite.

    Raises:
        HABITAPIError: When the file cannot be opened for write/update.
    """
    try:
        with open(path, "r+b"):
            return
    except OSError as open_exc:
        if is_filesystem_permission_error(open_exc):
            raise_unwritable_destination(path, cause=open_exc)
        raise HABITAPIError(
            f"Cannot open existing output file {path} for overwrite: {open_exc}"
        ) from open_exc


def write_via_temp_then_replace(
    destination: PathLike,
    writer: Callable[[Path], None],
) -> Path:
    """
    Write to a sibling temp file, then atomically replace ``destination``.

    The temporary path keeps the same suffix as ``destination`` so format
    sniffers (e.g. SimpleITK) select the correct encoder. On any failure the
    temp file is removed when possible; permission-like errors become
    :class:`~habit.exceptions.HABITAPIError`.

    Args:
        destination: Final path the caller expects after a successful write.
        writer: Callable that receives the temporary path and must create it
            (e.g. ``lambda p: sitk.WriteImage(image, str(p))``).

    Returns:
        The destination path as a :class:`~pathlib.Path`.

    Raises:
        HABITAPIError: On permission / ACL failures while writing or replacing.
        Exception: Other errors from ``writer`` are re-raised after cleanup.
    """
    dest = Path(destination)
    parent = dest.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if is_filesystem_permission_error(exc):
            raise_unwritable_destination(parent, cause=exc)
        raise

    # Keep the real suffix (including compound ones like ``.nii.gz``) so
    # SimpleITK picks the correct ImageIO writer from the temp path.
    encoder_suffix = _encoder_suffix(dest)
    stem_for_temp = dest.name[: -len(encoder_suffix)] if encoder_suffix else dest.name
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{stem_for_temp}.",
        suffix=encoder_suffix,
        dir=str(parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    # mkstemp creates an empty file; some writers refuse non-empty targets.
    try:
        tmp_path.unlink()
    except OSError:
        pass

    try:
        writer(tmp_path)
        os.replace(str(tmp_path), str(dest))
    except BaseException as exc:
        _cleanup_temp(tmp_path)
        if is_filesystem_permission_error(exc):
            raise_unwritable_destination(dest, cause=exc)
        raise
    return dest


def _encoder_suffix(path: Path) -> str:
    """
    Return the on-disk extension SimpleITK (or similar) should see.

    ``Path.suffix`` is only the final segment (``.gz`` for ``*.nii.gz``), which
    is not enough for ImageIO selection. Compound medical-imaging suffixes are
    recognised explicitly.

    Args:
        path: Destination path whose extension encodes the file format.

    Returns:
        Extension including the leading dot, or ``""`` when none is present.
    """
    name = path.name.lower()
    for compound in (".nii.gz",):
        if name.endswith(compound):
            return compound
    return path.suffix


def _cleanup_temp(tmp_path: Path) -> None:
    """Best-effort removal of a sibling temp file after a failed write."""
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except OSError:
        pass

