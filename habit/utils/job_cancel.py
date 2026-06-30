# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
Cooperative cancellation for GUI background jobs.

The parent GUI thread sets a flag and writes a small cancel file. Worker
processes spawned during a job inherit ``HABIT_GUI_CANCEL_FILE`` and poll the
file so cancellation propagates across ``multiprocessing`` boundaries.
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterator, Optional, TypeVar

T = TypeVar("T")


class JobCancelledError(Exception):
    """Raised when the user stops a GUI background job."""


_CANCEL_EVENT = threading.Event()
_CANCEL_FILE: Optional[Path] = None


def bind_cancel_file(path: Optional[Path] = None) -> Path:
    """
    Initialize cancel state for a new GUI job.

    Args:
        path: Optional cancel-flag path. When omitted, a temp file is used.

    Returns:
        Path: Cancel flag file used for this job.
    """
    global _CANCEL_FILE

    _CANCEL_EVENT.clear()
    cancel_path = path if path is not None else Path(tempfile.gettempdir()) / "habit_gui_cancel.flag"
    cancel_path.parent.mkdir(parents=True, exist_ok=True)
    cancel_path.write_text("0", encoding="utf-8")
    _CANCEL_FILE = cancel_path
    os.environ["HABIT_GUI_CANCEL_FILE"] = str(cancel_path)
    return cancel_path


def clear_cancel_state() -> None:
    """Reset cancel flag and remove the cancel file after a job ends."""
    global _CANCEL_FILE

    _CANCEL_EVENT.clear()
    if _CANCEL_FILE is not None and _CANCEL_FILE.is_file():
        try:
            _CANCEL_FILE.unlink()
        except OSError:
            pass
    _CANCEL_FILE = None
    os.environ.pop("HABIT_GUI_CANCEL_FILE", None)


def request_job_cancel() -> None:
    """Signal the running GUI job to stop after the current work item finishes."""
    _CANCEL_EVENT.set()
    cancel_file = _CANCEL_FILE
    if cancel_file is None:
        env_path = os.environ.get("HABIT_GUI_CANCEL_FILE")
        if env_path:
            cancel_file = Path(env_path)
    if cancel_file is not None:
        try:
            cancel_file.parent.mkdir(parents=True, exist_ok=True)
            cancel_file.write_text("1", encoding="utf-8")
        except OSError:
            pass


def is_job_cancelled() -> bool:
    """
    Return True when the user has requested job cancellation.

    Safe to call from main process and spawned worker processes.
    """
    if _CANCEL_EVENT.is_set():
        return True
    cancel_file = os.environ.get("HABIT_GUI_CANCEL_FILE")
    if not cancel_file:
        return False
    try:
        return Path(cancel_file).read_text(encoding="utf-8").strip() == "1"
    except OSError:
        return False


def raise_if_job_cancelled() -> None:
    """Raise :class:`JobCancelledError` when cancellation was requested."""
    if is_job_cancelled():
        raise JobCancelledError("Job cancelled by user")


def iter_until_cancelled(iterator: Iterator[T], pool: Any = None) -> Iterator[T]:
    """
    Yield items from ``iterator`` until cancellation is requested.

    When ``pool`` is a ``multiprocessing.Pool``, it is terminated on cancel so
    in-flight worker processes do not keep running.

    Args:
        iterator: Source iterator (e.g. ``pool.imap(...)``).
        pool: Optional multiprocessing pool to terminate on cancel.

    Yields:
        Items from ``iterator`` until cancel or exhaustion.

    Raises:
        JobCancelledError: When the user stops the job mid-iteration.
    """
    for item in iterator:
        if is_job_cancelled():
            if pool is not None:
                try:
                    pool.terminate()
                    pool.join()
                except Exception:  # noqa: BLE001 — best-effort pool shutdown
                    pass
            raise JobCancelledError("Job cancelled by user")
        yield item
