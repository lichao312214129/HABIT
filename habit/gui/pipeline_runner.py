# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
Shared background job runner for Gradio GUI tabs.
Captures stdout/stderr and logging into a live-updating text stream via yield.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Generator, Optional, Tuple, Union

from habit.utils.job_cancel import (
    bind_cancel_file,
    clear_cancel_state,
    is_job_cancelled,
    JobCancelledError,
)

# Serialize GUI pipeline jobs. ``run_background_job`` replaces the global
# ``sys.stdout``/``sys.stderr`` while a job runs; allowing two jobs to overlap
# would let one job restore the other's captured stream and scramble both logs.
# Only one pipeline runs at a time across all tabs.
_JOB_LOCK = threading.Lock()

# Upper bound on the number of trailing lines pushed to the Gradio log widget on
# each update, to keep the streamed payload small for long-running jobs. The full
# trace is always available in the on-disk ``processing.log``.
_MAX_DISPLAY_LINES: int = 4000


# Minimum seconds between streamed log yields while a job is running. Progress
# bars (tqdm) rewrite the same terminal line many times per second; throttling
# UI updates keeps the page and log scroll position stable for the user.
_MIN_YIELD_INTERVAL_SEC: float = 1.0


def _should_yield_log_update(
    new_text: str,
    last_text: Optional[str],
    last_yield_monotonic: float,
    *,
    force: bool = False,
) -> bool:
    """
    Decide whether to push another log snapshot to the Gradio Textbox.

    Always yield on the first snapshot and on the final forced flush. During
    streaming, skip updates that only rewrite the trailing progress line unless
    enough time has passed or multiple log lines changed at once.

    Args:
        new_text: Latest normalized log text.
        last_text: Previously yielded text, or None on the first poll.
        last_yield_monotonic: ``time.monotonic()`` timestamp of the last yield.
        force: When True, always yield (used for the final success/error block).

    Returns:
        bool: True when the GUI should receive a new log value.
    """
    if force:
        return True
    if last_text is None:
        return True
    if new_text == last_text:
        return False
    if time.monotonic() - last_yield_monotonic >= _MIN_YIELD_INTERVAL_SEC:
        return True

    old_lines = last_text.splitlines()
    new_lines = new_text.splitlines()
    if len(new_lines) > len(old_lines):
        return True
    if len(new_lines) == len(old_lines):
        changed_rows = sum(1 for old_row, new_row in zip(old_lines, new_lines) if old_row != new_row)
        if changed_rows > 1:
            return True
    return False


def _ensure_text(value: Any) -> str:
    """Coerce arbitrary captured chunks to ``str`` for safe display and joins."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        return value.decode(encoding, errors="replace")
    return str(value)


class _GuiLogBuffer:
    """Thread-safe accumulator for raw stdout/stderr text."""

    def __init__(self, encoding: str) -> None:
        self._encoding: str = encoding
        self._lock = threading.Lock()
        self._buffer = io.StringIO()

    def write(self, chunk: Any) -> None:
        """Append a stdout/stderr chunk without stripping or splitting lines."""
        text = _ensure_text(chunk)
        if not text:
            return
        with self._lock:
            self._buffer.write(text)

    def getvalue(self) -> str:
        """Return the full captured stream text."""
        with self._lock:
            return self._buffer.getvalue()


def _normalize_stream_text(text: str) -> str:
    """
    Normalize carriage-return progress updates for Gradio Textbox display.

    tqdm and similar tools rewrite the same terminal line with ``\\r``; the GUI
    cannot render in-place updates, so keep the latest segment per line.
    """
    if not text:
        return ""
    normalized_lines: list[str] = []
    for line in text.splitlines():
        if "\r" in line:
            line = line.split("\r")[-1]
        normalized_lines.append(line)
    if text.endswith("\n") or text.endswith("\r"):
        return "\n".join(normalized_lines) + "\n"
    return "\n".join(normalized_lines)


def _read_log_file(log_file: Path) -> str:
    """Read a log file best-effort; return empty string when unavailable."""
    try:
        if not log_file.is_file():
            return ""
        return log_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _build_display_text(
    stream_buffer: _GuiLogBuffer,
    log_file: Optional[Path],
) -> str:
    """
    Build the console text for the GUI from a single authoritative source.

    HABIT routes pipeline logs through the ``habit`` logger whose console handler
    writes to ``sys.stdout`` (captured here). The same records are also written to
    the on-disk ``processing.log``. Merging both sources previously showed every
    log line two or three times, so the live stdout/stderr capture is treated as
    the single source of truth. The on-disk log is only used as a fallback when the
    live capture is empty (e.g. a pipeline that logs to file but not to console).

    Args:
        stream_buffer: Captured stdout/stderr (and stray root-logger records).
        log_file: Optional pipeline log path used only as an empty-stream fallback.

    Returns:
        str: De-duplicated console text for the Gradio log widget.
    """
    stream_text = _normalize_stream_text(stream_buffer.getvalue())
    if stream_text.strip():
        return stream_text
    if log_file is None:
        return stream_text
    return _read_log_file(log_file).strip()


def run_background_job(
    func: Callable[..., Any],
    args: Tuple[Any, ...] = (),
    kwargs: dict | None = None,
    poll_interval_sec: float = 0.3,
    tail_lines: Optional[int] = None,
    log_file: Optional[Union[str, Path]] = None,
) -> Generator[str, None, None]:
    """
    Run ``func`` in a daemon thread and yield joined log text while it runs.

    Args:
        func: Callable executed in the background thread.
        args: Positional arguments for ``func``.
        kwargs: Keyword arguments for ``func``.
        poll_interval_sec: Sleep interval between log polls.
        tail_lines: Optional maximum trailing lines per yield. ``None`` falls back
            to ``_MAX_DISPLAY_LINES`` to keep the streamed payload bounded.
        log_file: Optional path to ``processing.log`` (or similar). Used only as a
            fallback when the live stdout/stderr capture is empty.

    Yields:
        str: Console log text for the Gradio log widget (only when it changes).
    """
    kwargs = kwargs or {}
    resolved_log_file: Optional[Path] = Path(log_file) if log_file else None
    cancel_file: Optional[Path] = None
    if resolved_log_file is not None:
        cancel_file = resolved_log_file.parent / ".habit_gui_cancel"

    # Only one pipeline job may run at a time (global stdout redirection is not
    # safe to overlap). Reject concurrent runs instead of scrambling both logs.
    if not _JOB_LOCK.acquire(blocking=False):
        yield (
            "⏳ Another HABIT task is already running. "
            "Please wait for it to finish before starting a new one."
        )
        return

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stream_encoding: str = getattr(old_stdout, "encoding", None) or "utf-8"
    stream_buffer = _GuiLogBuffer(encoding=stream_encoding)

    class StreamToLog:
        """Redirect stdout/stderr writes into the shared GUI log buffer."""

        encoding: str = stream_encoding

        def __init__(self, buffer: _GuiLogBuffer) -> None:
            self._buffer = buffer

        def write(self, text: Any) -> None:
            self._buffer.write(text)

        def flush(self) -> None:
            pass

        def isatty(self) -> bool:
            # Force tqdm and similar tools to emit plain newlines for the GUI.
            return False

    class ListHandler(logging.Handler):
        """Capture logging records that never reach stdout."""

        def __init__(self, buffer: _GuiLogBuffer) -> None:
            super().__init__()
            self._buffer = buffer

        def emit(self, record: logging.LogRecord) -> None:
            try:
                message = self.format(record)
            except Exception:  # noqa: BLE001 — logging must not break the GUI job
                return
            self._buffer.write(message + "\n")

    cap = StreamToLog(stream_buffer)
    sys.stdout = cap  # type: ignore[assignment]
    sys.stderr = cap  # type: ignore[assignment]
    os.environ["HABIT_GUI_JOB"] = "1"
    bind_cancel_file(cancel_file)

    root_logger = logging.getLogger()
    root_handler = ListHandler(stream_buffer)
    root_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(root_handler)

    result: dict[str, Any] = {
        "success": False,
        "cancelled": False,
        "error": None,
        "traceback": None,
        "done": False,
    }

    def thread_target() -> None:
        try:
            func(*args, **kwargs)
            result["success"] = True
        except JobCancelledError as exc:
            result["cancelled"] = True
            result["error"] = _ensure_text(exc)
        except BaseException as exc:  # noqa: BLE001 — include SystemExit from CLI helpers
            if isinstance(exc, SystemExit):
                code = exc.code if exc.code is not None else 1
                if code == 0:
                    result["success"] = True
                    return
                result["error"] = f"Process exited with code {code}"
            else:
                result["error"] = _ensure_text(exc)
                result["traceback"] = traceback.format_exc()
        finally:
            result["done"] = True

    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()

    cap_lines: int = tail_lines if (tail_lines and tail_lines > 0) else _MAX_DISPLAY_LINES

    def _yield_text(full_text: str) -> str:
        safe_text = _ensure_text(full_text)
        lines = safe_text.splitlines()
        if len(lines) <= cap_lines:
            return safe_text
        return "\n".join(lines[-cap_lines:])

    last_yielded: Optional[str] = None
    last_yield_monotonic: float = 0.0
    try:
        while not result["done"]:
            time.sleep(poll_interval_sec)
            if is_job_cancelled() and not result.get("cancelled"):
                stream_buffer.write(
                    "\n⏹ Stop requested — finishing current step, then stopping...\n"
                )
            display_text = _build_display_text(stream_buffer, resolved_log_file)
            if display_text:
                text = _yield_text(display_text)
                if _should_yield_log_update(
                    text,
                    last_yielded,
                    last_yield_monotonic,
                ):
                    last_yielded = text
                    last_yield_monotonic = time.monotonic()
                    yield text

        if result.get("cancelled"):
            stream_buffer.write("\n⏹ Task cancelled by user.\n")
        elif result["success"]:
            stream_buffer.write("\n✅ Task completed successfully.\n")
        else:
            error_text = _ensure_text(result["error"] or "Unknown error")
            stream_buffer.write(f"\n❌ Task failed: {error_text}\n")
            tb_text = result.get("traceback")
            if tb_text:
                stream_buffer.write(_ensure_text(tb_text))

        final_text = _yield_text(_build_display_text(stream_buffer, resolved_log_file))
        if _should_yield_log_update(
            final_text,
            last_yielded,
            last_yield_monotonic,
            force=True,
        ):
            yield final_text
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        os.environ.pop("HABIT_GUI_JOB", None)
        clear_cancel_state()
        root_logger.removeHandler(root_handler)
        _JOB_LOCK.release()
