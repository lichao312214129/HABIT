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
    Merge on-disk pipeline logs with live stdout/stderr capture.

    When a log file is configured, treat it as the authoritative detailed trace
    and append any extra live terminal output (click/tqdm/progress) below it.
    """
    stream_text = _normalize_stream_text(stream_buffer.getvalue())
    if log_file is None:
        return stream_text

    file_text = _read_log_file(log_file).strip()
    if not file_text:
        return stream_text
    if not stream_text.strip():
        return file_text

    return (
        f"{file_text}\n\n"
        f"=== Live terminal capture ===\n"
        f"{stream_text.rstrip()}"
    )


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
        tail_lines: Optional maximum trailing lines per yield. ``None`` returns
            the full captured output (default).
        log_file: Optional path to ``processing.log`` (or similar) written by
            the pipeline; its contents are merged into the console display.

    Yields:
        str: Concatenated log text for the Gradio Console log widget.
    """
    kwargs = kwargs or {}
    resolved_log_file: Optional[Path] = Path(log_file) if log_file else None

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

    root_logger = logging.getLogger()
    root_handler = ListHandler(stream_buffer)
    root_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(root_handler)

    result: dict[str, Any] = {
        "success": False,
        "error": None,
        "traceback": None,
        "done": False,
    }

    def thread_target() -> None:
        try:
            func(*args, **kwargs)
            result["success"] = True
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

    def _yield_text(full_text: str) -> str:
        safe_text = _ensure_text(full_text)
        if tail_lines is None or tail_lines <= 0:
            return safe_text
        lines = safe_text.splitlines()
        if len(lines) <= tail_lines:
            return safe_text
        return "\n".join(lines[-tail_lines:])

    try:
        while not result["done"]:
            time.sleep(poll_interval_sec)
            display_text = _build_display_text(stream_buffer, resolved_log_file)
            if display_text:
                yield _yield_text(display_text)

        if result["success"]:
            stream_buffer.write("\n✅ Task completed successfully.\n")
        else:
            error_text = _ensure_text(result["error"] or "Unknown error")
            stream_buffer.write(f"\n❌ Task failed: {error_text}\n")
            tb_text = result.get("traceback")
            if tb_text:
                stream_buffer.write(_ensure_text(tb_text))

        final_text = _build_display_text(stream_buffer, resolved_log_file)
        yield _yield_text(final_text)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        os.environ.pop("HABIT_GUI_JOB", None)
        root_logger.removeHandler(root_handler)
