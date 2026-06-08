# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
Shared background job runner for Gradio GUI tabs.
Captures stdout/stderr and logging into a live-updating text stream via yield.
"""

import logging
import sys
import threading
import time
from typing import Any, Callable, Generator, List, Tuple


def run_background_job(
    func: Callable[..., Any],
    args: Tuple[Any, ...] = (),
    kwargs: dict | None = None,
    poll_interval_sec: float = 0.5,
    tail_lines: int = 200,
) -> Generator[str, None, None]:
    """
    Run ``func`` in a daemon thread and yield joined log lines while it runs.

    Args:
        func: Callable executed in the background thread.
        args: Positional arguments for ``func``.
        kwargs: Keyword arguments for ``func``.
        poll_interval_sec: Sleep interval between log polls.
        tail_lines: Maximum number of trailing log lines to return per yield.

    Yields:
        str: Concatenated log text (most recent ``tail_lines`` lines).
    """
    kwargs = kwargs or {}
    log_capture: List[str] = []

    class StreamToLog:
        """Redirect stdout/stderr writes into an in-memory log list."""

        def write(self, text: str) -> None:
            clean: str = text.strip()
            if clean:
                log_capture.append(clean)

        def flush(self) -> None:
            pass

    class ListHandler(logging.Handler):
        """Append formatted logging records into the same log list."""

        def emit(self, record: logging.LogRecord) -> None:
            log_capture.append(self.format(record))

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    cap = StreamToLog()
    sys.stdout = cap
    sys.stderr = cap

    root_logger = logging.getLogger()
    handler = ListHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(handler)

    result: dict[str, Any] = {"success": False, "error": None, "done": False}

    def thread_target() -> None:
        try:
            func(*args, **kwargs)
            result["success"] = True
        except Exception as exc:  # noqa: BLE001 — surface pipeline errors to GUI
            result["error"] = str(exc)
        finally:
            result["done"] = True

    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()

    try:
        while not result["done"]:
            time.sleep(poll_interval_sec)
            if log_capture:
                yield "\n".join(log_capture[-tail_lines:])

        if result["success"]:
            log_capture.append("✅ Task completed successfully.")
        else:
            log_capture.append(f"❌ Task failed: {result['error']}")
        yield "\n".join(log_capture[-tail_lines:])
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        root_logger.removeHandler(handler)
