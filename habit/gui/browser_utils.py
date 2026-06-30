# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
Browser helpers for launching URLs from CLI / GUI entry points.

WSL does not provide a working default browser for Python's webbrowser module,
so we delegate to the Windows host when running under WSL.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
import webbrowser

logger = logging.getLogger(__name__)

_WSL_CMD_EXE = "/mnt/c/Windows/System32/cmd.exe"


def is_wsl() -> bool:
    """
    Return True when the current process runs inside Windows Subsystem for Linux.

    Returns:
        bool: True if WSL environment is detected.
    """
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        with open("/proc/version", encoding="utf-8", errors="ignore") as handle:
            return "microsoft" in handle.read().lower()
    except OSError:
        return False


def _open_url_via_windows_host(url: str) -> bool:
    """
    Open a URL with the Windows default browser from WSL.

    Args:
        url: Fully qualified HTTP(S) URL.

    Returns:
        bool: True when a Windows launcher command was invoked successfully.
    """
    if shutil.which("wslview"):
        result = subprocess.run(
            ["wslview", url],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return True

    cmd_exe = _WSL_CMD_EXE if os.path.isfile(_WSL_CMD_EXE) else "cmd.exe"
    result = subprocess.run(
        [cmd_exe, "/c", "start", "", url],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def open_system_browser(url: str) -> bool:
    """
    Open the given URL in the platform-appropriate default browser.

    Args:
        url: Fully qualified HTTP(S) URL.

    Returns:
        bool: True when a browser launch was attempted without subprocess errors.
    """
    if is_wsl():
        opened = _open_url_via_windows_host(url)
        if not opened:
            logger.warning("Failed to open browser via Windows host from WSL: %s", url)
        return opened

    try:
        return bool(webbrowser.open(url, new=2))
    except Exception as exc:
        logger.warning("Failed to open browser: %s (%s)", url, exc)
        return False


def schedule_browser_open(url: str, delay_seconds: float = 2.0) -> None:
    """
    Open a browser in a background thread after a short delay.

    Gradio binds its HTTP port asynchronously; waiting briefly avoids opening
    the browser before the local server accepts connections.

    Args:
        url: Fully qualified HTTP(S) URL.
        delay_seconds: Seconds to wait before opening the browser.
    """

    def _worker() -> None:
        time.sleep(delay_seconds)
        open_system_browser(url)

    threading.Thread(target=_worker, daemon=True).start()


def should_use_host_browser() -> bool:
    """
    Return True when Python's webbrowser module is unlikely to work locally.

    Returns:
        bool: True for WSL environments.
    """
    return is_wsl()
