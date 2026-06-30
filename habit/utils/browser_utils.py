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
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_WSL_CMD_EXE = "/mnt/c/Windows/System32/cmd.exe"
_LOCALHOST_BIND_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


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


def schedule_browser_open(
    url: str,
    delay_seconds: float = 2.0,
    server_port: Optional[int] = None,
) -> None:
    """
    Open a browser in a background thread after a short delay.

    Gradio binds its HTTP port asynchronously; waiting briefly avoids opening
    the browser before the local server accepts connections.

    Args:
        url: Fully qualified HTTP(S) URL.
        delay_seconds: Minimum seconds to wait before opening the browser.
        server_port: When set, poll until this local port accepts connections
            (up to ``max(delay_seconds, 30)`` seconds) before opening.
    """

    def _worker() -> None:
        if server_port is not None:
            deadline = time.time() + max(delay_seconds, 30.0)
            while time.time() < deadline:
                if is_port_listening_locally(server_port):
                    break
                time.sleep(0.3)
        else:
            time.sleep(delay_seconds)
        open_system_browser(url)

    threading.Thread(target=_worker, daemon=True).start()


def ensure_localhost_no_proxy(env: Optional[dict[str, str]] = None) -> None:
    """
    Ensure loopback hosts bypass HTTP/S proxy settings.

    Corporate proxies and tools like Clash can intercept localhost traffic and
    break Gradio's self-checks unless these hosts appear in NO_PROXY.

    Args:
        env: Optional environment mapping to update in place. When omitted,
            updates ``os.environ``.
    """
    target: dict[str, str] = os.environ if env is None else env
    for key in ("NO_PROXY", "no_proxy"):
        existing = target.get(key, "")
        parts = [part.strip() for part in existing.split(",") if part.strip()]
        for entry in _LOCALHOST_BIND_HOSTS:
            if entry not in parts:
                parts.append(entry)
        target[key] = ",".join(parts)


def should_use_host_browser() -> bool:
    """
    Return True when Python's webbrowser module is unlikely to work locally.

    Returns:
        bool: True for WSL environments.
    """
    return is_wsl()


def get_wsl_primary_ip() -> Optional[str]:
    """
    Return the primary IPv4 address assigned to the WSL network interface.

    Windows browsers cannot reliably reach WSL services bound to 127.0.0.1 on
    Win10/Win11 NAT networking, so callers expose the GUI via this address.

    Returns:
        Optional[str]: First IPv4 from ``hostname -I``, or None if unavailable.
    """
    try:
        result = subprocess.run(
            ["hostname", "-I"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.debug("Unable to resolve WSL IP via hostname -I: %s", exc)
        return None

    for token in result.stdout.strip().split():
        if "." in token and ":" not in token:
            return token
    return None


def resolve_gui_bind_host(requested_host: str) -> str:
    """
    Choose the Gradio bind address for the current platform.

    Under WSL, localhost forwarding from Windows is often broken. Binding to
    all interfaces lets the Windows host reach the server via the WSL IP.

    Args:
        requested_host: Host passed from CLI or environment variables.

    Returns:
        str: Host string suitable for ``demo.launch(server_name=...)``.
    """
    normalized_host = requested_host.strip().lower()
    if is_wsl() and normalized_host in _LOCALHOST_BIND_HOSTS:
        return "0.0.0.0"
    return requested_host


def resolve_gui_browser_url(bind_host: str, port: int) -> str:
    """
    Build the URL that the user should open in a desktop browser.

    Args:
        bind_host: Gradio bind host after ``resolve_gui_bind_host``.
        port: TCP port for the Gradio server.

    Returns:
        str: Fully qualified HTTP URL for browser access.
    """
    if is_wsl():
        wsl_ip = get_wsl_primary_ip()
        if wsl_ip:
            return f"http://{wsl_ip}:{port}"

    normalized_host = bind_host.strip().lower()
    if normalized_host in {"0.0.0.0", "::"}:
        return f"http://127.0.0.1:{port}"
    return f"http://{bind_host}:{port}"


def get_gradio_bind_host(requested_host: str) -> str:
    """
    Choose the Gradio bind address for CLI / GUI entry points.

    Alias for :func:`resolve_gui_bind_host`.

    Args:
        requested_host: Host passed from CLI or environment variables.

    Returns:
        str: Host string suitable for ``demo.launch(server_name=...)``.
    """
    return resolve_gui_bind_host(requested_host)


def get_gradio_browser_url(requested_host: str, port: int) -> str:
    """
    Build the browser URL from the user-requested bind host and port.

    Args:
        requested_host: Host passed from CLI or environment variables.
        port: TCP port for the Gradio server.

    Returns:
        str: Fully qualified HTTP URL for browser access.
    """
    bind_host = resolve_gui_bind_host(requested_host)
    return resolve_gui_browser_url(bind_host, port)


def get_wsl_browser_access_hint(port: int) -> str:
    """
    Return a short message explaining how to reach the GUI from Windows under WSL.

    Args:
        port: TCP port the Gradio server listens on.

    Returns:
        str: Human-readable hint for CLI / log output.
    """
    wsl_ip = get_wsl_primary_ip()
    if wsl_ip:
        return (
            f"Open http://{wsl_ip}:{port} in your Windows browser "
            "(127.0.0.1 may not forward correctly)."
        )
    return (
        "Use the URL above from your Windows browser "
        "(127.0.0.1 may not forward correctly)."
    )


def _port_bind_host(requested_host: str) -> str:
    """
    Map a Gradio bind host to the address used for local port availability checks.

    Args:
        requested_host: Host passed from CLI or environment variables.

    Returns:
        str: IPv4 loopback address suitable for ``socket.bind``.
    """
    normalized_host = requested_host.strip().lower()
    if normalized_host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return requested_host


def is_gui_port_available(requested_host: str, port: int) -> bool:
    """
    Return True when the current process can bind the requested GUI port.

    Args:
        requested_host: Host passed from CLI or environment variables.
        port: Candidate TCP port.

    Returns:
        bool: True if ``socket.bind`` succeeds for the host/port pair.
    """
    if is_port_listening_locally(port):
        return False

    bind_host = _port_bind_host(requested_host)
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((bind_host, port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def is_port_listening_locally(port: int) -> bool:
    """
    Return True when any process is already listening on the local TCP port.

    On Windows, ``SO_REUSEADDR`` allows multiple binders on the same loopback
    port (for example ``wslrelay.exe`` plus Gradio), so a bind-only probe is
    not enough to detect conflicts.

    Args:
        port: Local TCP port number.

    Returns:
        bool: True when ``netstat`` reports a LISTENING socket on the port.
    """
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return False

        markers = (f":{port} ", f":{port}\t")
        for line in result.stdout.splitlines():
            if "LISTENING" not in line:
                continue
            if any(marker in line for marker in markers):
                return True
        return False

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.5)
    try:
        probe.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def describe_windows_port_owner(port: int) -> Optional[str]:
    """
    Best-effort lookup of which process is listening on a TCP port (Windows only).

    Args:
        port: Local TCP port number.

    Returns:
        Optional[str]: Human-readable ``process (pid)`` string, or None if unknown.
    """
    if sys.platform != "win32":
        return None

    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    needle = f":{port} "
    listening_pid: Optional[str] = None
    for line in result.stdout.splitlines():
        if "LISTENING" not in line or needle not in line:
            continue
        parts = line.split()
        if parts:
            listening_pid = parts[-1]
            break

    if listening_pid is None or not listening_pid.isdigit():
        return None

    try:
        task = subprocess.run(
            ["tasklist", "/FI", f"PID eq {listening_pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return f"pid {listening_pid}"

    first_line = task.stdout.strip().splitlines()[0] if task.stdout.strip() else ""
    if first_line:
        process_name = first_line.split(",")[0].strip('"')
        return f"{process_name} (pid {listening_pid})"
    return f"pid {listening_pid}"


def resolve_gui_server_port(
    requested_host: str,
    preferred_port: int,
    try_count: int = 30,
) -> Tuple[int, Optional[int], Optional[str]]:
    """
    Choose a GUI port, falling back when the preferred port is already taken.

    On Windows with WSL installed, ``8501`` is often held by ``wslrelay.exe`` even
    though the browser still opens that URL and times out.

    Args:
        requested_host: Host passed from CLI or environment variables.
        preferred_port: First port to try.
        try_count: Number of consecutive ports to scan.

    Returns:
        Tuple[int, Optional[int], Optional[str]]:
            - Port that the GUI should bind.
            - Previously requested port when a fallback was chosen, else None.
            - Short description of the process occupying the preferred port, if any.
    """
    blocked_by: Optional[str] = None
    if not is_gui_port_available(requested_host, preferred_port):
        blocked_by = describe_windows_port_owner(preferred_port)

    for offset in range(try_count):
        candidate = preferred_port + offset
        if is_gui_port_available(requested_host, candidate):
            replaced = preferred_port if candidate != preferred_port else None
            return candidate, replaced, blocked_by

    raise OSError(
        f"No free GUI port found in range {preferred_port}-{preferred_port + try_count - 1}."
    )
