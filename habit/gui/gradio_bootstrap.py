# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
Gradio bootstrap helpers for Windows hosts where asyncio event-loop creation deadlocks.

Gradio 4+ eagerly builds its queue inside ``Blocks.__init__`` via ``safe_get_lock()``,
which calls ``asyncio.new_event_loop()``. On some Windows setups the loop bootstrap
hangs forever in TCP ``socket.accept()`` while creating the internal self-pipe, so the
GUI never reaches ``launch()`` and no browser page opens.

We patch Gradio's lock helpers to use thread-backed async primitives that do not need
a freshly created event loop during import/Blocks construction, replace broken TCP
``127.0.0.1`` socketpair fallbacks on Windows, and extend the uvicorn startup grace period.
"""

from __future__ import annotations

import errno
import socket
import sys
import threading
import time
from typing import Any, Callable

_SOCKETPAIR_PATCHED = False
_ORIGINAL_SOCKETPAIR: Callable[..., tuple[socket.socket, socket.socket]] | None = None


def _local_tcp_bind_candidates() -> list[str]:
    """
    Build ordered bind-host candidates for emulating ``socket.socketpair`` on Windows.

    Some hosts block TCP to ``127.0.0.1`` (for example Clash TUN / fake-ip setups) while
    other local interface addresses still accept loopback-style connections.

    Returns:
        list[str]: Host strings to try, most portable first.
    """
    candidates: list[str] = ["127.0.0.1"]
    try:
        hostname_addrs = socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        hostname_addrs = []
    for addr in hostname_addrs:
        if addr.startswith("127.") or addr in candidates:
            continue
        candidates.append(addr)
    return candidates


def _tcp_socketpair_on_host(
    host: str,
    family: int = socket.AF_INET,
    sock_type: int = socket.SOCK_STREAM,
    proto: int = 0,
    timeout_sec: float = 3.0,
) -> tuple[socket.socket, socket.socket]:
    """
    Create a connected TCP socket pair bound to a specific local host.

    Args:
        host: Local IPv4 address used for bind/connect.
        family: Socket address family (``AF_INET`` or ``AF_INET6``).
        sock_type: Socket type; only ``SOCK_STREAM`` is supported.
        proto: Socket protocol; only zero is supported.
        timeout_sec: Per-attempt connect/accept timeout in seconds.

    Returns:
        tuple[socket.socket, socket.socket]: Connected stream socket pair.

    Raises:
        OSError: When bind, connect, or accept fails on this host.
        TimeoutError: When accept does not complete in time.
        ValueError: When unsupported family/type/proto is requested.
    """
    if family not in (socket.AF_INET, socket.AF_INET6):
        raise ValueError("Only AF_INET and AF_INET6 socket address families are supported")
    if sock_type != socket.SOCK_STREAM:
        raise ValueError("Only SOCK_STREAM socket type is supported")
    if proto != 0:
        raise ValueError("Only protocol zero is supported")

    lsock = socket.socket(family, sock_type, proto)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        lsock.bind((host, 0))
        lsock.listen(1)
        addr = lsock.getsockname()
        accepted: list[socket.socket] = []
        accept_error: list[BaseException] = []

        def _accept_worker() -> None:
            try:
                lsock.settimeout(timeout_sec)
                server_sock, _peer = lsock.accept()
                accepted.append(server_sock)
            except BaseException as exc:
                accept_error.append(exc)

        accept_thread = threading.Thread(target=_accept_worker, daemon=True)
        accept_thread.start()
        time.sleep(0.05)

        csock = socket.socket(family, sock_type, proto)
        try:
            csock.settimeout(timeout_sec)
            csock.connect(addr)
        except BaseException:
            csock.close()
            raise

        accept_thread.join(timeout=timeout_sec + 0.5)
        if accept_error:
            csock.close()
            raise accept_error[0]
        if not accepted:
            csock.close()
            raise TimeoutError(f"socketpair accept timed out on host {host!r}")
        return accepted[0], csock
    finally:
        lsock.close()


def _windows_tcp_socketpair(
    family: int = socket.AF_INET,
    sock_type: int = socket.SOCK_STREAM,
    proto: int = 0,
) -> tuple[socket.socket, socket.socket]:
    """
    Windows ``socketpair`` replacement resilient to broken ``127.0.0.1`` TCP loopback.

    Args:
        family: Requested socket address family.
        sock_type: Requested socket type.
        proto: Requested socket protocol.

    Returns:
        tuple[socket.socket, socket.socket]: Connected socket pair.

    Raises:
        OSError: When no local bind host can establish a connected pair.
    """
    if family == getattr(socket, "AF_UNIX", None) and _ORIGINAL_SOCKETPAIR is not None:
        try:
            return _ORIGINAL_SOCKETPAIR(family, sock_type, proto)
        except OSError:
            pass

    if family == socket.AF_INET6:
        host_candidates = ["::1"]
    elif family in (socket.AF_INET, getattr(socket, "AF_UNIX", socket.AF_INET)):
        host_candidates = _local_tcp_bind_candidates()
    else:
        raise ValueError("Only AF_INET and AF_INET6 socket address families are supported")

    errors: list[str] = []
    for host in host_candidates:
        per_host_timeout = 1.5 if host == "127.0.0.1" else 3.0
        try:
            return _tcp_socketpair_on_host(
                host=host,
                family=socket.AF_INET if family != socket.AF_INET6 else socket.AF_INET6,
                sock_type=sock_type,
                proto=proto,
                timeout_sec=per_host_timeout,
            )
        except OSError as exc:
            errors.append(f"{host}: {exc}")
        except TimeoutError as exc:
            errors.append(f"{host}: {exc}")

    detail = "; ".join(errors[-4:]) if errors else "no host candidates"
    raise OSError(errno.ECONNREFUSED, f"Unable to create a Windows socket pair ({detail})")


def _patched_socketpair(
    family: int | None = None,
    sock_type: int = socket.SOCK_STREAM,
    proto: int = 0,
) -> tuple[socket.socket, socket.socket]:
    """
    Drop-in replacement for ``socket.socketpair`` used on Windows.

    Args:
        family: Socket family; ``None`` maps to ``AF_INET`` when AF_UNIX is unavailable.
        sock_type: Socket type.
        proto: Socket protocol.

    Returns:
        tuple[socket.socket, socket.socket]: Connected socket pair.
    """
    if family is None:
        family = socket.AF_INET
    return _windows_tcp_socketpair(family=family, sock_type=sock_type, proto=proto)


_patched_socketpair.__name__ = "_patched_socketpair"


def _patch_windows_socketpair() -> None:
    """
    Patch CPython's TCP ``socketpair`` fallback before asyncio creates event loops.

    Safe to call multiple times.
    """
    global _SOCKETPAIR_PATCHED, _ORIGINAL_SOCKETPAIR
    if _SOCKETPAIR_PATCHED or sys.platform != "win32":
        return

    import socket as socket_module

    _ORIGINAL_SOCKETPAIR = getattr(socket_module, "socketpair", None)
    socket_module._fallback_socketpair = _windows_tcp_socketpair  # type: ignore[attr-defined]
    socket_module.socketpair = _patched_socketpair  # type: ignore[assignment]
    _SOCKETPAIR_PATCHED = True


def _patch_uvicorn_event_loop() -> None:
    """
    Force uvicorn to use ``SelectorEventLoop`` on Windows.

    Uvicorn defaults to ``ProactorEventLoop`` on Windows, which still depends on
    ``socket.socketpair`` but ignores ``WindowsSelectorEventLoopPolicy``.
    """
    import uvicorn.loops.asyncio as uvicorn_asyncio

    def _selector_loop_factory(
        use_subprocess: bool = False,
    ) -> type[asyncio.SelectorEventLoop]:
        del use_subprocess
        return asyncio.SelectorEventLoop

    uvicorn_asyncio.asyncio_loop_factory = _selector_loop_factory  # type: ignore[assignment]


_patch_windows_socketpair()

import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    _patch_uvicorn_event_loop()


class _ThreadAsyncLock:
    """Minimal asyncio-compatible lock backed by ``threading.Lock``."""

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()

    async def __aenter__(self) -> "_ThreadAsyncLock":
        await asyncio.sleep(0)
        self._lock.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._lock.release()

    async def acquire(self) -> None:
        await asyncio.sleep(0)
        self._lock.acquire()

    def locked(self) -> bool:
        return self._lock.locked()

    def release(self) -> None:
        self._lock.release()


class _ThreadAsyncEvent:
    """Minimal asyncio-compatible event backed by ``threading.Event``."""

    def __init__(self) -> None:
        self._event: threading.Event = threading.Event()

    async def wait(self) -> bool:
        while not self._event.is_set():
            await asyncio.sleep(0.01)
        return True

    def is_set(self) -> bool:
        return self._event.is_set()

    def set(self) -> None:
        self._event.set()

    def clear(self) -> None:
        self._event.clear()


def _safe_get_lock() -> _ThreadAsyncLock:
    return _ThreadAsyncLock()


def _safe_get_stop_event() -> _ThreadAsyncEvent:
    return _ThreadAsyncEvent()


def _patch_gradio_server_startup(timeout_sec: float = 120.0) -> None:
    """
    Extend Gradio's uvicorn startup wait on Windows.

    The default 5-second timeout surfaces as a misleading "Cannot find empty port"
    error when asyncio/uvicorn is still warming up on slower hosts.
    """
    import gradio.http_server as gradio_http_server
    from gradio.exceptions import ServerFailedToStartError

    original_run_in_thread: Callable[..., None] = gradio_http_server.Server.run_in_thread

    def _run_in_thread_with_longer_timeout(self: gradio_http_server.Server) -> None:
        self.thread = threading.Thread(target=self.run, daemon=True)
        if self.reloader:
            self.watch_thread = threading.Thread(target=self.watch, daemon=True)
            self.watch_thread.start()
        self.thread.start()
        start = time.time()
        while not self.started:
            time.sleep(1e-3)
            if time.time() - start > timeout_sec:
                raise ServerFailedToStartError(
                    "Server failed to start. Please check that the port is available."
                )

    gradio_http_server.Server.run_in_thread = _run_in_thread_with_longer_timeout  # type: ignore[method-assign]


def _prefer_env_gradio_over_user_site() -> None:
    """
    Prefer the conda/env Gradio install when a stray user-site copy also exists.

    A partial ``pip install --user gradio`` can be picked up first and fail at import
    time (for example missing ``aiofiles``).
    """
    import site
    from pathlib import Path

    user_gradio = Path(site.getusersitepackages()) / "gradio"
    if not user_gradio.is_dir():
        return

    for entry in sys.path:
        if "site-packages" in entry and "Roaming" not in entry:
            if sys.path[0] != entry:
                sys.path.insert(0, entry)
            return


def apply_gradio_windows_patches() -> None:
    """
    Patch Gradio helpers before ``Blocks()`` is constructed.

    ``gradio.queueing`` binds ``safe_get_lock`` at import time, so both ``gradio.utils``
    and ``gradio.queueing`` must be patched. Safe to call multiple times.
    """
    _prefer_env_gradio_over_user_site()

    if sys.platform != "win32":
        return

    import gradio.queueing as gradio_queueing
    import gradio.utils as gradio_utils

    gradio_utils.safe_get_lock = _safe_get_lock  # type: ignore[assignment]
    gradio_utils.safe_get_stop_event = _safe_get_stop_event  # type: ignore[assignment]
    gradio_queueing.safe_get_lock = _safe_get_lock  # type: ignore[assignment]
    gradio_queueing.safe_get_stop_event = _safe_get_stop_event  # type: ignore[assignment]
    _patch_gradio_server_startup()
