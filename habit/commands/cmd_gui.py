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
"""
Launch the HABIT web GUI.

Starts the next-generation React GUI (``habit-gui/``) on a single local port.
The GUI is still under active development and may not be ready for production use.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import click

from habit.exceptions import OptionalDependencyError
from habit.utils.browser_utils import (
    ensure_localhost_no_proxy,
    get_wsl_browser_access_hint,
    is_wsl,
    schedule_browser_open,
)
from habit.utils.gui_paths import (
    find_habit_gui_root,
    gui_api_dir,
    gui_repo_root,
    gui_static_dir,
)


def _require_gui_dependencies() -> None:
    """
    Ensure FastAPI / uvicorn are available for the next-generation GUI.

    Raises:
        OptionalDependencyError: When required packages are missing.
            Subclasses ``ImportError``, consistent with every other
            optional-dependency path in HABIT.
    """
    missing: list[str] = []
    for package in ("fastapi", "uvicorn"):
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        raise OptionalDependencyError(
            "Missing GUI dependencies: "
            + ", ".join(missing)
            + '. Install with: pip install "habitat-analysis[gui]"'
        )


def _resolve_habit_cli_executable() -> str:
    """
    Resolve the HABIT CLI executable in the current environment.

    Returns:
        str: Path to ``habit`` on PATH, or ``sys.executable`` for ``python -m habit``.
    """
    script = shutil.which("habit")
    if script:
        return script
    return sys.executable


def _habit_cli_uses_module() -> bool:
    """Return True when jobs should run as ``python -m habit``."""
    return shutil.which("habit") is None


def run_next_gui_server(
    host: str = "127.0.0.1",
    port: int = 8501,
    *,
    open_browser: bool = True,
) -> None:
    """
    Launch the next-generation HABIT GUI (FastAPI + static React build).

    Uses the **current Python interpreter** for the API and bridge subprocesses,
    so registry/schema export share the same environment as ``habit preprocess``.

    Args:
        host: Bind host for the combined API + web server.
        port: Local port (default 8501, same as legacy Gradio GUI).
        open_browser: When True, open the default browser once the port is ready.
    """
    _require_gui_dependencies()

    try:
        gui_root = find_habit_gui_root()
    except FileNotFoundError as exc:
        click.secho(str(exc), fg="red", err=True)
        raise SystemExit(1) from exc

    static_dir = gui_static_dir(gui_root)
    if not static_dir.is_dir() or not (static_dir / "index.html").is_file():
        click.secho(
            f"Web UI build not found: {static_dir}\n"
            "Build it once from the repository:\n"
            "  cd habit-gui/web && npm install && npm run build",
            fg="red",
            err=True,
        )
        raise SystemExit(1)

    api_dir = gui_api_dir(gui_root)
    repo_root = gui_repo_root(gui_root)
    click.secho(f"Static UI: {static_dir}", fg="white")
    click.secho(f"API package: {api_dir}", fg="white")
    browser_url = f"http://{host}:{port}/"

    click.secho("===================================================", fg="cyan")
    click.secho("   HABIT — Next Generation Web GUI                ", fg="cyan", bold=True)
    click.secho("===================================================", fg="cyan")
    click.secho(f"Open in browser: {browser_url}", fg="green")
    click.secho(f"Python: {sys.executable}", fg="white")
    if is_wsl():
        click.secho(get_wsl_browser_access_hint(port), fg="yellow")
    click.secho("Press Ctrl+C to stop.", fg="yellow")
    click.secho(
        "Loading parameter schemas at startup (~5–10 s on first launch)…",
        fg="yellow",
    )

    env = os.environ.copy()
    env["HABIT_GUI_ROOT"] = str(gui_root)
    env["HABIT_REPO_ROOT"] = str(repo_root)
    env["HABIT_GUI_API_HOST"] = host
    env["HABIT_GUI_API_PORT"] = str(port)
    env["HABIT_GUI_STATIC_DIR"] = str(static_dir)
    env["HABIT_CLI"] = _resolve_habit_cli_executable()
    env["HABIT_CLI_USE_MODULE"] = "1" if _habit_cli_uses_module() else "0"
    env["PYTHONPATH"] = str(api_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    ensure_localhost_no_proxy(env)

    if open_browser:
        schedule_browser_open(browser_url, delay_seconds=1.5, server_port=port)

    cmd: list[str] = [sys.executable, "-m", "habit_gui_api"]
    try:
        subprocess.run(cmd, env=env, cwd=str(api_dir), check=True)
    except KeyboardInterrupt:
        click.secho("\nHABIT GUI stopped.", fg="yellow")
    except subprocess.CalledProcessError as exc:
        click.secho(f"\nHABIT GUI exited with code {exc.returncode}.", fg="red", err=True)
        raise SystemExit(exc.returncode) from exc


# Backward-compatible alias used by older imports.
run_gui_server = run_next_gui_server
