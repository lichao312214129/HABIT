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
"""``habit fetch-demo``: download the official preprocessed imaging pack once."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

import click

from habit.common import echo_error, echo_success, exit_with_error


def run_fetch_demo(
    *,
    force: bool = False,
    work_dir: Optional[str] = None,
) -> Path:
    """
    Download (or reuse) the official demo pack and print its layout.

    Args:
        force: Re-download even when a valid cache exists.
        work_dir: Optional work directory. When set, create
            ``<work_dir>/demo_data/preprocessed`` pointing at the cache so
            shipped YAML files that use ``../../demo_data/preprocessed`` keep
            working.

    Returns:
        Absolute preprocessed root (``images/`` + ``masks/``).
    """
    from habit.datasets import fetch_demo
    from habit.exceptions import DataFormatError, HabitError

    try:
        demo_root = fetch_demo(force=force, verbose=True)
    except DataFormatError as exc:
        exit_with_error(f"Error: {exc}")
    except HabitError as exc:
        echo_error(f"Error downloading demo data: {exc}")
        raise SystemExit(1) from exc

    echo_success(f"Demo preprocessed root: {demo_root}")
    if work_dir:
        linked = _link_into_work_dir(Path(work_dir), demo_root)
        echo_success(f"Linked for CLI YAML: {linked}")
    else:
        click.echo(
            "Python: DATA = fetch_demo()  then cohort_from_directory(DATA, ...)\n"
            "CLI YAML: pass --work-dir <work_dir> so demo_data/preprocessed exists "
            "next to config/, or paste the path above into data_dir."
        )
    return demo_root


def _link_into_work_dir(work_dir: Path, demo_root: Path) -> Path:
    """
    Point ``work_dir/demo_data/preprocessed`` at ``demo_root``.

    Args:
        work_dir: User work directory (contains or will contain ``config/``).
        demo_root: Canonical cached preprocessed root.

    Returns:
        The link / junction path.

    Raises:
        SystemExit: If the destination exists and is not already the cache.
    """
    work_dir = work_dir.expanduser().resolve()
    link = work_dir / "demo_data" / "preprocessed"
    if link.exists() or link.is_symlink():
        try:
            if link.resolve() == demo_root.resolve():
                return link
        except OSError:
            pass
        exit_with_error(
            f"Error: {link} already exists and is not the cached demo pack. "
            "Remove it or pass a different --work-dir."
        )
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(demo_root, target_is_directory=True)
        return link
    except OSError:
        if os.name == "nt":
            completed = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(demo_root)],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                return link
            echo_error(completed.stderr.strip() or completed.stdout.strip())
        exit_with_error(
            f"Error: could not link {link} -> {demo_root}. "
            "Paste the cached path into your YAML data_dir instead."
        )
    return link
