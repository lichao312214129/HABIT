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
"""Gap tests for ``examples/habitat_v1_two_step_demo.py`` wiring."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_SCRIPT: Path = (
    Path(__file__).resolve().parents[2] / "examples" / "habitat_v1_two_step_demo.py"
)
EXPECTED_MODEL: Path = (
    Path(__file__).resolve().parents[2]
    / "demo_data"
    / "results"
    / "examples"
    / "habitat_v1_two_step_demo"
    / "habitat_model.habitatmodel"
)


def _run_example(
    repo_root: Path,
    *extra_args: str,
    timeout_sec: int = 300,
) -> subprocess.CompletedProcess[str]:
    """
    Execute the shipped two-step demo script from the repository root.

    Args:
        repo_root: Working directory for the subprocess.
        *extra_args: Extra CLI flags forwarded to the script.
        timeout_sec: Maximum wall-clock seconds before aborting.

    Returns:
        Completed process with captured stdout/stderr.
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing}" if existing else str(repo_root)
    )
    return subprocess.run(
        [sys.executable, str(EXAMPLE_SCRIPT), *extra_args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
        env=env,
    )


@pytest.mark.integration
def test_example_demo_dry_run(repo_root: Path, demo_data_root: Path) -> None:
    """``--dry-run`` validates spec, components and cohort without compute."""
    assert demo_data_root.is_dir()
    result = _run_example(repo_root, "--dry-run", timeout_sec=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Dry-run OK" in result.stdout


@pytest.mark.integration
def test_example_demo_full_run(repo_root: Path, demo_data_root: Path) -> None:
    """
    Full demo run completes and writes ``habitat_model.habitatmodel``.

    Output directory is cleared first so a stale artefact cannot mask failure.
    """
    assert demo_data_root.is_dir()
    out_dir = EXPECTED_MODEL.parent
    if out_dir.exists():
        shutil.rmtree(out_dir)

    result = _run_example(repo_root, timeout_sec=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Done." in result.stdout
    assert EXPECTED_MODEL.is_file(), f"Expected model archive at {EXPECTED_MODEL}"
