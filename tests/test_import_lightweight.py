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
"""Contract tests keeping ``import habit`` lightweight.

A bare ``import habit`` must stay cheap enough for notebook and CLI startup:
the v0.1 orchestration stack (sklearn/pandas/scipy via ``habit.core``) and the
imaging backends (SimpleITK, torch, PyRadiomics, matplotlib) load lazily on
first symbol access, never at package import time. Every assertion runs in a
fresh interpreter (``sys.executable -c``) because ``sys.modules`` is global
state that the pytest process itself has already polluted.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Tuple

import pytest

#: Third-party stacks a bare ``import habit`` must never pull in.
_BARE_IMPORT_FORBIDDEN: Tuple[str, ...] = (
    "sklearn",
    "pandas",
    "scipy",
    "SimpleITK",
    "torch",
    "matplotlib",
    "radiomics",
)

#: ``habit.contracts`` value objects are pandas/numpy-typed by design (a
#: ``FeatureTable`` wraps a DataFrame), so only the orchestration and imaging
#: stacks are forbidden there.
_CONTRACTS_FORBIDDEN: Tuple[str, ...] = (
    "sklearn",
    "scipy",
    "SimpleITK",
    "torch",
    "matplotlib",
    "radiomics",
)

#: ``habit.kernels`` are pure numerical functions; numpy/scipy are their
#: legitimate mathematical foundation, everything heavier is forbidden.
_KERNELS_FORBIDDEN: Tuple[str, ...] = (
    "sklearn",
    "pandas",
    "SimpleITK",
    "torch",
    "matplotlib",
    "radiomics",
)

#: Wall-clock budget for a bare ``import habit`` in a fresh interpreter. The
#: local median is ~0.02 s; the threshold stays generous to absorb CI jitter.
_IMPORT_BUDGET_SEC = 2.0


def _run_fresh_interpreter(script: str) -> str:
    """
    Run ``script`` in a clean Python subprocess and return combined output.

    Args:
        script: Python source executed via ``sys.executable -c``.

    Returns:
        ``stdout`` plus ``stderr`` of the finished process.

    Raises:
        AssertionError: When the subprocess exits with a non-zero status.
    """
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    assert completed.returncode == 0, output
    return output


def _loaded_modules_script(module: str, forbidden: Tuple[str, ...]) -> str:
    """
    Build a subprocess script reporting which forbidden modules got loaded.

    Args:
        module: Fully qualified module imported by the script.
        forbidden: Top-level package names that must stay out of
            ``sys.modules`` after the import.

    Returns:
        Source printing ``LOADED=<comma-separated offenders>``.
    """
    names = ",".join(forbidden)
    return (
        "import sys\n"
        f"import {module}\n"
        f"forbidden = '{names}'.split(',')\n"
        "loaded = sorted(name for name in forbidden if name in sys.modules)\n"
        "print('LOADED=' + ','.join(loaded))\n"
    )


@pytest.mark.unit
def test_import_habit_excludes_heavy_stacks() -> None:
    """Bare ``import habit`` loads none of the heavy third-party stacks."""
    output = _run_fresh_interpreter(
        _loaded_modules_script("habit", _BARE_IMPORT_FORBIDDEN)
    )
    assert "LOADED=\n" in output or "LOADED=\r\n" in output, output


@pytest.mark.unit
def test_import_habit_within_time_budget() -> None:
    """Bare ``import habit`` completes well under one second locally."""
    script = (
        "import time\n"
        "start = time.perf_counter()\n"
        "import habit\n"
        "elapsed = time.perf_counter() - start\n"
        "print(f'ELAPSED={elapsed:.3f}')\n"
        f"assert elapsed < {_IMPORT_BUDGET_SEC}, f'import habit took {{elapsed:.3f}}s'\n"
    )
    output = _run_fresh_interpreter(script)
    assert "ELAPSED=" in output, output


@pytest.mark.unit
@pytest.mark.parametrize(
    ("module", "forbidden"),
    [
        ("habit.contracts", _CONTRACTS_FORBIDDEN),
        ("habit.kernels", _KERNELS_FORBIDDEN),
    ],
    ids=["contracts", "kernels"],
)
def test_layered_package_excludes_heavy_stacks(
    module: str, forbidden: Tuple[str, ...]
) -> None:
    """v1.0 layered packages import without the v0.1 orchestration stack."""
    output = _run_fresh_interpreter(_loaded_modules_script(module, forbidden))
    assert "LOADED=\n" in output or "LOADED=\r\n" in output, output
