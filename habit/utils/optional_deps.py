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
"""Helpers for optional third-party backends that are hard to install via pip.

PyRadiomics is the main offender: PyPI has no usable Windows binaries for
CPython 3.10–3.14 (broken 3.1.0 sdist; no 3.0.1 win_amd64 wheels), and PyPI
forbids PEP 508 direct URL references in uploaded package metadata. HABIT
therefore keeps PyRadiomics out of the default dependency set, declares the
``radiomics`` extra for non-Windows only, and installs Windows wheels through
:mod:`habit.install_radiomics`.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Optional

from habit.exceptions import OptionalDependencyError

#: Human-readable install recipe shown whenever PyRadiomics is missing.
PYRADIOMICS_INSTALL_HINT: str = (
    "PyRadiomics is required for radiomics / voxel_radiomics / "
    "supervoxel_radiomics workflows.\n"
    "\n"
    "Install the HABIT radiomics extra, then ensure PyRadiomics is present:\n"
    "  pip install \"habitat-analysis[radiomics]\"\n"
    "  python -m habit.install_radiomics\n"
    "\n"
    "On Windows, ``python -m habit.install_radiomics`` installs the prebuilt\n"
    "3.1.0 wheel from the HABIT GitHub Release (cp310–cp314, win_amd64).\n"
    "On macOS / Linux it installs ``pyradiomics`` from PyPI\n"
    "(``pyradiomics>=3.0.1,<3.2``).\n"
    "\n"
    "Do NOT use bare ``pip install pyradiomics`` on Windows — PyPI serves a\n"
    "broken sdist that fails to compile."
)


def pyradiomics_install_hint(*, python_version: Optional[tuple[int, int]] = None) -> str:
    """
    Return the install hint, with an extra warning on unsupported Windows Python.

    Args:
        python_version: ``(major, minor)`` override; defaults to the running
            interpreter.

    Returns:
        Multi-line install guidance string.
    """
    from habit.install_radiomics import SUPPORTED_WINDOWS_CPYTHON_MINORS

    version = python_version or sys.version_info[:2]
    hint = PYRADIOMICS_INSTALL_HINT
    if sys.platform == "win32" and (
        version[0] != 3 or version[1] not in SUPPORTED_WINDOWS_CPYTHON_MINORS
    ):
        supported = ", ".join(f"3.{minor}" for minor in SUPPORTED_WINDOWS_CPYTHON_MINORS)
        hint += (
            "\n\n"
            f"This interpreter is Python {version[0]}.{version[1]} on Windows. "
            f"HABIT publishes prebuilt PyRadiomics wheels only for {supported}. "
            "Switch to a supported Python, or install PyRadiomics another way."
        )
    return hint


def require_pyradiomics() -> ModuleType:
    """
    Import the ``radiomics`` package or raise :class:`OptionalDependencyError`.

    On Windows, a missing install triggers one automatic attempt to fetch the
    HABIT GitHub Release wheel via :mod:`habit.install_radiomics` before the
    error is raised.

    Returns:
        The imported ``radiomics`` module.

    Raises:
        OptionalDependencyError: When PyRadiomics is not installed / importable.
    """
    try:
        return importlib.import_module("radiomics")
    except ModuleNotFoundError as exc:
        if exc.name not in {None, "radiomics"} and not str(exc.name or "").startswith(
            "radiomics."
        ):
            raise
        if sys.platform == "win32":
            from habit.install_radiomics import try_install_windows_wheel

            if try_install_windows_wheel():
                try:
                    return importlib.import_module("radiomics")
                except ImportError:
                    pass
        raise OptionalDependencyError(pyradiomics_install_hint()) from exc
    except ImportError as exc:
        # Broken partial installs (failed C extension, wrong ABI, …).
        raise OptionalDependencyError(
            "PyRadiomics is installed but failed to import "
            f"({type(exc).__name__}: {exc}).\n\n" + pyradiomics_install_hint()
        ) from exc
