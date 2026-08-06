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

PyRadiomics is the main offender: PyPI often serves only an sdist, the sdist
does not declare ``numpy`` as a build dependency, and Python 3.12+ breaks the
upstream ``versioneer`` / ``SafeConfigParser`` packaging path. HABIT therefore
keeps PyRadiomics out of the default ``pip install habitat-analysis`` dependency set and
loads it lazily with an actionable error message.
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
    "Recommended (conda-forge):\n"
    "  conda install -c conda-forge pyradiomics\n"
    "\n"
    "pip (prefer the 3.0.1 wheel; avoid 3.1.0 sdist on PyPI):\n"
    "  pip install \"pyradiomics==3.0.1\"\n"
    "  pip install \"habitat-analysis[radiomics]\"\n"
    "\n"
    "Do NOT use bare ``pip install pyradiomics`` — it often pulls the broken\n"
    "3.1.0 source distribution (missing versioneer). Python 3.12+: prefer conda.\n"
    "\n"
    "Windows portable ZIP: the installer installs a prebuilt "
    "cp310 wheel from installer/vendor/ when present."
)


def pyradiomics_install_hint(*, python_version: Optional[tuple[int, int]] = None) -> str:
    """
    Return the install hint, with an extra warning on unsupported Python.

    Args:
        python_version: ``(major, minor)`` override; defaults to the running
            interpreter.

    Returns:
        Multi-line install guidance string.
    """
    version = python_version or sys.version_info[:2]
    hint = PYRADIOMICS_INSTALL_HINT
    if version >= (3, 12):
        hint += (
            "\n\n"
            f"This interpreter is Python {version[0]}.{version[1]}. "
            "Official PyRadiomics releases on PyPI do not install cleanly on "
            "Python 3.12+. Use conda-forge, or run HABIT on Python 3.10/3.11 "
            "when you need radiomics features."
        )
    return hint


def require_pyradiomics() -> ModuleType:
    """
    Import the ``radiomics`` package or raise :class:`OptionalDependencyError`.

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
        raise OptionalDependencyError(pyradiomics_install_hint()) from exc
    except ImportError as exc:
        # Broken partial installs (failed C extension, wrong ABI, …).
        raise OptionalDependencyError(
            "PyRadiomics is installed but failed to import "
            f"({type(exc).__name__}: {exc}).\n\n" + pyradiomics_install_hint()
        ) from exc
