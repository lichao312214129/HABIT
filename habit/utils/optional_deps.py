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
CPython 3.10–3.14 (broken 3.1.0 sdist; no 3.0.1 win_amd64 wheels). HABIT
therefore never declares ``pyradiomics`` as a pip dependency; users install
it separately. On Windows, use the prebuilt wheels published on the HABIT
GitHub Release ``v1.0.2``; on macOS / Linux use PyPI or conda-forge.
See the Installation tutorial for the full wheel URL table.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Optional

from habit.exceptions import OptionalDependencyError

#: CPython minor versions for which HABIT publishes ``win_amd64`` wheels.
SUPPORTED_WINDOWS_CPYTHON_MINORS: tuple[int, ...] = (10, 11, 12, 13, 14)

#: GitHub Release tag that hosts the HABIT-built PyRadiomics Windows wheels.
PYRADIOMICS_WHEEL_RELEASE: str = "v1.0.2"

#: Upstream PyRadiomics version encoded in the Release wheel filenames.
PYRADIOMICS_WHEEL_VERSION: str = "3.1.0"

#: Docs page with the platform-specific install recipe and wheel table.
INSTALLATION_DOCS_URL: str = (
    "https://lichao312214129.github.io/HABIT/tutorial/installation.html"
)

#: Human-readable install recipe shown whenever PyRadiomics is missing.
PYRADIOMICS_INSTALL_HINT: str = (
    "PyRadiomics is required for radiomics / voxel_radiomics / "
    "supervoxel_radiomics workflows.\n"
    "\n"
    "Install PyRadiomics separately (HABIT does not pull it via pip extras):\n"
    "\n"
    "  Windows (prebuilt wheel from the HABIT GitHub Release v1.0.2;\n"
    "  pick the URL matching your CPython — do NOT use bare\n"
    "  ``pip install pyradiomics``, which downloads a broken sdist):\n"
    "    pip install https://github.com/lichao312214129/HABIT/releases/"
    "download/v1.0.2/pyradiomics-3.1.0-cp310-cp310-win_amd64.whl\n"
    "    # … or cp311 / cp312 / cp313 / cp314 — see the Installation docs\n"
    "\n"
    "  macOS / Linux:\n"
    "    pip install \"pyradiomics>=3.0.1,<3.2\"\n"
    "    # or: conda install -c conda-forge pyradiomics\n"
    "\n"
    f"Full wheel table and notes: {INSTALLATION_DOCS_URL}"
)


def windows_pyradiomics_wheel_url(
    *,
    python_version: Optional[tuple[int, int]] = None,
    release: str = PYRADIOMICS_WHEEL_RELEASE,
    pyradiomics_version: str = PYRADIOMICS_WHEEL_VERSION,
) -> str:
    """
    Return the GitHub Release URL for the prebuilt Windows PyRadiomics wheel.

    Args:
        python_version: ``(major, minor)`` interpreter tag; defaults to the
            running interpreter.
        release: HABIT GitHub Release tag that hosts the wheel assets.
        pyradiomics_version: Version string embedded in the wheel filename.

    Returns:
        Absolute HTTPS URL of the ``win_amd64`` wheel.

    Raises:
        ValueError: When no wheel exists for the requested CPython version.
    """
    version = python_version or sys.version_info[:2]
    if version[0] != 3 or version[1] not in SUPPORTED_WINDOWS_CPYTHON_MINORS:
        supported = ", ".join(f"3.{minor}" for minor in SUPPORTED_WINDOWS_CPYTHON_MINORS)
        raise ValueError(
            f"No HABIT prebuilt PyRadiomics wheel for Python {version[0]}.{version[1]} "
            f"on Windows. Supported: {supported}."
        )
    tag = f"cp3{version[1]}"
    filename = (
        f"pyradiomics-{pyradiomics_version}-{tag}-{tag}-win_amd64.whl"
    )
    return (
        f"https://github.com/lichao312214129/HABIT/releases/download/"
        f"{release}/{filename}"
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
    version = python_version or sys.version_info[:2]
    hint = PYRADIOMICS_INSTALL_HINT
    if sys.platform == "win32":
        if version[0] != 3 or version[1] not in SUPPORTED_WINDOWS_CPYTHON_MINORS:
            supported = ", ".join(
                f"3.{minor}" for minor in SUPPORTED_WINDOWS_CPYTHON_MINORS
            )
            hint += (
                "\n\n"
                f"This interpreter is Python {version[0]}.{version[1]} on Windows. "
                f"HABIT publishes prebuilt PyRadiomics wheels only for {supported}. "
                "Switch to a supported Python, or install PyRadiomics another way."
            )
        else:
            try:
                url = windows_pyradiomics_wheel_url(python_version=version)
            except ValueError:
                pass
            else:
                hint += (
                    "\n\n"
                    f"For this interpreter, the matching wheel is:\n"
                    f"  pip install {url}"
                )
    return hint


def require_pyradiomics() -> ModuleType:
    """
    Import the ``radiomics`` package or raise :class:`OptionalDependencyError`.

    No automatic download or install is attempted; the error message points at
    the documented separate-install recipe (Windows Release wheel / PyPI /
    conda-forge).

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
