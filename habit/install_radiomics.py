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
"""Install PyRadiomics for HABIT, using prebuilt Windows wheels when needed.

PyPI ships no usable PyRadiomics Windows binaries for CPython 3.10–3.14: the
3.1.0 sdist fails to build, and 3.0.1 has no ``win_amd64`` wheels. HABIT
publishes self-built 3.1.0 wheels as GitHub Release assets.

PyPI also rejects PEP 508 direct URL references in uploaded package metadata,
so ``habitat-analysis[radiomics]`` cannot declare those wheel URLs. On Windows
this module installs the matching Release wheel via ``pip``; on other
platforms it installs the normal PyPI range used by the ``radiomics`` extra.

Usage::

    python -m habit.install_radiomics
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Optional, Sequence

#: GitHub Release tag that hosts the HABIT-built PyRadiomics Windows wheels.
#: Bump this when a newer Release republishes updated wheel assets.
PYRADIOMICS_WHEEL_RELEASE: str = "v1.0.2"

#: Upstream PyRadiomics version encoded in the Release wheel filenames.
PYRADIOMICS_WHEEL_VERSION: str = "3.1.0"

#: Non-Windows / fallback requirement (same range as the ``radiomics`` extra).
PYRADIOMICS_PYPI_SPEC: str = "pyradiomics>=3.0.1,<3.2"

#: CPython minor versions for which HABIT publishes ``win_amd64`` wheels.
SUPPORTED_WINDOWS_CPYTHON_MINORS: tuple[int, ...] = (10, 11, 12, 13, 14)


def windows_wheel_url(
    *,
    python_version: Optional[tuple[int, int]] = None,
    release: str = PYRADIOMICS_WHEEL_RELEASE,
    pyradiomics_version: str = PYRADIOMICS_WHEEL_VERSION,
) -> str:
    """
    Return the GitHub Release URL for the prebuilt Windows wheel.

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


def radiomics_requirement(
    *,
    platform: Optional[str] = None,
    python_version: Optional[tuple[int, int]] = None,
) -> str:
    """
    Return the pip requirement string for the current (or given) platform.

    Args:
        platform: ``sys.platform`` value; defaults to the running platform.
        python_version: ``(major, minor)``; defaults to the running interpreter.

    Returns:
        A pip-installable requirement (PyPI range or direct wheel URL).
    """
    plat = sys.platform if platform is None else platform
    if plat == "win32":
        return windows_wheel_url(python_version=python_version)
    return PYRADIOMICS_PYPI_SPEC


def install_radiomics(
    *,
    platform: Optional[str] = None,
    python_version: Optional[tuple[int, int]] = None,
    pip_args: Optional[Sequence[str]] = None,
) -> None:
    """
    Install PyRadiomics with ``python -m pip``.

    On Windows this installs the HABIT GitHub Release wheel for the active
    CPython; elsewhere it installs ``pyradiomics`` from PyPI.

    Args:
        platform: Override ``sys.platform`` (tests / dry diagnostics).
        python_version: Override interpreter version tag.
        pip_args: Extra arguments forwarded to ``pip install``.

    Raises:
        ValueError: When Windows has no wheel for this Python version.
        subprocess.CalledProcessError: When pip exits non-zero.
    """
    requirement = radiomics_requirement(
        platform=platform,
        python_version=python_version,
    )
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        requirement,
        *(list(pip_args) if pip_args else []),
    ]
    print("Installing PyRadiomics via:", " ".join(command), file=sys.stderr)
    subprocess.check_call(command)


def try_install_windows_wheel() -> bool:
    """
    Best-effort Windows wheel install used by :func:`require_pyradiomics`.

    Returns:
        ``True`` when pip reported success; ``False`` when skipped (non-Windows)
        or when install failed (caller should raise OptionalDependencyError).
    """
    if sys.platform != "win32":
        return False
    try:
        install_radiomics()
    except (ValueError, subprocess.CalledProcessError, OSError) as exc:
        print(
            f"WARNING: automatic PyRadiomics wheel install failed: {exc}",
            file=sys.stderr,
        )
        return False
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    CLI entry point for ``python -m habit.install_radiomics``.

    Args:
        argv: Optional argument vector (without the program name).

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Install PyRadiomics for HABIT. On Windows, uses the prebuilt "
            "wheel from the HABIT GitHub Release; elsewhere installs from PyPI."
        )
    )
    parser.add_argument(
        "--print-requirement",
        action="store_true",
        help="Print the pip requirement string and exit (no install).",
    )
    parser.add_argument(
        "pip_args",
        nargs="*",
        help="Extra arguments forwarded to pip install (after --).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.print_requirement:
        try:
            print(radiomics_requirement())
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return 0
    try:
        install_radiomics(pip_args=args.pip_args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "Install a supported Python (3.10–3.14) or obtain PyRadiomics "
            "another way, then retry.",
            file=sys.stderr,
        )
        return 2
    except subprocess.CalledProcessError as exc:
        return int(exc.returncode or 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
