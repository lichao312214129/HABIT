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
"""Setuptools build configuration for the HABIT package."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

try:
    import tomllib
except ModuleNotFoundError:
    # Python 3.10 does not include ``tomllib``. Build isolation installs
    # ``tomli`` from pyproject.toml so legacy setup.py invocations use the same
    # standards-compliant parser without introducing a second metadata source.
    import tomli as tomllib  # type: ignore[no-redef]


_ROOT = Path(__file__).resolve().parent
_CEXT_SRC = "habit/kernels/radiomics/cext/src"
_SV_CMATRICES_MODULE = "habit.kernels.radiomics.cext._sv_cmatrices"


def _read_version() -> str:
    """
    Read the package version without importing ``habit`` during the build.

    Importing the package here would execute its public API initialization before
    runtime dependencies are installed in the isolated build environment.

    Returns:
        str: The version declared in ``habit/_version.py``.
    """
    version_scope: Dict[str, object] = {}
    version_file = _ROOT / "habit" / "_version.py"
    exec(
        compile(version_file.read_text(encoding="utf-8"), str(version_file), "exec"),
        version_scope,
    )
    return str(version_scope["__version__"])


def _read_pyproject() -> Dict[str, Any]:
    """
    Load the canonical package metadata declared in ``pyproject.toml``.

    Keeping this parsing in one helper makes ``project.dependencies`` the only
    manually maintained runtime dependency list while preserving the existing
    setuptools entry point needed to compile HABIT's C extension.

    Returns:
        Dict[str, Any]: Parsed TOML metadata for the current source tree.
    """
    pyproject_file = _ROOT / "pyproject.toml"
    with pyproject_file.open("rb") as stream:
        return dict(tomllib.load(stream))


def _read_runtime_dependencies() -> List[str]:
    """
    Return runtime requirements from the PEP 621 project metadata.

    Returns:
        List[str]: Exact direct dependency specifications consumed by setuptools.

    Raises:
        ValueError: If ``project.dependencies`` is absent or malformed.
    """
    project = _read_pyproject().get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml must define a [project] table.")
    dependencies = project.get("dependencies")
    if not isinstance(dependencies, list) or not all(
        isinstance(item, str) for item in dependencies
    ):
        raise ValueError("pyproject.toml project.dependencies must be a string list.")
    return list(dependencies)


def _read_python_requirement() -> str:
    """
    Return the interpreter constraint from the canonical project metadata.

    Returns:
        str: PEP 440 Python version requirement used by setuptools.

    Raises:
        ValueError: If ``project.requires-python`` is absent or malformed.
    """
    project = _read_pyproject().get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml must define a [project] table.")
    requirement = project.get("requires-python")
    if not isinstance(requirement, str):
        raise ValueError("pyproject.toml project.requires-python must be a string.")
    return requirement


class _OptionalBuildExt(build_ext):
    """
    Build C extensions when a compiler is available; otherwise skip them.

    Supervoxel radiomics has a pure-Python / PyRadiomics ``cMatrices`` fallback
    in ``habit.kernels.radiomics.cext``, so a missing compiler must not make
    ``pip install HABIT`` fail on Windows machines without MSVC or on slim CI
    images. A successful native build is still preferred when possible.
    """

    def build_extensions(self) -> None:
        try:
            super().build_extensions()
        except Exception as exc:  # noqa: BLE001 — intentional soft-fail for packaging
            self._skip_all_extensions(exc)

    def build_extension(self, ext: Extension) -> None:
        try:
            super().build_extension(ext)
        except Exception as exc:  # noqa: BLE001
            self._skip_extension(ext, exc)

    def _skip_all_extensions(self, exc: BaseException) -> None:
        for ext in list(self.extensions):
            self._skip_extension(ext, exc)
        self.extensions = []

    def _skip_extension(self, ext: Extension, exc: BaseException) -> None:
        sys.stderr.write(
            f"\nWARNING: skipping HABIT C extension {ext.name!r}: {exc}\n"
            "         Supervoxel radiomics will use the Python fallback "
            "(habit.kernels.radiomics.cext). Install a C compiler to enable "
            "the native acceleration path.\n\n"
        )


def _extension_modules() -> Sequence[Extension]:
    """Return the optional supervoxel radiomics C extension list."""
    return [
        Extension(
            _SV_CMATRICES_MODULE,
            [
                f"{_CEXT_SRC}/_sv_cmatrices.c",
                f"{_CEXT_SRC}/sv_cmatrices.c",
            ],
            include_dirs=[_CEXT_SRC, np.get_include()],
        )
    ]


setup(
    # Keep in sync with [project].name in pyproject.toml (PyPI forbids "HABIT").
    name="habitat-analysis",
    version=_read_version(),
    description="Habitat Analysis: Biomedical Imaging Toolkit (HABIT)",
    author="lichao19870617@163.com",
    license="Apache-2.0",
    # Restrict discovery to HABIT itself. The repository's ``tests`` directory
    # is an importable package but must not be installed into user environments.
    packages=find_packages(include=("habit", "habit.*")),
    include_package_data=True,
    package_data={
        # Bundled PyRadiomics parameter presets (default params_file fallbacks).
        "habit": ["py.typed"],
        "habit.resources.radiomics": ["*.yaml"],
    },
    cmdclass={"build_ext": _OptionalBuildExt},
    ext_modules=list(_extension_modules()),
    install_requires=_read_runtime_dependencies(),
    # Entry points are intentionally NOT declared here: an explicit dict would
    # override the PEP 621 metadata ([project.scripts] console script and the
    # [project.entry-points."habit.*"] plugin groups) in pyproject.toml, which
    # is the single source of truth for HABIT's packaging contract.
    python_requires=_read_python_requirement(),
)
