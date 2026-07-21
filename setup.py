"""Setuptools build configuration for the HABIT package."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from setuptools import Extension, find_packages, setup

try:
    import tomllib
except ModuleNotFoundError:
    # Python 3.10 does not include ``tomllib``. Build isolation installs
    # ``tomli`` from pyproject.toml so legacy setup.py invocations use the same
    # standards-compliant parser without introducing a second metadata source.
    import tomli as tomllib


_ROOT = Path(__file__).resolve().parent
_CEXT_SRC = "habit/core/habitat_analysis/clustering_features/supervoxel_cext/src"
_SV_CMATRICES_MODULE = (
    "habit.core.habitat_analysis.clustering_features.supervoxel_cext._sv_cmatrices"
)


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


setup(
    name="HABIT",
    version=_read_version(),
    description="Habitat Analysis: Biomedical Imaging Toolkit",
    author="lichao19870617@163.com",
    license="HABIT Software License",
    # Restrict discovery to HABIT itself. The repository's ``tests`` directory
    # is an importable package but must not be installed into user environments.
    packages=find_packages(include=("habit", "habit.*")),
    include_package_data=True,
    package_data={
        # Bundled PyRadiomics parameter presets (default params_file fallbacks).
        "habit": ["py.typed"],
        "habit.resources.radiomics": ["*.yaml"],
    },
    ext_modules=[
        Extension(
            _SV_CMATRICES_MODULE,
            [
                f"{_CEXT_SRC}/_sv_cmatrices.c",
                f"{_CEXT_SRC}/sv_cmatrices.c",
            ],
            include_dirs=[_CEXT_SRC, np.get_include()],
        ),
    ],
    install_requires=_read_runtime_dependencies(),
    entry_points={
        "console_scripts": [
            "habit=habit.cli:cli",
        ],
    },
    python_requires=_read_python_requirement(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3.10",
    ],
)
