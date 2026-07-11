"""Regression tests for HABIT's non-editable distribution contents."""

import re
from pathlib import Path
from typing import Set

from setuptools import find_packages


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _habit_packages() -> Set[str]:
    """
    Discover packages with the same restriction used by ``setup.py``.

    Returns:
        Set[str]: Importable package names that belong to HABIT.
    """
    return set(
        find_packages(
            where=str(PROJECT_ROOT),
            include=("habit", "habit.*"),
        )
    )


def test_machine_learning_statistics_is_distributable() -> None:
    """The evaluation dependency must be included in non-editable installs."""
    packages = _habit_packages()
    assert "habit.core.machine_learning.statistics" in packages


def test_distribution_excludes_repository_tests() -> None:
    """User installations must not expose the repository's test packages."""
    packages = _habit_packages()
    assert not any(name == "tests" or name.startswith("tests.") for name in packages)


def test_package_version_and_python_support_are_consistent() -> None:
    """Build metadata and Poetry metadata must describe the tested runtime."""
    version_scope: dict[str, object] = {}
    version_file = PROJECT_ROOT / "habit" / "_version.py"
    exec(
        compile(version_file.read_text(encoding="utf-8"), str(version_file), "exec"),
        version_scope,
    )
    package_version = str(version_scope["__version__"])
    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    version_match = re.search(
        r'^version\s*=\s*"([^"]+)"\s*$',
        pyproject_text,
        flags=re.MULTILINE,
    )
    assert version_match is not None
    assert version_match.group(1) == package_version
    assert 'python = ">=3.10,<3.11"' in pyproject_text
    assert (PROJECT_ROOT / "habit" / "py.typed").is_file()
