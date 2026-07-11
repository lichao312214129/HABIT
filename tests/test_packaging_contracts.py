"""Regression tests for HABIT's non-editable distribution contents."""

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
