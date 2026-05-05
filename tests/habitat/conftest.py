"""
Shared pytest fixtures for tests under ``tests/habitat/``.

The CLI reads paths relative to the process working directory; demo YAML files
assume invocation from the repository root. Tests that run ``habit ...`` against
those YAMLs should depend on ``cwd_project_root`` so behavior matches real users.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# tests/habitat/conftest.py -> parents[2] is the repository root
_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cwd_project_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Change working directory to the repository root for the duration of the test.

    Without this, relative paths inside demo YAML (e.g. ``./preprocessed/``) may
    resolve incorrectly when pytest is launched from another cwd.
    """
    monkeypatch.chdir(_REPO_ROOT)
