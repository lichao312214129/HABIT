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
"""Tests for optional-dependency install hints (PyRadiomics packaging)."""

from __future__ import annotations

from types import ModuleType
from unittest import mock

import pytest

from habit.exceptions import OptionalDependencyError
from habit.utils.optional_deps import (
    pyradiomics_install_hint,
    require_pyradiomics,
)


def test_pyradiomics_hint_mentions_no_build_isolation_and_conda() -> None:
    """The public install recipe must stay actionable for pip and conda users."""
    hint = pyradiomics_install_hint(python_version=(3, 10))
    assert "conda-forge" in hint
    assert "--no-build-isolation" in hint
    assert "HABIT[radiomics]" in hint
    assert "3.12" not in hint or "Python 3.10" in hint


def test_pyradiomics_hint_warns_on_python_312() -> None:
    """Python 3.12+ must get an explicit upstream-packaging warning."""
    hint = pyradiomics_install_hint(python_version=(3, 12))
    assert "3.12" in hint
    assert "conda-forge" in hint


def test_require_pyradiomics_raises_optional_dependency_error() -> None:
    """Missing radiomics must surface OptionalDependencyError, not ModuleNotFoundError."""
    with mock.patch(
        "habit.utils.optional_deps.importlib.import_module",
        side_effect=ModuleNotFoundError(name="radiomics"),
    ):
        with pytest.raises(OptionalDependencyError) as exc_info:
            require_pyradiomics()
    assert "HABIT[radiomics]" in str(exc_info.value)
    assert "--no-build-isolation" in str(exc_info.value)


def test_require_pyradiomics_returns_module_when_available() -> None:
    """When PyRadiomics is installed (CI / developer envs), import succeeds."""
    pytest.importorskip("radiomics")
    module = require_pyradiomics()
    assert isinstance(module, ModuleType)
    assert module.__name__ == "radiomics"
