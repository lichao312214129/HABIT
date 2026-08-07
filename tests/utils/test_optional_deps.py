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


def test_pyradiomics_hint_mentions_habitat_analysis_and_helper() -> None:
    """The public install recipe must point at the radiomics helper."""
    hint = pyradiomics_install_hint(python_version=(3, 10))
    assert "habitat-analysis[radiomics]" in hint
    assert "python -m habit.install_radiomics" in hint
    assert "pip install pyradiomics" in hint


def test_pyradiomics_hint_warns_on_unsupported_windows_python() -> None:
    """Unsupported Windows CPython must get an explicit wheel-coverage warning."""
    with mock.patch("habit.utils.optional_deps.sys.platform", "win32"):
        hint = pyradiomics_install_hint(python_version=(3, 9))
    assert "3.9" in hint
    assert "prebuilt PyRadiomics wheels" in hint


def test_require_pyradiomics_raises_optional_dependency_error() -> None:
    """Missing radiomics must surface OptionalDependencyError, not ModuleNotFoundError."""
    with mock.patch(
        "habit.utils.optional_deps.importlib.import_module",
        side_effect=ModuleNotFoundError(name="radiomics"),
    ), mock.patch(
        "habit.utils.optional_deps.sys.platform",
        "linux",
    ):
        with pytest.raises(OptionalDependencyError) as exc_info:
            require_pyradiomics()
    assert "habitat-analysis[radiomics]" in str(exc_info.value)
    assert "python -m habit.install_radiomics" in str(exc_info.value)


def test_require_pyradiomics_auto_installs_windows_wheel() -> None:
    """On Windows, one automatic wheel install is attempted before failing."""
    fake_module = ModuleType("radiomics")
    import_calls = {"n": 0}

    def _import(name: str) -> ModuleType:
        import_calls["n"] += 1
        if import_calls["n"] == 1:
            raise ModuleNotFoundError(name="radiomics")
        return fake_module

    with mock.patch(
        "habit.utils.optional_deps.importlib.import_module",
        side_effect=_import,
    ), mock.patch(
        "habit.utils.optional_deps.sys.platform",
        "win32",
    ), mock.patch(
        "habit.install_radiomics.try_install_windows_wheel",
        return_value=True,
    ) as auto_install:
        module = require_pyradiomics()
    auto_install.assert_called_once_with()
    assert module is fake_module


def test_require_pyradiomics_returns_module_when_available() -> None:
    """When PyRadiomics is installed (CI / developer envs), import succeeds."""
    pytest.importorskip("radiomics")
    module = require_pyradiomics()
    assert isinstance(module, ModuleType)
    assert module.__name__ == "radiomics"
