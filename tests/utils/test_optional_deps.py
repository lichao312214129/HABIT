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
    INSTALLATION_DOCS_URL,
    pyradiomics_install_hint,
    require_pyradiomics,
    windows_pyradiomics_wheel_url,
)


def test_pyradiomics_hint_points_at_separate_install() -> None:
    """The public install recipe must document separate PyRadiomics install."""
    hint = pyradiomics_install_hint(python_version=(3, 10))
    assert "pip install pyradiomics" in hint
    assert "conda-forge" in hint
    assert INSTALLATION_DOCS_URL in hint
    assert "habit.install_radiomics" not in hint
    assert "v1.0.2" in hint


def test_pyradiomics_hint_warns_on_unsupported_windows_python() -> None:
    """Unsupported Windows CPython must get an explicit wheel-coverage warning."""
    with mock.patch("habit.utils.optional_deps.sys.platform", "win32"):
        hint = pyradiomics_install_hint(python_version=(3, 9))
    assert "3.9" in hint
    assert "prebuilt PyRadiomics wheels" in hint


def test_pyradiomics_hint_includes_matching_windows_wheel() -> None:
    """Supported Windows CPython hints include the concrete Release wheel URL."""
    with mock.patch("habit.utils.optional_deps.sys.platform", "win32"):
        hint = pyradiomics_install_hint(python_version=(3, 12))
    expected = windows_pyradiomics_wheel_url(python_version=(3, 12))
    assert expected in hint


@pytest.mark.parametrize("minor", [10, 11, 12, 13, 14])
def test_windows_wheel_url_maps_supported_cpython(minor: int) -> None:
    """Each supported CPython minor must map to the matching Release asset."""
    url = windows_pyradiomics_wheel_url(python_version=(3, minor))
    tag = f"cp3{minor}"
    assert url.startswith(
        "https://github.com/lichao312214129/HABIT/releases/download/v1.0.2/"
    )
    assert f"pyradiomics-3.1.0-{tag}-{tag}-win_amd64.whl" in url


def test_windows_wheel_url_rejects_unsupported_python() -> None:
    """Unsupported interpreters must fail loudly instead of inventing a URL."""
    with pytest.raises(ValueError, match="No HABIT prebuilt PyRadiomics wheel"):
        windows_pyradiomics_wheel_url(python_version=(3, 9))


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
    message = str(exc_info.value)
    assert INSTALLATION_DOCS_URL in message
    assert "habit.install_radiomics" not in message


def test_require_pyradiomics_does_not_auto_install_on_windows() -> None:
    """On Windows, a missing install must raise immediately (no auto-download)."""
    with mock.patch(
        "habit.utils.optional_deps.importlib.import_module",
        side_effect=ModuleNotFoundError(name="radiomics"),
    ), mock.patch(
        "habit.utils.optional_deps.sys.platform",
        "win32",
    ):
        with pytest.raises(OptionalDependencyError) as exc_info:
            require_pyradiomics()
    assert "win_amd64" in str(exc_info.value)
    assert "habit.install_radiomics" not in str(exc_info.value)


def test_require_pyradiomics_returns_module_when_available() -> None:
    """When PyRadiomics is installed (CI / developer envs), import succeeds."""
    pytest.importorskip("radiomics")
    module = require_pyradiomics()
    assert isinstance(module, ModuleType)
    assert module.__name__ == "radiomics"
