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
"""Tests for the Windows PyRadiomics wheel installer helper."""

from __future__ import annotations

import pytest

from habit.install_radiomics import (
    PYRADIOMICS_PYPI_SPEC,
    PYRADIOMICS_WHEEL_RELEASE,
    radiomics_requirement,
    windows_wheel_url,
)


@pytest.mark.parametrize("minor", [10, 11, 12, 13, 14])
def test_windows_wheel_url_maps_supported_cpython(minor: int) -> None:
    """Each supported CPython minor must map to the matching Release asset."""
    url = windows_wheel_url(python_version=(3, minor))
    tag = f"cp3{minor}"
    assert url.startswith(
        f"https://github.com/lichao312214129/HABIT/releases/download/"
        f"{PYRADIOMICS_WHEEL_RELEASE}/"
    )
    assert f"pyradiomics-3.1.0-{tag}-{tag}-win_amd64.whl" in url


def test_windows_wheel_url_rejects_unsupported_python() -> None:
    """Unsupported interpreters must fail loudly instead of inventing a URL."""
    with pytest.raises(ValueError, match="No HABIT prebuilt PyRadiomics wheel"):
        windows_wheel_url(python_version=(3, 9))


def test_radiomics_requirement_platform_split() -> None:
    """Windows uses the Release wheel URL; other platforms use the PyPI range."""
    win = radiomics_requirement(platform="win32", python_version=(3, 12))
    assert win.endswith("pyradiomics-3.1.0-cp312-cp312-win_amd64.whl")
    assert radiomics_requirement(platform="linux") == PYRADIOMICS_PYPI_SPEC
    assert radiomics_requirement(platform="darwin") == PYRADIOMICS_PYPI_SPEC
