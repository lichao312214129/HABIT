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
"""Tests for bundled PyRadiomics preset resolution."""

from __future__ import annotations

import os

import pytest

from habit.utils.radiomics_preset_utils import (
    available_presets,
    get_preset_path,
    is_preset_ref,
    resolve_params_file,
)


def test_available_presets_includes_all_workflows() -> None:
    """All workflow presets must be registered."""
    presets = set(available_presets())
    assert {"voxel", "supervoxel", "roi", "habitat"}.issubset(presets)


def test_get_preset_path_returns_existing_file() -> None:
    """Bundled preset YAML files must exist on disk after install."""
    for key in ("voxel", "supervoxel", "roi", "habitat"):
        path = get_preset_path(key)
        assert os.path.isfile(path), f"preset '{key}' missing at {path}"


def test_resolve_params_file_user_path_wins() -> None:
    """User-provided paths are returned unchanged."""
    assert resolve_params_file("./custom.yaml", "voxel") == "./custom.yaml"


def test_resolve_params_file_none_uses_preset() -> None:
    """Missing value falls back to the requested preset bundle."""
    path = resolve_params_file(None, "voxel")
    assert path.endswith("params_voxel_radiomics.yaml")
    assert os.path.isfile(path)


def test_resolve_params_file_at_preset_ref() -> None:
    """@preset: references resolve to bundled files."""
    assert is_preset_ref("@preset:habitat")
    path = resolve_params_file("@preset:habitat", "voxel")
    assert path.endswith("parameter_habitat.yaml")


def test_resolve_params_file_unknown_preset_raises() -> None:
    """Invalid @preset references raise ValueError."""
    with pytest.raises(ValueError, match="Unknown radiomics preset"):
        resolve_params_file("@preset:unknown", "voxel")
