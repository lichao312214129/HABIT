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
"""Round-trip and hit/miss tests for voxel-radiomics field cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from habit.contracts.habitat import VoxelFeatureField
from habit.domain.voxel_features.cache import (
    load_cached_voxel_field,
    save_cached_voxel_field,
    voxel_radiomics_cache_key,
)

from .conftest import make_field


@pytest.mark.unit
def test_voxel_feature_field_save_load_roundtrip(tmp_path: Path) -> None:
    """Arrays, names, geometry and subject id survive the zip archive."""
    original = make_field("P1", n_voxels=8)
    path = tmp_path / "p1.vxff.zip"
    original.save(path)
    loaded = VoxelFeatureField.load(path)
    assert loaded.subject_id == original.subject_id
    assert loaded.feature_names == original.feature_names
    np.testing.assert_array_equal(loaded.values, original.values)
    np.testing.assert_array_equal(loaded.voxel_index, original.voxel_index)
    assert loaded.geometry.shape == original.geometry.shape
    assert loaded.geometry.spacing == original.geometry.spacing


@pytest.mark.unit
def test_cache_key_ignores_nothing_about_scientific_settings() -> None:
    """Same ROI settings hash equal; a bin-width change does not."""
    common = dict(
        kernel_radius=3,
        roi="tumor",
        modalities=["CT"],
        params_file=None,
        output_float32=True,
        crop_to_roi=True,
    )
    a = voxel_radiomics_cache_key(
        "s1",
        params={"setting": {"binWidth": 12.0}},
        **common,
    )
    b = voxel_radiomics_cache_key(
        "s1",
        params={"setting": {"binWidth": 12.0}},
        **common,
    )
    c = voxel_radiomics_cache_key(
        "s1",
        params={"setting": {"binWidth": 25.0}},
        **common,
    )
    assert a == b
    assert a != c


@pytest.mark.unit
def test_cache_hit_and_miss(tmp_path: Path) -> None:
    """A written archive is returned; a missing key is None."""
    field = make_field("P2", n_voxels=6)
    key = voxel_radiomics_cache_key(
        field.subject_id,
        kernel_radius=3,
        roi="tumor",
        modalities=["CT"],
        params={"setting": {"binWidth": 12.0}},
        params_file=None,
        output_float32=True,
        crop_to_roi=True,
    )
    assert load_cached_voxel_field(tmp_path, field.subject_id, key) is None
    save_cached_voxel_field(tmp_path, key, field)
    hit = load_cached_voxel_field(tmp_path, field.subject_id, key)
    assert hit is not None
    np.testing.assert_array_equal(hit.values, field.values)
