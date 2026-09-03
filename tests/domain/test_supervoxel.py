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
"""Tests for the SLIC supervoxelizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from habit.api.exceptions import ConfigurationError, HABITAPIError
from habit._protocols import Supervoxelizer
from habit.supervoxel import SlicSupervoxelizer, SupervoxelizerRegistry

from .conftest import make_field


@pytest.mark.unit
def test_slic_satisfies_protocol() -> None:
    """The built-in supervoxelizer structurally satisfies its protocol."""
    assert isinstance(SlicSupervoxelizer(n_supervoxels=2), Supervoxelizer)


@pytest.mark.unit
def test_slic_partition_invariants() -> None:
    """Labels live on the field grid; zero marks voxels outside the ROI."""
    field = make_field("P1", n_voxels=16)
    unit = SlicSupervoxelizer(n_supervoxels=4)(field)
    assert unit.subject_id == "P1"
    assert unit.label_array.shape == tuple(field.geometry.shape)
    labels = np.asarray(unit.label_array)
    covered = labels[tuple(field.voxel_index.T)]
    assert covered.min() >= 1
    outside = labels.copy()
    outside[tuple(field.voxel_index.T)] = 0
    assert outside.max() == 0
    assert "supervoxelizer.slic" == unit.provenance.produced_by


@pytest.mark.unit
def test_slic_features_are_per_supervoxel_means() -> None:
    """Each feature row is the mean of its supervoxel's member voxels."""
    field = make_field("P1", n_voxels=16)
    unit = SlicSupervoxelizer(n_supervoxels=4)(field)
    assert tuple(unit.features.columns) == field.feature_names
    voxel_labels = np.asarray(unit.label_array)[tuple(field.voxel_index.T)]
    expected = (
        pd.DataFrame(field.values, columns=list(field.feature_names))
        .groupby(voxel_labels.astype(np.int64))
        .mean()
    )
    pd.testing.assert_frame_equal(
        unit.features.sort_index(), expected.sort_index(), check_names=False
    )
    assert unit.features.index.name == "supervoxel"


@pytest.mark.unit
def test_slic_n_supervoxels_clamped_to_voxel_count() -> None:
    """Requesting more supervoxels than voxels cannot fail."""
    field = make_field("P1", n_voxels=4)
    unit = SlicSupervoxelizer(n_supervoxels=100)(field)
    assert unit.features.shape[0] >= 1
    assert unit.features.shape[0] <= 4


@pytest.mark.unit
def test_slic_construction_and_registry_validation() -> None:
    """Constructor and registry reject non-positive supervoxel counts."""
    with pytest.raises(HABITAPIError):
        SlicSupervoxelizer(n_supervoxels=0)
    with pytest.raises(ConfigurationError):
        SupervoxelizerRegistry.create("slic", n_supervoxels=-3)
    created = SupervoxelizerRegistry.create("slic", n_supervoxels=2)
    assert isinstance(created, SlicSupervoxelizer)


@pytest.mark.unit
def test_slic_spec_fingerprint_tracks_params() -> None:
    """The spec fingerprint changes with algorithm parameters."""
    assert (
        SlicSupervoxelizer(n_supervoxels=2).spec.fingerprint()
        != SlicSupervoxelizer(n_supervoxels=3).spec.fingerprint()
    )
