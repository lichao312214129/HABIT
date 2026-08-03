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
"""Tests for the raw-intensity voxel feature extractor."""

from __future__ import annotations

import numpy as np
import pytest

from habit.api.exceptions import GeometryError, HABITAPIError
from habit.contracts import ArrayImageRef, Geometry, Subject, VoxelFeatureField
from habit.domain.protocols import VoxelFeatureExtractor
from habit.domain.voxel_features import RawVoxelFeatures, VoxelFeatureExtractorRegistry

from .conftest import make_subject


@pytest.mark.unit
def test_raw_features_satisfies_protocol() -> None:
    """The built-in extractor structurally satisfies its domain protocol."""
    assert isinstance(RawVoxelFeatures(modalities=["T1"]), VoxelFeatureExtractor)


@pytest.mark.unit
def test_raw_features_single_modality_values() -> None:
    """Rows are ROI voxels; columns are the requested modality intensities."""
    subject = make_subject("P1")
    extractor = RawVoxelFeatures(modalities=["T1"])
    field = extractor(subject)
    assert isinstance(field, VoxelFeatureField)
    assert field.subject_id == "P1"
    assert field.feature_names == ("T1",)
    expected_index = np.argwhere(np.asarray(subject.mask("tumor").data) > 0)
    np.testing.assert_array_equal(field.voxel_index, expected_index)
    expected_values = np.asarray(subject.image("T1").data)[
        np.asarray(subject.mask("tumor").data) > 0
    ]
    np.testing.assert_allclose(field.values[:, 0], expected_values)
    assert "voxel_feature_extractor.raw" == field.provenance.produced_by


@pytest.mark.unit
def test_raw_features_multimodality_column_order() -> None:
    """Column order follows the requested modality order exactly."""
    subject = make_subject("P1", modalities=("T1", "T2"))
    field = RawVoxelFeatures(modalities=["T2", "T1"])(subject)
    assert field.feature_names == ("T2", "T1")
    mask = np.asarray(subject.mask("tumor").data) > 0
    np.testing.assert_allclose(field.values[:, 0], np.asarray(subject.image("T2").data)[mask])
    np.testing.assert_allclose(field.values[:, 1], np.asarray(subject.image("T1").data)[mask])


@pytest.mark.unit
def test_raw_features_missing_modality_raises_key_error() -> None:
    """A missing modality is an honest lookup failure."""
    subject = make_subject("P1")
    with pytest.raises(KeyError):
        RawVoxelFeatures(modalities=["FLAIR"])(subject)


@pytest.mark.unit
def test_raw_features_geometry_mismatch_raises() -> None:
    """Modalities on a different grid than the mask cannot be combined."""
    subject = make_subject("P1")
    mismatched = Geometry.from_array((4, 4, 4))
    subject.images["T1"] = ArrayImageRef(
        array=np.zeros((4, 4, 4)), geometry=mismatched
    )
    with pytest.raises(GeometryError):
        RawVoxelFeatures(modalities=["T1"])(subject)


@pytest.mark.unit
def test_raw_features_requires_a_modality() -> None:
    """An empty modality list is rejected at construction."""
    with pytest.raises(HABITAPIError):
        RawVoxelFeatures(modalities=[])


@pytest.mark.unit
def test_raw_features_spec_and_registry_creation() -> None:
    """Spec fingerprints are stable and the registry validates parameters."""
    first = RawVoxelFeatures(modalities=["T1"])
    second = RawVoxelFeatures(modalities=["T1"])
    assert first.spec.fingerprint() == second.spec.fingerprint()
    created = VoxelFeatureExtractorRegistry.create("raw", modalities=["T1"])
    assert isinstance(created, RawVoxelFeatures)
