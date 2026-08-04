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
"""Tests for the supervoxel feature extractor domain."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from habit.api.exceptions import HABITAPIError
from habit.domain.protocols import SupervoxelFeatureExtractor
from habit.domain.supervoxel import (
    GmmSupervoxelizer,
    KMeansSupervoxelizer,
    SlicSupervoxelizer,
    SupervoxelizerRegistry,
)
from habit.domain.supervoxel_features import (
    MeanVoxelFeatures,
    SupervoxelFeatureExtractorRegistry,
    SupervoxelRadiomicsFeatures,
)

from .conftest import make_field, make_subject


@pytest.mark.unit
@pytest.mark.parametrize("name", ["mean_voxel_features", "supervoxel_radiomics"])
def test_builtin_extractors_satisfy_protocol(name: str) -> None:
    """Every registered extractor structurally satisfies its protocol."""
    extractor = SupervoxelFeatureExtractorRegistry.create(name)
    assert isinstance(extractor, SupervoxelFeatureExtractor)
    assert extractor.spec.name == name


@pytest.mark.unit
def test_mean_extractor_reproduces_the_supervoxelizer_default() -> None:
    """Naming the default explicitly must not change any number.

    The supervoxelizers attach feature means themselves; running
    ``mean_voxel_features`` over the same partition has to be a no-op, which
    is what makes the default safe to leave unstated in a config.
    """
    field = make_field("P1", n_voxels=16)
    subject = make_subject("P1")
    partition = SlicSupervoxelizer(n_supervoxels=4)(field)

    described = MeanVoxelFeatures(field=field)(subject, partition)

    pd.testing.assert_frame_equal(described.features, partition.features)
    assert np.array_equal(described.label_array, partition.label_array)


@pytest.mark.unit
def test_mean_extractor_without_field_is_idempotent() -> None:
    """Constructed without voxel features, the extractor passes them through."""
    field = make_field("P1", n_voxels=16)
    subject = make_subject("P1")
    partition = SlicSupervoxelizer(n_supervoxels=4)(field)

    described = MeanVoxelFeatures()(subject, partition)

    pd.testing.assert_frame_equal(described.features, partition.features)


@pytest.mark.unit
def test_extractor_records_itself_in_provenance() -> None:
    """The record names the extractor, not just the supervoxelizer."""
    field = make_field("P1", n_voxels=16)
    subject = make_subject("P1")
    partition = SlicSupervoxelizer(n_supervoxels=4)(field)

    described = MeanVoxelFeatures(field=field)(subject, partition)

    assert (
        described.provenance.produced_by
        == "supervoxel_feature_extractor.mean_voxel_features"
    )


@pytest.mark.unit
def test_extractor_never_redraws_the_partition() -> None:
    """An extractor describes regions; it must not move their boundaries."""
    field = make_field("P1", n_voxels=16)
    subject = make_subject("P1")
    partition = KMeansSupervoxelizer(n_supervoxels=3)(field)

    described = MeanVoxelFeatures(field=field)(subject, partition)

    assert np.array_equal(described.label_array, partition.label_array)
    assert described.geometry == partition.geometry
    assert described.subject_id == partition.subject_id


@pytest.mark.unit
def test_radiomics_rejects_conflicting_parameter_sources() -> None:
    """params_file and params are mutually exclusive, as elsewhere in v1."""
    with pytest.raises(HABITAPIError):
        SupervoxelRadiomicsFeatures(params_file="p.yaml", params={"binWidth": 25})


@pytest.mark.unit
def test_radiomics_rejects_absent_modality() -> None:
    """A misspelt modality fails at the call site with the available names."""
    field = make_field("P1", n_voxels=16)
    subject = make_subject("P1", modalities=("T1",))
    partition = SlicSupervoxelizer(n_supervoxels=4)(field)

    extractor = SupervoxelRadiomicsFeatures(modalities=("T2",))

    with pytest.raises(HABITAPIError, match="T2"):
        extractor(subject, partition)


@pytest.mark.unit
def test_radiomics_spec_carries_every_constructor_parameter() -> None:
    """The spec must be able to rebuild the component (YAML isomorphism)."""
    extractor = SupervoxelRadiomicsFeatures(
        modalities=("T1",), supervoxel_batch=8, use_torch_radiomics=False
    )
    rebuilt = SupervoxelFeatureExtractorRegistry.create(
        extractor.spec.name, **extractor.spec.params
    )
    assert rebuilt.spec.fingerprint() == extractor.spec.fingerprint()


# ---------------------------------------------------------------------------
# Feature-space supervoxelizers (kmeans / gmm)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("name", ["slic", "kmeans", "gmm"])
def test_every_v0_supervoxel_algorithm_is_registered(name: str) -> None:
    """v0.1 offered kmeans, gmm and slic; all three must be constructible."""
    assert name in SupervoxelizerRegistry.available()
    unit = SupervoxelizerRegistry.create(name, n_supervoxels=2)(
        make_field("P1", n_voxels=16)
    )
    assert unit.features.shape[0] >= 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "supervoxelizer", [KMeansSupervoxelizer(n_supervoxels=2), GmmSupervoxelizer(n_supervoxels=2)]
)
def test_feature_space_partitions_are_one_based_inside_the_roi(
    supervoxelizer: object,
) -> None:
    """Zero is reserved for background, so ROI labels start at one."""
    field = make_field("P1", n_voxels=16)
    unit = supervoxelizer(field)
    labels = np.asarray(unit.label_array)
    inside = labels[tuple(field.voxel_index.T)]
    assert inside.min() >= 1
    outside = labels.copy()
    outside[tuple(field.voxel_index.T)] = 0
    assert outside.max() == 0


@pytest.mark.unit
def test_feature_space_supervoxelizers_are_seedable() -> None:
    """Seeding is explicit, never a constructor parameter (v1 naming rule)."""
    field = make_field("P1", n_voxels=16, two_blobs=False)
    first = KMeansSupervoxelizer(n_supervoxels=3)
    first.set_random_state(7)
    second = KMeansSupervoxelizer(n_supervoxels=3)
    second.set_random_state(7)
    assert np.array_equal(first(field).label_array, second(field).label_array)
