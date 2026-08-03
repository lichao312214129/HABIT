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
"""End-to-end tests for SubjectPipeline on fully synthetic arrays."""

from __future__ import annotations

import numpy as np
import pytest

from habit.api.exceptions import HABITAPIError
from habit.contracts import Cohort, HabitatMap
from habit.domain import (
    HabitatVolumeFeatures,
    IthHabitatFeatures,
    KMeansHabitatModelFitter,
    MsiHabitatFeatures,
    RawVoxelFeatures,
    SlicSupervoxelizer,
    SubjectPipeline,
)
from habit.domain.pipeline import _voxel_units

from .conftest import make_field, make_subject


def _fitted_pipeline(*, supervoxels: bool = True, seed: int = 11) -> SubjectPipeline:
    """Fit a two-habitat model on synthetic subjects and build the pipeline."""
    voxel_features = RawVoxelFeatures(modalities=["T1"])
    supervoxelizer = SlicSupervoxelizer(n_supervoxels=8) if supervoxels else None
    cohort = Cohort([make_subject(f"S{i}", seed=i) for i in range(3)])
    if supervoxelizer is None:
        units = [_voxel_units(voxel_features(subject)) for subject in cohort]
    else:
        units = [supervoxelizer(voxel_features(subject)) for subject in cohort]
    fitter = KMeansHabitatModelFitter(n_habitats=2, n_init=5)
    fitter.set_random_state(seed)
    model = fitter.fit(units, cohort=cohort)
    return SubjectPipeline(voxel_features, supervoxelizer, model.assigner())


@pytest.mark.unit
def test_pipeline_labels_unseen_subject() -> None:
    """The composed chain labels a subject end to end."""
    pipeline = _fitted_pipeline()
    habitat_map = pipeline(make_subject("new", seed=42))
    assert isinstance(habitat_map, HabitatMap)
    assert habitat_map.subject_id == "new"
    assert habitat_map.model_id == pipeline.habitat_assigner.model.model_id
    labels = set(np.unique(np.asarray(habitat_map.label_array)))
    assert labels <= {0, 1, 2}
    assert labels - {0}  # at least one habitat present


@pytest.mark.unit
def test_pipeline_is_deterministic_for_fixed_seed() -> None:
    """A fitted pipeline assigns identical labels on repeated calls."""
    pipeline = _fitted_pipeline()
    subject = make_subject("new", seed=5)
    first = pipeline(subject)
    second = pipeline(subject)
    np.testing.assert_array_equal(
        np.asarray(first.label_array), np.asarray(second.label_array)
    )


@pytest.mark.unit
def test_pipeline_without_supervoxelizer_clusters_voxels_directly() -> None:
    """``supervoxelizer=None`` selects the direct-clustering designs."""
    pipeline = _fitted_pipeline(supervoxels=False)
    habitat_map = pipeline(make_subject("new", seed=9))
    labels = set(np.unique(np.asarray(habitat_map.label_array)))
    assert labels <= {0, 1, 2}
    assert labels - {0}


@pytest.mark.unit
def test_voxel_units_wrap_field_as_singleton_partition() -> None:
    """Each voxel becomes its own clustering unit, preserving order."""
    field = make_field("P1", n_voxels=6)
    units = _voxel_units(field)
    labels = np.asarray(units.label_array)
    np.testing.assert_array_equal(
        labels[tuple(field.voxel_index.T)], np.arange(1, 7)
    )
    assert units.features.shape == (6, 2)
    np.testing.assert_allclose(units.features.to_numpy(), field.values)


@pytest.mark.unit
def test_pipeline_extract_features_joins_families() -> None:
    """extract_features returns one row joined across all families."""
    pipeline = _fitted_pipeline()
    table = pipeline.extract_features(
        make_subject("new", seed=3),
        [MsiHabitatFeatures(), IthHabitatFeatures(), HabitatVolumeFeatures()],
    )
    assert table.frame.shape[0] == 1
    assert table.id_columns == ("subject",)
    assert "ith_score" in table.feature_columns
    assert "contrast" in table.feature_columns
    assert "habitat_1_volume_fraction" in table.feature_columns
    assert table.frame.iloc[0]["subject"] == "new"


@pytest.mark.unit
def test_pipeline_extract_features_requires_extractors() -> None:
    """An empty extractor list is an explicit error."""
    pipeline = _fitted_pipeline()
    with pytest.raises(HABITAPIError):
        pipeline.extract_features(make_subject("new", seed=3), [])


@pytest.mark.unit
def test_pipeline_spec_covers_every_stage() -> None:
    """The composed fingerprint changes when any stage changes."""
    base = _fitted_pipeline()
    tweaked = SubjectPipeline(
        RawVoxelFeatures(modalities=["T1"]),
        SlicSupervoxelizer(n_supervoxels=16),
        base.habitat_assigner,
    )
    assert base.spec.name == "subject_pipeline"
    assert base.spec.fingerprint() != tweaked.spec.fingerprint()


@pytest.mark.unit
def test_pipeline_requires_core_steps() -> None:
    """Only the supervoxelizer may be omitted."""
    with pytest.raises(HABITAPIError):
        SubjectPipeline(None, None, None)  # type: ignore[arg-type]


@pytest.mark.unit
def test_pipeline_runs_under_cohort_map() -> None:
    """The pipeline is an ordinary subject-level operator for Cohort.map."""
    pipeline = _fitted_pipeline()
    cohort = Cohort([make_subject(f"E{i}", seed=20 + i) for i in range(2)])
    maps = cohort.map(pipeline)
    assert [m.subject_id for m in maps] == ["E0", "E1"]
    assert all(isinstance(m, HabitatMap) for m in maps)
