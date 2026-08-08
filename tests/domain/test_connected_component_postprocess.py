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
"""Tests for ConnectedComponentPostprocess and pipeline wiring."""

from __future__ import annotations

import numpy as np
import pytest

from habit.contracts import Cohort, Geometry, HabitatMap, Supervoxelization
from habit.domain import ConnectedComponentPostprocess, SubjectPipeline
from habit.domain.habitat_model import KMeansHabitatModelFitter
from habit.domain.pipeline import voxel_units
from habit.domain.supervoxel_features import aggregate_voxel_means
from habit.domain.voxel_features import RawVoxelFeatures
from habit.spec.specs import Spec

from .conftest import make_field, make_subject, provenance


@pytest.mark.unit
def test_apply_to_habitat_map_cleans_and_updates_provenance() -> None:
    """Habitat cleanup reassigns tiny islands and chains provenance."""
    labels = np.zeros((4, 4, 4), dtype=np.int32)
    labels[1:4, 1:4, 1:4] = 1
    labels[0, 0, 0] = 2
    habitat_map = HabitatMap(
        subject_id="S0",
        label_array=labels,
        geometry=Geometry.from_array((4, 4, 4)),
        model_id="m",
        habitat_ids=(1, 2),
        provenance=provenance(),
    )
    op = ConnectedComponentPostprocess(min_component_size=5, connectivity=1)
    cleaned = op.apply_to_habitat_map(habitat_map)
    assert cleaned.label_array[0, 0, 0] == 1
    assert cleaned.provenance.produced_by == "postprocess.connected_components"
    assert cleaned.habitat_ids == (1, 2)
    assert int(np.count_nonzero(cleaned.label_array > 0)) == int(
        np.count_nonzero(labels > 0)
    )


@pytest.mark.unit
def test_apply_to_supervoxelization_realigns_features() -> None:
    """Supervoxel cleanup rebuilds means so features match surviving labels."""
    field = make_field("P1", n_voxels=20)
    labels = np.zeros(tuple(int(v) for v in field.geometry.shape), dtype=np.int32)
    labels[tuple(field.voxel_index.T)] = 1
    island_idx = tuple(int(v) for v in field.voxel_index[0])
    labels[island_idx] = 2
    units = Supervoxelization(
        subject_id=field.subject_id,
        label_array=labels,
        features=aggregate_voxel_means(field, labels),
        geometry=field.geometry,
        provenance=field.provenance,
    )
    op = ConnectedComponentPostprocess(min_component_size=5, connectivity=1)
    cleaned = op.apply_to_supervoxelization(units, field)
    surviving = set(np.unique(cleaned.label_array)) - {0}
    assert surviving == {1}
    assert set(cleaned.features.index.astype(int)) == {1}


@pytest.mark.unit
def test_pipeline_postprocess_habitat_updates_provenance() -> None:
    """SubjectPipeline applies habitat cleanup after assignment."""
    voxel_features = RawVoxelFeatures(modalities=["T1"])
    subjects = [make_subject(f"S{i}", seed=i) for i in range(3)]
    units = [voxel_units(voxel_features(subject)) for subject in subjects]
    fitter = KMeansHabitatModelFitter(n_habitats=2, n_init=5)
    fitter.set_random_state(11)
    model = fitter.fit(units, cohort=Cohort(subjects))
    pipeline = SubjectPipeline(
        voxel_features,
        None,
        model.assigner(),
        postprocess_habitat=ConnectedComponentPostprocess(
            min_component_size=2, connectivity=1
        ),
    )
    habitat_map = pipeline(make_subject("new", seed=42))
    assert isinstance(habitat_map, HabitatMap)
    assert habitat_map.provenance.produced_by == "postprocess.connected_components"
    assert Spec.from_dict(pipeline.spec.params["postprocess_habitat"]).name == (
        "connected_components"
    )
