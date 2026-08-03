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
"""Tests for nearest-centroid habitat assignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from habit.api.exceptions import CompatibilityError
from habit.domain.assignment import HabitatAssignerRegistry, NearestCentroidAssigner
from habit.domain.protocols import HabitatAssigner

from .conftest import make_model, make_supervoxelization


def _units_for_assignment():
    """Two supervoxels: one near the origin centroid, one near (10, 10)."""
    features = pd.DataFrame(
        [[0.2, 0.1], [9.8, 10.1]], columns=["f1", "f2"], index=[1, 2]
    )
    features.index.name = "supervoxel"
    return make_supervoxelization("P1", features)


@pytest.mark.unit
def test_assigner_satisfies_protocol() -> None:
    """The built-in assigner structurally satisfies its protocol."""
    assigner = NearestCentroidAssigner(model=make_model())
    assert isinstance(assigner, HabitatAssigner)
    assert assigner.model.model_id == "test-model"


@pytest.mark.unit
def test_nearest_centroid_assignment() -> None:
    """Each supervoxel takes the habitat of its closest centroid; 0 stays 0."""
    model = make_model()
    assigner = NearestCentroidAssigner(model=model)
    habitat_map = assigner(_units_for_assignment())
    labels = np.asarray(habitat_map.label_array)
    flat = labels.ravel()
    assert flat[0] == 1  # supervoxel 1 sits by the origin centroid
    assert flat[1] == 2  # supervoxel 2 sits by the (10, 10) centroid
    assert flat[2] == 0  # background remains background
    assert habitat_map.model_id == "test-model"
    assert habitat_map.habitat_ids == (1, 2)
    assert habitat_map.provenance.produced_by == "habitat_assigner.nearest_centroid"


@pytest.mark.unit
def test_assigner_spec_binds_model_identity() -> None:
    """The spec fingerprint changes with the bound model, guarding caches."""
    first = NearestCentroidAssigner(model=make_model())
    other_model = make_model()
    object.__setattr__(other_model, "model_id", "other-model")
    second = NearestCentroidAssigner(model=other_model)
    assert first.spec.params["model_id"] == "test-model"
    assert first.spec.fingerprint() != second.spec.fingerprint()


@pytest.mark.unit
def test_assigner_rejects_missing_features() -> None:
    """A unit lacking a model-required feature is a CompatibilityError."""
    features = pd.DataFrame([[0.2, 0.1]], columns=["f1", "unrelated"], index=[1])
    unit = make_supervoxelization("P1", features)
    with pytest.raises(CompatibilityError):
        NearestCentroidAssigner(model=make_model())(unit)


@pytest.mark.unit
def test_assigner_rejects_labels_without_feature_rows() -> None:
    """A label present in the array but absent from the features fails loudly."""
    features = pd.DataFrame([[0.2, 0.1]], columns=["f1", "f2"], index=[1])
    unit = make_supervoxelization("P1", features)
    unit.label_array.ravel()[3] = 7  # label 7 has no feature row
    with pytest.raises(CompatibilityError):
        NearestCentroidAssigner(model=make_model())(unit)


@pytest.mark.unit
def test_assigner_ignores_extra_feature_columns() -> None:
    """Richer feature frames stay assignable; column order follows the model."""
    features = pd.DataFrame(
        [[99.0, 0.2, 0.1]], columns=["extra", "f1", "f2"], index=[1]
    )
    unit = make_supervoxelization("P1", features)
    habitat_map = NearestCentroidAssigner(model=make_model())(unit)
    assert np.asarray(habitat_map.label_array).ravel()[0] == 1


@pytest.mark.unit
def test_model_assigner_factory_roundtrip() -> None:
    """``HabitatModel.assigner()`` builds a working bound assigner."""
    model = make_model()
    assigner = model.assigner()
    assert isinstance(assigner, NearestCentroidAssigner)
    assert assigner.model is model
    habitat_map = assigner(_units_for_assignment())
    assert habitat_map.model_id == model.model_id


@pytest.mark.unit
def test_assigner_registry_creation() -> None:
    """The registry constructs assigners with the model passed through."""
    model = make_model()
    assigner = HabitatAssignerRegistry.create("nearest_centroid", model=model)
    assert isinstance(assigner, NearestCentroidAssigner)
    assert HabitatAssignerRegistry.available() == ("nearest_centroid",)


@pytest.mark.unit
def test_assigner_requires_a_fitted_model() -> None:
    """Predicting without a fitted model is unrepresentable."""
    with pytest.raises(CompatibilityError):
        NearestCentroidAssigner(model=None)  # type: ignore[arg-type]
