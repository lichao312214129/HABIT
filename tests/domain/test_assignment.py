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

from habit.exceptions import CompatibilityError
from habit.habitat_model.assignment import HabitatAssignerRegistry, NearestCentroidAssigner
from habit._protocols import HabitatAssigner

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


@pytest.mark.unit
def test_kmeans_assigner_matches_sklearn_predict() -> None:
    """Hard labels are ``KMeans.predict``, including on rows that were not fit."""
    from sklearn.cluster import KMeans

    from habit.habitat_model import KMeansHabitatModelFitter
    from habit.habitat_model._predict import predict_labels

    from .conftest import two_cluster_units

    units = two_cluster_units()
    matrix = np.vstack([unit.features.to_numpy(dtype=float) for unit in units])
    fitter = KMeansHabitatModelFitter(n_habitats=2, n_init=10, max_iter=50)
    fitter.set_random_state(0)
    model = fitter.fit(units)
    reference = KMeans(n_clusters=2, n_init=10, max_iter=50, random_state=0)
    reference.fit(matrix)
    np.testing.assert_allclose(model.centroids, reference.cluster_centers_)
    assert np.array_equal(predict_labels(model, matrix), reference.labels_)
    held_out = matrix + np.array([0.4, -0.2])
    assert np.array_equal(predict_labels(model, held_out), reference.predict(held_out))
    from habit.exceptions import HABITAPIError

    with pytest.raises(HABITAPIError, match="predict_proba"):
        model.assigner().predict_proba(units[0])


@pytest.mark.unit
def test_gmm_assigner_matches_sklearn_predict_and_proba() -> None:
    """GMM labels and probabilities are ``GaussianMixture.predict`` / ``predict_proba``."""
    from sklearn.mixture import GaussianMixture

    from habit.habitat_model import GmmHabitatModelFitter
    from habit.habitat_model._predict import predict_labels

    from .conftest import two_cluster_units

    units = two_cluster_units()
    matrix = np.vstack([unit.features.to_numpy(dtype=float) for unit in units])
    fitter = GmmHabitatModelFitter(
        n_habitats=2, max_iter=50, n_init=5, covariance_type="full"
    )
    fitter.set_random_state(5)
    model = fitter.fit(units)
    reference = GaussianMixture(
        n_components=2,
        random_state=5,
        covariance_type="full",
        n_init=5,
        max_iter=50,
    )
    reference.fit(matrix)
    np.testing.assert_allclose(model.centroids, reference.means_)
    np.testing.assert_allclose(model.estimator_state["weights"], reference.weights_)
    assert np.array_equal(predict_labels(model, matrix), reference.predict(matrix))
    proba = model.assigner().predict_proba(units[0])
    expected = reference.predict_proba(units[0].features.to_numpy(dtype=float))
    np.testing.assert_allclose(proba.to_numpy(), expected)
    assert list(proba.columns) == ["habitat_1", "habitat_2"]
    assert np.allclose(proba.to_numpy().sum(axis=1), 1.0)


@pytest.mark.unit
def test_gmm_without_mixture_parameters_cannot_assign() -> None:
    """A centroids-only GMM archive cannot invent ``GaussianMixture.predict``."""
    from habit.exceptions import HABITAPIError

    model = make_model()
    object.__setattr__(
        model,
        "spec_payload",
        {"habitat_model_fitter": {"name": "gmm", "params": {}}},
    )
    object.__setattr__(model, "model_id", "gmm-legacy")
    with pytest.raises(HABITAPIError, match="covariances"):
        NearestCentroidAssigner(model=model)(_units_for_assignment())


@pytest.mark.unit
def test_centroid_metrics_use_the_matching_centre(tmp_path) -> None:
    """Euclidean uses the mean, Manhattan the median, cosine the mean direction."""
    from habit.contracts import CohortFingerprint, HabitatModel
    from habit.habitat_model._predict import assign_to_centers, class_centers

    from .conftest import provenance

    matrix = np.array(
        [[0.0, 0.0], [2.0, 0.0], [10.0, 10.0], [14.0, 10.0]],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1])
    means = class_centers(matrix, labels, "euclidean")
    medians = class_centers(matrix, labels, "manhattan")
    np.testing.assert_allclose(means[0], [1.0, 0.0])
    np.testing.assert_allclose(medians[0], [1.0, 0.0])
    directions = class_centers(
        np.array([[3.0, 0.0], [0.0, 4.0], [0.0, -8.0]], dtype=np.float64),
        np.array([0, 0, 1]),
        "cosine",
    )
    # L2-normalised (3, 0) and (0, 4) average to a diagonal direction.
    np.testing.assert_allclose(directions[0], [0.5, 0.5])
    painted = assign_to_centers(matrix, means, "euclidean")
    assert np.array_equal(painted, labels)

    model = HabitatModel(
        model_id="centre-model",
        n_habitats=2,
        feature_names=("f1", "f2"),
        centroids=means,
        preprocessing_state={},
        spec_payload={"habitat_model_fitter": {"name": "consensus", "params": {}}},
        cohort_fingerprint=CohortFingerprint(
            n_subjects=1, modalities=(), subject_id_digest="ab" * 32
        ),
        provenance=provenance(),
        estimator_state={"rule": "nearest_centroid", "metric": "euclidean"},
    )
    written = model.save(tmp_path / "centres.habitatmodel")
    loaded = HabitatModel.load(written)
    assert loaded.estimator_state["metric"] == "euclidean"
    assert loaded.estimator_state["rule"] == "nearest_centroid"
