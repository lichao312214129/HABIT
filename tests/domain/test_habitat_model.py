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
"""Tests for the cohort-level habitat model fitters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from habit.api.exceptions import CompatibilityError, HABITAPIError
from habit.contracts import HabitatModel
from habit.domain.habitat_model import (
    GmmHabitatModelFitter,
    HabitatModelFitterRegistry,
    KMeansHabitatModelFitter,
)
from habit.domain.habitat_model._base import fingerprint_from_units, pool_supervoxel_features
from habit.domain.protocols import HabitatModelFitter, Seedable

from .conftest import make_supervoxelization, two_cluster_units


@pytest.mark.unit
def test_fitters_satisfy_protocols() -> None:
    """Both built-in fitters are HabitatModelFitter and Seedable."""
    kmeans = KMeansHabitatModelFitter(n_habitats=2)
    gmm = GmmHabitatModelFitter(n_habitats=2)
    assert isinstance(kmeans, HabitatModelFitter)
    assert isinstance(kmeans, Seedable)
    assert isinstance(gmm, HabitatModelFitter)
    assert isinstance(gmm, Seedable)


@pytest.mark.unit
def test_kmeans_fit_recovers_two_blobs() -> None:
    """Two well-separated blobs yield a two-habitat self-describing model."""
    units = two_cluster_units()
    fitter = KMeansHabitatModelFitter(n_habitats=2, n_init=10)
    fitter.set_random_state(13)
    model = fitter.fit(units)
    assert isinstance(model, HabitatModel)
    assert model.n_habitats == 2
    assert model.feature_names == ("f1", "f2")
    # One centroid near the origin blob, one near the (10, 10) blob.
    low = model.centroids.min(axis=0)
    high = model.centroids.max(axis=0)
    assert np.all(low < 2.0)
    assert np.all(high > 8.0)
    assert model.model_id.startswith("kmeans-")
    assert model.provenance.produced_by == "habitat_model_fitter.kmeans"
    assert model.provenance.random_seed == 13
    assert model.cohort_fingerprint.n_subjects == 3


@pytest.mark.unit
def test_kmeans_fit_is_seed_reproducible() -> None:
    """The same seed produces byte-identical centroids across fits."""
    units = two_cluster_units()
    first = KMeansHabitatModelFitter(n_habitats=2, n_init=10)
    second = KMeansHabitatModelFitter(n_habitats=2, n_init=10)
    first.set_random_state(99)
    second.set_random_state(99)
    model_a = first.fit(units)
    model_b = second.fit(units)
    np.testing.assert_array_equal(model_a.centroids, model_b.centroids)
    assert model_a.model_id == model_b.model_id


@pytest.mark.unit
def test_kmeans_selects_n_habitats_by_validation_score() -> None:
    """Automatic selection returns a count inside the declared search range."""
    units = two_cluster_units(supervoxels_per_subject=6)
    fitter = KMeansHabitatModelFitter(
        n_habitats=None, min_habitats=2, max_habitats=3, n_init=5
    )
    fitter.set_random_state(3)
    model = fitter.fit(units)
    assert 2 <= model.n_habitats <= 3


@pytest.mark.unit
def test_kmeans_elbow_selects_in_range_and_is_seed_reproducible() -> None:
    """The elbow path runs through fit() and reproduces under a fixed seed."""
    units = two_cluster_units(supervoxels_per_subject=8)
    selections = []
    for _ in range(2):
        fitter = KMeansHabitatModelFitter(
            n_habitats=None,
            min_habitats=2,
            max_habitats=5,
            validation="elbow",
            n_init=10,
        )
        fitter.set_random_state(11)
        selections.append(fitter.fit(units).n_habitats)
    assert selections[0] == selections[1]
    assert 2 <= selections[0] <= 5


@pytest.mark.unit
def test_kmeans_kneedle_alias_uses_same_curve() -> None:
    """``kneedle`` selects from the same inertia curve as ``elbow``."""
    units = two_cluster_units(supervoxels_per_subject=8)
    selections = {}
    for method in ("elbow", "kneedle"):
        fitter = KMeansHabitatModelFitter(
            n_habitats=None,
            min_habitats=2,
            max_habitats=5,
            validation=method,
            n_init=10,
        )
        fitter.set_random_state(11)
        selections[method] = fitter.fit(units).n_habitats
    assert selections["elbow"] == selections["kneedle"]


@pytest.mark.unit
def test_kmeans_votes_across_several_validation_criteria() -> None:
    """
    A list of criteria is scored jointly and each casts one vote.

    Reaching the v0.1 multi-method behaviour from the v1.0 API; the report
    must record every criterion so the vote stays auditable.
    """
    units = two_cluster_units(supervoxels_per_subject=8)
    fitter = KMeansHabitatModelFitter(
        n_habitats=None,
        min_habitats=2,
        max_habitats=5,
        validation=["silhouette", "calinski_harabasz", "kneedle"],
        n_init=10,
    )
    fitter.set_random_state(7)
    model = fitter.fit(units)

    report = model.preprocessing_state["selection_report"]
    assert report["methods"] == ["silhouette", "calinski_harabasz", "kneedle"]
    assert report["directions"]["kneedle"] == "knee"
    assert all(len(curve) == 4 for curve in report["scores"].values())
    assert report["selected"] == model.n_habitats
    assert model.preprocessing_state["validation"] == [
        "silhouette",
        "calinski_harabasz",
        "kneedle",
    ]


@pytest.mark.unit
def test_kmeans_gap_criterion_is_available_and_reproducible() -> None:
    """The gap statistic is selectable and deterministic across fits."""
    units = two_cluster_units(supervoxels_per_subject=8)
    selections = []
    for _ in range(2):
        fitter = KMeansHabitatModelFitter(
            n_habitats=None,
            min_habitats=2,
            max_habitats=4,
            validation="gap",
            n_init=10,
        )
        fitter.set_random_state(4)
        model = fitter.fit(units)
        selections.append(model.n_habitats)
        assert model.preprocessing_state["selection_report"]["directions"] == {
            "gap": "maximize"
        }
    assert selections[0] == selections[1]


@pytest.mark.unit
def test_gmm_supports_structure_based_criteria() -> None:
    """GMM selection accepts silhouette alongside the information criteria."""
    units = two_cluster_units(supervoxels_per_subject=8)
    fitter = GmmHabitatModelFitter(
        n_habitats=None,
        min_habitats=2,
        max_habitats=4,
        validation=["bic", "silhouette"],
        max_iter=50,
    )
    fitter.set_random_state(2)
    model = fitter.fit(units)
    report = model.preprocessing_state["selection_report"]
    assert report["methods"] == ["bic", "silhouette"]
    assert report["directions"] == {"bic": "minimize", "silhouette": "maximize"}
    assert report["selected"] == model.n_habitats


@pytest.mark.unit
def test_kmeans_selection_uses_the_shared_kernel_rule() -> None:
    """
    The fitter delegates knee detection to the shared selection kernel.

    Guards the v1.0 convergence: the fitter must not carry a private rule
    that could pick a different habitat count than the v0.1 path.
    """
    from habit.kernels.cluster_selection import knee_index

    units = two_cluster_units(supervoxels_per_subject=8)
    fitter = KMeansHabitatModelFitter(
        n_habitats=None,
        min_habitats=2,
        max_habitats=5,
        validation="elbow",
        n_init=10,
    )
    fitter.set_random_state(11)
    model = fitter.fit(units)

    report = model.preprocessing_state["selection_report"]
    curve = report["scores"]["elbow"]
    assert report["candidates"] == [2, 3, 4, 5]
    assert report["directions"] == {"elbow": "knee"}
    assert report["selected"] == model.n_habitats
    assert report["candidates"][knee_index(curve)] == model.n_habitats


@pytest.mark.unit
def test_gmm_fit_recovers_two_blobs() -> None:
    """The GMM fitter stores mixture means as centroids."""
    units = two_cluster_units()
    fitter = GmmHabitatModelFitter(n_habitats=2, max_iter=50)
    fitter.set_random_state(5)
    model = fitter.fit(units)
    assert model.n_habitats == 2
    assert model.model_id.startswith("gmm-")
    assert model.preprocessing_state["covariance_type"] == "full"
    low = model.centroids.min(axis=0)
    high = model.centroids.max(axis=0)
    assert np.all(low < 3.0)
    assert np.all(high > 7.0)


@pytest.mark.unit
def test_fitter_validation_errors() -> None:
    """Misconfigured fitters fail at construction with a clear error."""
    with pytest.raises(HABITAPIError):
        KMeansHabitatModelFitter(validation="adjusted_rand")
    with pytest.raises(HABITAPIError):
        KMeansHabitatModelFitter(n_habitats=None, min_habitats=5, max_habitats=5)
    with pytest.raises(HABITAPIError):
        GmmHabitatModelFitter(validation="inertia")
    with pytest.raises(HABITAPIError):
        GmmHabitatModelFitter(covariance_type="diagonal")
    # An empty criterion list must fail rather than fall back to a default.
    with pytest.raises(HABITAPIError):
        KMeansHabitatModelFitter(validation=[])
    # One unsupported name spoils the whole vote.
    with pytest.raises(HABITAPIError):
        KMeansHabitatModelFitter(validation=["silhouette", "adjusted_rand"])


@pytest.mark.unit
def test_fit_requires_units() -> None:
    """Fitting without units is an explicit error."""
    fitter = KMeansHabitatModelFitter(n_habitats=2)
    with pytest.raises(HABITAPIError):
        fitter.fit([])


@pytest.mark.unit
def test_fit_rejects_incompatible_feature_columns() -> None:
    """A subject with different feature columns cannot enter the pool."""
    units = list(two_cluster_units())
    odd_features = pd.DataFrame(
        np.zeros((2, 2)), columns=["other_a", "other_b"], index=[1, 2]
    )
    units.append(make_supervoxelization("P4", odd_features))
    fitter = KMeansHabitatModelFitter(n_habitats=2)
    with pytest.raises(CompatibilityError):
        fitter.fit(units)


@pytest.mark.unit
def test_pool_supervoxel_features_preserves_order() -> None:
    """Pooling concatenates rows in the given unit order, never sorting."""
    units = two_cluster_units(subject_ids=("A", "B"), supervoxels_per_subject=2)
    matrix, names = pool_supervoxel_features(units)
    assert names == ("f1", "f2")
    expected = np.vstack([unit.features.to_numpy() for unit in units])
    np.testing.assert_allclose(matrix, expected)


@pytest.mark.unit
def test_fingerprint_from_units_is_order_sensitive() -> None:
    """The digest proves both the subject set and its order."""
    units = two_cluster_units(subject_ids=("A", "B"))
    reordered = (units[1], units[0])
    assert fingerprint_from_units(units).subject_id_digest != (
        fingerprint_from_units(reordered).subject_id_digest
    )


@pytest.mark.unit
def test_fit_with_cohort_records_cohort_fingerprint() -> None:
    """Passing the cohort swaps the unit-derived digest for the cohort one."""
    from habit.contracts import Cohort

    from .conftest import make_subject

    cohort = Cohort([make_subject("S1"), make_subject("S2")], name="train")
    units = two_cluster_units(subject_ids=("S1", "S2"))
    fitter = KMeansHabitatModelFitter(n_habitats=2, n_init=5)
    fitter.set_random_state(1)
    model = fitter.fit(units, cohort=cohort)
    assert model.cohort_fingerprint.name == "train"
    assert model.cohort_fingerprint.subject_id_digest == (
        cohort.summarize().subject_id_digest
    )


@pytest.mark.unit
def test_registry_creates_fitters_with_validation() -> None:
    """Both fitters are constructible by name with schema coercion."""
    kmeans = HabitatModelFitterRegistry.create("kmeans", n_habitats=2, n_init="5")
    assert isinstance(kmeans, KMeansHabitatModelFitter)
    assert kmeans.n_init == 5
    gmm = HabitatModelFitterRegistry.create("gmm", n_habitats=2)
    assert isinstance(gmm, GmmHabitatModelFitter)
    assert set(HabitatModelFitterRegistry.available()) == {"gmm", "kmeans"}


@pytest.mark.unit
def test_fitted_model_save_load_roundtrip(tmp_path) -> None:
    """A fitted model persists through the versioned artefact format."""
    units = two_cluster_units()
    fitter = KMeansHabitatModelFitter(n_habitats=2, n_init=5)
    fitter.set_random_state(2)
    model = fitter.fit(units)
    path = model.save(tmp_path / "model.habitatmodel")
    loaded = HabitatModel.load(path)
    assert loaded.model_id == model.model_id
    np.testing.assert_array_equal(loaded.centroids, model.centroids)
    assert loaded.feature_names == model.feature_names
