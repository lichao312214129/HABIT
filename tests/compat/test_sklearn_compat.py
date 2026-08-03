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
"""Contract tests for ``habit.compat.sklearn`` (the ``*Estimator`` adapters)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from habit.api.exceptions import HABITAPIError
from habit.compat.sklearn import (
    HabitatFeaturesEstimator,
    as_classifier,
    as_estimator,
    as_transformer,
)
from habit.domain.classification.models import LogisticRegressionClassifier
from habit.domain.feature_selection.selectors import AnovaSelector
from habit.domain.table_preprocessing.methods import ZScorePreprocessor
from habit.spec import HabitatSpec, Spec
from tests.domain.conftest import make_feature_table, make_subject


def _cohort(n: int = 4):
    """Build a small synthetic cohort with two intensity clusters."""
    return [make_subject(f"P{i}", seed=i) for i in range(n)]


# ----------------------------------------------------------------------
# HabitatFeaturesEstimator
# ----------------------------------------------------------------------


@pytest.mark.unit
def test_habitat_estimator_fit_transform_layout(direct_spec: HabitatSpec) -> None:
    """fit_transform returns one row per subject in the fit-time layout."""
    estimator = as_estimator(direct_spec)
    matrix = estimator.fit_transform(_cohort())

    assert matrix.shape == (4, 4)
    names = estimator.get_feature_names_out()
    assert matrix.shape[1] == len(names)
    assert {"habitat_1_volume_fraction", "habitat_2_volume_fraction"} <= set(names)
    # The fitted habitat definition is a first-class artefact on the adapter.
    assert estimator.model_.n_habitats == 2
    assert estimator.spec_.random_seed == 0


@pytest.mark.unit
def test_habitat_estimator_transform_keeps_fit_columns(direct_spec: HabitatSpec) -> None:
    """transform on a later cohort reuses the fitted habitat definition."""
    estimator = as_estimator(direct_spec)
    estimator.fit(_cohort())

    projected = estimator.transform(_cohort(2))
    assert projected.shape == (2, 4)
    # transform equals the fit_transform rows on the same subjects: the
    # definition was learned once and is applied identically afterwards.
    full = estimator.fit_transform(_cohort())
    np.testing.assert_allclose(projected, full[:2])


@pytest.mark.unit
def test_habitat_estimator_two_step_and_overrides(two_step_spec: HabitatSpec) -> None:
    """The SLIC design runs; constructor overrides fold into the spec."""
    estimator = as_estimator(two_step_spec, n_habitats=2, n_supervoxels=6, random_seed=1)
    matrix = estimator.fit_transform(_cohort())

    assert matrix.shape == (4, 4)
    assert estimator.spec_.supervoxelizer.params["n_supervoxels"] == 6
    assert estimator.spec_.random_seed == 1
    # The user's spec object is never mutated by the overrides.
    assert two_step_spec.supervoxelizer.params["n_supervoxels"] == 8


@pytest.mark.unit
def test_habitat_estimator_guards(direct_spec: HabitatSpec) -> None:
    """Unfitted use, empty cohorts and impossible overrides fail loudly."""
    estimator = as_estimator(direct_spec)
    with pytest.raises(NotFittedError):
        estimator.transform(_cohort(1))
    with pytest.raises(HABITAPIError, match="no subjects"):
        estimator.fit([])
    with pytest.raises(HABITAPIError, match="iterable of Subject"):
        estimator.fit([object()])
    # n_supervoxels only makes sense for two-step designs.
    with pytest.raises(HABITAPIError, match="two-step"):
        as_estimator(direct_spec, n_supervoxels=10).fit(_cohort(1))
    # A spec without habitat feature families cannot produce a matrix.
    broken = HabitatSpec(
        name="broken",
        voxel_feature_extractor=direct_spec.voxel_feature_extractor,
        supervoxelizer=None,
        habitat_model_fitter=direct_spec.habitat_model_fitter,
        habitat_assigner=direct_spec.habitat_assigner,
    )
    with pytest.raises(HABITAPIError, match="habitat_features is empty"):
        as_estimator(broken).fit(_cohort(1))


@pytest.mark.unit
def test_habitat_estimator_clone_and_pipeline(direct_spec: HabitatSpec) -> None:
    """sklearn.clone drops fitted state; the adapter drives a Pipeline."""
    estimator = as_estimator(direct_spec)
    estimator.fit(_cohort())
    fresh = clone(estimator)
    assert not hasattr(fresh, "model_")

    from sklearn.linear_model import LogisticRegression

    pipe = Pipeline(
        [
            ("habitats", as_estimator(direct_spec)),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    cohort = _cohort(6)
    y = np.arange(6) % 2
    pipe.fit(cohort, y)
    predictions = pipe.predict(cohort)
    assert predictions.shape == (6,)
    # The grid-search entry point: n_habitats is a real settable parameter.
    pipe.set_params(habitats__n_habitats=3)
    assert pipe.named_steps["habitats"].n_habitats == 3


# ----------------------------------------------------------------------
# TableTransformerEstimator
# ----------------------------------------------------------------------


@pytest.mark.unit
def test_table_transformer_preprocessor_roundtrip() -> None:
    """A preprocessor keeps FeatureTable semantics through fit/transform."""
    table = make_feature_table(tuple(f"S{i:02d}" for i in range(10)))
    estimator = as_transformer(ZScorePreprocessor())
    transformed = estimator.fit_transform(table)

    assert list(transformed.feature_columns) == list(table.feature_columns)
    assert transformed.outcome_column == "y"
    # Z-scored training columns have ~zero mean and unit variance.
    matrix = transformed.feature_matrix().to_numpy()
    np.testing.assert_allclose(matrix.mean(axis=0), 0.0, atol=1e-8)
    np.testing.assert_allclose(matrix.std(axis=0), 1.0, atol=0.2)


@pytest.mark.unit
def test_table_transformer_selector_and_component_guards() -> None:
    """Supervised selectors narrow the columns; misuse fails loudly."""
    table = make_feature_table(tuple(f"S{i:02d}" for i in range(20)))
    selector = as_transformer(AnovaSelector(n_features_to_select=2))
    selected = selector.fit_transform(table)
    assert len(selected.feature_columns) == 2
    # The signal column separates the classes, so ANOVA must keep it.
    assert "signal" in selected.feature_columns

    with pytest.raises(NotFittedError):
        as_transformer(ZScorePreprocessor()).transform(table)
    with pytest.raises(HABITAPIError, match="TablePreprocessor or FeatureSelector"):
        as_transformer(LogisticRegressionClassifier()).fit(table)
    with pytest.raises(HABITAPIError, match="FeatureTable"):
        as_transformer(ZScorePreprocessor()).fit(np.zeros((3, 3)))


# ----------------------------------------------------------------------
# TableClassifierEstimator
# ----------------------------------------------------------------------


@pytest.mark.unit
def test_table_classifier_full_surface() -> None:
    """fit/predict/predict_proba/score all honour the in-table outcome."""
    table = make_feature_table(tuple(f"S{i:02d}" for i in range(30)), seed=1)
    classifier = as_classifier(LogisticRegressionClassifier(max_iter=500))
    classifier.fit(table)

    predictions = classifier.predict(table)
    assert predictions.shape == (30,)
    probabilities = classifier.predict_proba(table)
    assert probabilities.shape == (30, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-8)
    assert set(classifier.classes_) == {"0", "1"}
    # The signal column separates the classes well above chance.
    assert classifier.score(table) > 0.9


@pytest.mark.unit
def test_table_classifier_outcome_contract() -> None:
    """y fills a missing outcome, must agree with an existing one."""
    table = make_feature_table(tuple(f"S{i:02d}" for i in range(12)))
    y = table.frame["y"].to_numpy()
    no_outcome = make_feature_table(tuple(f"S{i:02d}" for i in range(12)), outcome=False)

    classifier = as_classifier(LogisticRegressionClassifier(max_iter=500))
    classifier.fit(no_outcome, y)
    assert set(classifier.classes_) == {"0", "1"}

    with pytest.raises(HABITAPIError, match="no outcome column and no y"):
        as_classifier(LogisticRegressionClassifier()).fit(no_outcome)
    with pytest.raises(HABITAPIError, match="disagrees"):
        as_classifier(LogisticRegressionClassifier()).fit(table, 1 - y)
    with pytest.raises(HABITAPIError, match="entries but the table has"):
        as_classifier(LogisticRegressionClassifier()).fit(no_outcome, y[:5])
    with pytest.raises(NotFittedError):
        as_classifier(LogisticRegressionClassifier()).predict(table)
    with pytest.raises(HABITAPIError, match="Classifier"):
        as_classifier(ZScorePreprocessor()).fit(table)


@pytest.mark.unit
def test_table_pipeline_score_uses_table_outcome() -> None:
    """Pipeline.score(table) works without an explicit y."""
    table = make_feature_table(tuple(f"S{i:02d}" for i in range(30)), seed=2)
    pipe = Pipeline(
        [
            ("scale", as_transformer(ZScorePreprocessor())),
            ("select", as_transformer(AnovaSelector(n_features_to_select=2))),
            ("model", as_classifier(LogisticRegressionClassifier(max_iter=500))),
        ]
    )
    pipe.fit(table)
    assert pipe.score(table) > 0.9
    proba = pipe.predict_proba(table)
    assert proba.shape == (30, 2)
