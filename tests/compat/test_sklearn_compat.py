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


def _with_chains(spec: HabitatSpec, **chains: object) -> HabitatSpec:
    """
    Return ``spec`` with preprocessing chains attached.

    Args:
        spec: Base specification.
        **chains: Chain fields to set, as tuples of ``Spec``.

    Returns:
        A new specification; the input is frozen and untouched.
    """
    import dataclasses

    return dataclasses.replace(spec, **chains)  # type: ignore[arg-type]


@pytest.mark.unit
def test_habitat_estimator_applies_the_voxel_preprocessing_chain(
    two_step_spec: HabitatSpec,
) -> None:
    """A configured voxel chain changes the fitted habitat definition.

    Without this, a spec could declare normalisation, record it in provenance,
    and produce centroids from unnormalised features.
    """
    plain = as_estimator(two_step_spec)
    plain.fit(_cohort())
    scaled = as_estimator(
        _with_chains(
            two_step_spec,
            voxel_feature_preprocessors=(Spec(name="minmax"),),
        )
    )
    scaled.fit(_cohort())
    assert not np.allclose(plain.model_.centroids, scaled.model_.centroids)
    # min-max per subject bounds every voxel feature, hence every centroid.
    assert scaled.model_.centroids.max() <= 1.0


@pytest.mark.unit
def test_habitat_estimator_fits_and_stores_the_cohort_chain(
    two_step_spec: HabitatSpec,
) -> None:
    """The cohort chain is fitted, stored in the model and reflected in its id.

    Storing the chain is what makes the model portable: centroids only mean
    something in the feature space they were fitted in.
    """
    plain = as_estimator(two_step_spec)
    plain.fit(_cohort())
    wired = as_estimator(
        _with_chains(
            two_step_spec,
            cohort_feature_preprocessors=(Spec(name="zscore"),),
        )
    )
    wired.fit(_cohort())

    state = wired.model_.preprocessing_state
    assert "cohort_feature_preprocessor" in state
    chain_state = state["cohort_feature_preprocessor"]
    # Imputation is prepended, so the stored chain has one more step than the
    # spec named -- and its states must match its own method count.
    steps = chain_state["spec"]["params"]["steps"]
    assert [step["name"] for step in steps] == ["impute", "zscore"]
    assert len(chain_state["states"]) == len(steps)
    assert (
        wired.model_.spec_payload["cohort_feature_preprocessor"]["name"]
        == "cohort_feature_preprocessor"
    )
    # Two models whose centroids live in different feature spaces are
    # different definitions and must not share an id.
    assert wired.model_.model_id != plain.model_.model_id


@pytest.mark.unit
def test_model_with_cohort_chain_survives_save_and_load(
    two_step_spec: HabitatSpec, tmp_path
) -> None:
    """A model carrying a fitted chain is savable and replays identically.

    Regression guard: the chain state used to hold ``Series`` and a fitted
    ``KBinsDiscretizer``, so ``save()`` raised on exactly the models worth
    sharing -- the ones defining habitats in a preprocessed feature space.
    """
    from habit.contracts.habitat import HabitatModel
    from habit.domain.feature_preprocessing import CohortPreprocessingChain

    spec = _with_chains(
        two_step_spec,
        cohort_feature_preprocessors=(
            Spec(name="zscore"),
            Spec(name="binning", params={"n_bins": 4}),
        ),
    )
    estimator = as_estimator(spec)
    estimator.fit(_cohort())

    restored = HabitatModel.load(estimator.model_.save(tmp_path / "m.habitatmodel"))
    assert restored.model_id == estimator.model_.model_id
    np.testing.assert_allclose(restored.centroids, estimator.model_.centroids)

    # The chain rebuilt from the loaded model must transform identically, or
    # external validation would silently use a different feature space.
    state = restored.preprocessing_state["cohort_feature_preprocessor"]
    rebuilt = CohortPreprocessingChain.from_state(state)
    original = CohortPreprocessingChain.from_state(
        estimator.model_.preprocessing_state["cohort_feature_preprocessor"]
    )
    units = estimator._components.pipeline(assigner=None).units(_cohort(1)[0])
    frame = units.feature_frame()
    np.testing.assert_array_equal(
        rebuilt.transform(frame).to_numpy(), original.transform(frame).to_numpy()
    )


@pytest.mark.unit
def test_habitat_estimator_replays_the_cohort_chain_at_transform_time(
    two_step_spec: HabitatSpec,
) -> None:
    """Prediction reuses the fitted chain, so results stay reproducible."""
    spec = _with_chains(
        two_step_spec,
        cohort_feature_preprocessors=(Spec(name="minmax"),),
    )
    estimator = as_estimator(spec)
    full = estimator.fit_transform(_cohort())
    projected = estimator.transform(_cohort(2))
    # Transforming a subset must reproduce the corresponding fit rows exactly;
    # a chain refitted on the subset would shift them.
    np.testing.assert_allclose(projected, full[:2])


@pytest.mark.unit
def test_habitat_estimator_builds_the_supervoxel_feature_extractor(
    two_step_spec: HabitatSpec,
) -> None:
    """A declared supervoxel feature extractor is constructed and applied.

    Regression guard: this slot existed in the spec and in the pipeline while
    the estimator's component factory never built it, so every run silently
    fell back to supervoxel means.
    """
    spec = _with_chains(
        two_step_spec,
        supervoxel_feature_preprocessors=(Spec(name="zscore"),),
    )
    estimator = as_estimator(spec)
    estimator.fit(_cohort())
    components = estimator._components
    assert components.supervoxel_chain is not None
    pipeline = components.pipeline(assigner=estimator._assigner)
    assert pipeline.supervoxel_feature_preprocessor is components.supervoxel_chain


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
