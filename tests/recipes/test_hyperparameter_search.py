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
"""Contract tests for hyperparameter search and nested cross-validation.

The whole file runs on the deterministic synthetic feature table, so it
finishes in seconds and never touches demo data.

What is being pinned down here is not "does GridSearchCV work" -- that is
scikit-learn's own contract -- but the two things the recipe adds on top:
the winning parameters land back in the ``MLSpec`` (so the provenance chain
survives tuning), and a nested run tunes strictly inside each outer fold's
training rows.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

import pytest

from habit.api.exceptions import HABITAPIError
from habit.datasets.synthetic import make_synthetic_feature_table
from habit.recipes.modeling import (
    CVResult,
    SearchResult,
    cross_validate,
    search_hyperparameters,
)
from habit.spec.specs import MLSpec, Spec


def _steps_spec(**overrides: Any) -> MLSpec:
    """Build a modelling spec in the single ordered ``steps`` layout."""
    fields: Dict[str, Any] = {
        "name": "search_demo",
        "classifier": Spec(name="LogisticRegression", params={"max_iter": 500}),
        "steps": (
            Spec(name="variance", params={"threshold": 0.0}),
            Spec(name="zscore"),
        ),
        "metrics": (Spec(name="accuracy"), Spec(name="auc")),
        "random_seed": 0,
    }
    fields.update(overrides)
    return MLSpec(**fields)


def _legacy_spec() -> MLSpec:
    """Build the same pipeline through the DEPRECATED three-chain layout."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return MLSpec(
            name="search_demo_legacy",
            classifier=Spec(name="LogisticRegression", params={"max_iter": 500}),
            pre_preprocessing_feature_selectors=(
                Spec(name="variance", params={"threshold": 0.0}),
            ),
            table_preprocessors=(Spec(name="zscore"),),
            metrics=(Spec(name="auc"),),
            random_seed=0,
        )


@pytest.mark.unit
def test_search_writes_the_winning_classifier_parameter_into_the_spec() -> None:
    """The tuned MLSpec, not just a fitted object, carries the winner."""
    table = make_synthetic_feature_table(n_rows=60, n_features=6, rng=1)
    grid = {"model__component__C": [0.001, 1.0, 100.0]}
    result = search_hyperparameters(table, _steps_spec(), grid, n_splits=3)

    assert isinstance(result, SearchResult)
    winner = result.best_params["model__component__C"]
    assert winner in grid["model__component__C"]
    assert result.spec.classifier.params["C"] == winner
    # Nothing else moved: the declared max_iter and the whole step chain are
    # carried over verbatim.
    assert result.spec.classifier.params["max_iter"] == 500
    assert [entry.name for entry in result.spec.steps] == ["variance", "zscore"]
    # The refitted model really is the tuned one.
    assert result.model is not None
    assert result.model.pipeline.model.spec.params["C"] == winner


@pytest.mark.unit
def test_search_tunes_a_transformation_step_by_its_registered_name() -> None:
    """``variance__component__threshold`` reaches the selector's own parameter."""
    table = make_synthetic_feature_table(n_rows=60, n_features=6, rng=2)
    result = search_hyperparameters(
        table,
        _steps_spec(),
        {"variance__component__threshold": [0.0, 0.5]},
        n_splits=3,
    )
    winner = result.best_params["variance__component__threshold"]
    assert result.spec.steps[0].name == "variance"
    assert result.spec.steps[0].params["threshold"] == winner
    # The untouched second step is unchanged.
    assert result.spec.steps[1] == _steps_spec().steps[1]


@pytest.mark.unit
def test_tuned_spec_round_trips_and_refingerprints() -> None:
    """A tuned spec is a spec: it serialises, reloads and fingerprints."""
    table = make_synthetic_feature_table(n_rows=60, n_features=6, rng=3)
    base = _steps_spec()
    result = search_hyperparameters(
        table, base, {"model__component__C": [0.001, 100.0]}, n_splits=3
    )
    payload = result.spec.to_dict()
    assert MLSpec.from_dict(payload) == result.spec
    # The fingerprint tracks the tuning: a different C is a different
    # definition, and the manifest records the tuned one.
    assert result.spec.fingerprint() != base.fingerprint()
    assert result.manifest.provenance.spec_fingerprint == result.spec.fingerprint()


@pytest.mark.unit
def test_search_preserves_the_declared_field_layout() -> None:
    """
    Tuning must not silently migrate a spec's serialisation shape.

    ``MLSpec`` emits the deprecated three-chain keys for a spec declared with
    them and the single ``steps`` key otherwise; that asymmetry keeps every
    already-published fingerprint stable. A tuned spec therefore has to keep
    its predecessor's layout, or the same analysis would fingerprint
    differently before and after tuning for a reason unrelated to tuning.
    """
    table = make_synthetic_feature_table(n_rows=60, n_features=6, rng=4)
    grid = {"model__component__C": [0.001, 100.0]}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = search_hyperparameters(table, _legacy_spec(), grid, n_splits=3)
    assert legacy.spec.declares_deprecated_chains
    legacy_payload = legacy.spec.to_dict()
    assert "steps" not in legacy_payload
    assert [entry["name"] for entry in legacy_payload["table_preprocessors"]] == [
        "zscore"
    ]

    modern = search_hyperparameters(table, _steps_spec(), grid, n_splits=3)
    assert not modern.spec.declares_deprecated_chains
    modern_payload = modern.spec.to_dict()
    assert "steps" in modern_payload
    assert "table_preprocessors" not in modern_payload


@pytest.mark.unit
def test_objective_defaults_to_the_first_declared_metric() -> None:
    """The panel a study reports is the panel it tuned on."""
    table = make_synthetic_feature_table(n_rows=60, n_features=6, rng=5)
    grid = {"model__component__C": [0.001, 100.0]}
    declared = search_hyperparameters(table, _steps_spec(), grid, n_splits=3)
    assert declared.objective == "accuracy"
    # With no panel declared the recipe falls back to AUC, not accuracy.
    bare = search_hyperparameters(
        table, _steps_spec(metrics=()), grid, n_splits=3
    )
    assert bare.objective == "auc"
    # An explicit objective overrides both.
    explicit = search_hyperparameters(
        table, _steps_spec(), grid, n_splits=3, objective="f1_score"
    )
    assert explicit.objective == "f1_score"


@pytest.mark.unit
def test_reported_score_is_the_metric_value_and_matches_its_trial() -> None:
    """``best_score`` is the metric's own number, never sklearn's negation."""
    table = make_synthetic_feature_table(n_rows=60, n_features=6, rng=6)
    result = search_hyperparameters(
        table,
        _steps_spec(),
        {"model__component__C": [0.001, 1.0, 100.0]},
        n_splits=3,
        objective="auc",
    )
    assert len(result.trials) == 3
    best_trial = next(trial for trial in result.trials if trial["rank"] == 1)
    assert best_trial["params"] == result.best_params
    assert best_trial["mean_score"] == pytest.approx(result.best_score)
    # An AUC is a probability-scale number; a negated score would fail here.
    assert 0.0 <= result.best_score <= 1.0


@pytest.mark.unit
def test_search_is_reproducible_from_the_seed() -> None:
    """Same table, same spec, same seed: same winner and same trial table."""
    table = make_synthetic_feature_table(n_rows=60, n_features=6, rng=7)
    grid = {"model__component__C": [0.001, 1.0, 100.0]}
    first = search_hyperparameters(table, _steps_spec(), grid, n_splits=3, seed=11)
    second = search_hyperparameters(table, _steps_spec(), grid, n_splits=3, seed=11)
    assert first.best_params == second.best_params
    assert [trial["mean_score"] for trial in first.trials] == [
        trial["mean_score"] for trial in second.trials
    ]
    # The seed override is folded into the tuned spec, so the record cannot
    # disagree with the run.
    assert first.spec.random_seed == 11


@pytest.mark.unit
def test_random_strategy_honours_the_candidate_budget() -> None:
    """``strategy="random"`` evaluates exactly ``n_iter`` candidates."""
    table = make_synthetic_feature_table(n_rows=60, n_features=6, rng=8)
    result = search_hyperparameters(
        table,
        _steps_spec(),
        {"model__component__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
        n_splits=3,
        strategy="random",
        n_iter=3,
        seed=5,
    )
    assert len(result.trials) == 3
    assert result.spec.classifier.params["C"] in (
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    )


@pytest.mark.unit
def test_unknown_strategy_is_rejected() -> None:
    """An unsupported backend fails loudly instead of falling back to grid."""
    table = make_synthetic_feature_table(n_rows=30, n_features=4, rng=9)
    with pytest.raises(HABITAPIError, match="strategy 'bayesian'"):
        search_hyperparameters(
            table,
            _steps_spec(),
            {"model__component__C": [1.0]},
            strategy="bayesian",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "key",
    [
        "model__C",  # missing the 'component' segment
        "model__component__C__extra",  # too deep to write back
        "C",  # not addressed at a step at all
        "model__component__",  # no parameter named
    ],
)
def test_a_key_that_cannot_be_written_back_is_rejected_before_searching(
    key: str,
) -> None:
    """A search whose result cannot be recorded is worse than no search."""
    table = make_synthetic_feature_table(n_rows=30, n_features=4, rng=10)
    with pytest.raises(HABITAPIError, match="component"):
        search_hyperparameters(table, _steps_spec(), {key: [1.0]}, n_splits=2)


@pytest.mark.unit
def test_a_key_naming_an_absent_step_is_rejected() -> None:
    """The error names the steps that ARE tunable."""
    table = make_synthetic_feature_table(n_rows=30, n_features=4, rng=11)
    with pytest.raises(HABITAPIError, match="does not have"):
        search_hyperparameters(
            table,
            _steps_spec(),
            {"lasso__component__cv": [3, 5]},
            n_splits=2,
        )


@pytest.mark.unit
def test_an_empty_grid_is_rejected() -> None:
    """Searching over nothing is a mistake, not a no-op."""
    table = make_synthetic_feature_table(n_rows=30, n_features=4, rng=12)
    with pytest.raises(HABITAPIError, match="non-empty mapping"):
        search_hyperparameters(table, _steps_spec(), {}, n_splits=2)
    with pytest.raises(HABITAPIError, match="nothing to search"):
        search_hyperparameters(table, _steps_spec(), [], n_splits=2)


@pytest.mark.unit
def test_search_needs_at_least_two_folds() -> None:
    """One fold cannot estimate anything to choose between candidates."""
    table = make_synthetic_feature_table(n_rows=30, n_features=4, rng=13)
    with pytest.raises(HABITAPIError, match="at least 2 folds"):
        search_hyperparameters(
            table, _steps_spec(), {"model__component__C": [1.0]}, n_splits=1
        )


@pytest.mark.unit
def test_refit_false_returns_the_tuned_spec_only() -> None:
    """Nested CV needs the tuned definition without a whole-table refit."""
    table = make_synthetic_feature_table(n_rows=60, n_features=6, rng=14)
    result = search_hyperparameters(
        table,
        _steps_spec(),
        {"model__component__C": [0.001, 100.0]},
        n_splits=3,
        refit=False,
    )
    assert result.model is None
    assert result.spec.classifier.params["C"] in (0.001, 100.0)


@pytest.mark.unit
def test_nested_cross_validation_tunes_inside_every_outer_fold() -> None:
    """Each outer fold reports its own winner, chosen on its training rows."""
    table = make_synthetic_feature_table(n_rows=80, n_features=6, rng=15)
    result = cross_validate(
        table,
        _steps_spec(),
        n_splits=3,
        inner_cv=2,
        param_grid={"model__component__C": [0.001, 100.0]},
    )
    assert isinstance(result, CVResult)
    assert len(result.fold_best_params) == 3
    for fold_params in result.fold_best_params:
        assert set(fold_params) == {"model__component__C"}
        assert fold_params["model__component__C"] in (0.001, 100.0)
    # Every outer fold's pipeline carries the parameter its own inner search
    # picked -- which is what makes the outer score an estimate of the tuning
    # procedure rather than of one fixed parameter set.
    for pipeline, fold_params in zip(result.pipelines, result.fold_best_params):
        assert (
            pipeline.model.spec.params["C"]
            == fold_params["model__component__C"]
        )
    assert set(result.mean_metrics) == {"accuracy", "auc"}


@pytest.mark.unit
def test_plain_cross_validation_reports_no_tuning() -> None:
    """Without nesting arguments nothing about cross_validate changes."""
    table = make_synthetic_feature_table(n_rows=60, n_features=6, rng=16)
    result = cross_validate(table, _steps_spec(), n_splits=3)
    assert result.fold_best_params == ()
    assert result.n_splits == 3


@pytest.mark.unit
def test_half_declared_nesting_is_rejected() -> None:
    """A grid without inner folds would tune on the rows it scores."""
    table = make_synthetic_feature_table(n_rows=40, n_features=4, rng=17)
    with pytest.raises(HABITAPIError, match="no inner_cv"):
        cross_validate(
            table,
            _steps_spec(),
            n_splits=2,
            param_grid={"model__component__C": [1.0]},
        )
    with pytest.raises(HABITAPIError, match="no param_grid"):
        cross_validate(table, _steps_spec(), n_splits=2, inner_cv=2)


@pytest.mark.unit
def test_nested_manifest_records_the_untuned_protocol() -> None:
    """
    There is no single tuned spec in a nested run, so none is claimed.

    Each outer fold selected its own parameters; recording one of them as
    "the" spec would misattribute the reported panel to a definition that
    only ever saw part of the data.
    """
    table = make_synthetic_feature_table(n_rows=60, n_features=6, rng=18)
    spec = _steps_spec()
    result = cross_validate(
        table,
        spec,
        n_splits=3,
        inner_cv=2,
        param_grid={"model__component__C": [0.001, 100.0]},
    )
    assert result.manifest.provenance.spec_fingerprint == spec.fingerprint()
