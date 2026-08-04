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
"""Contract tests for the tabular ML recipes (train_model / cross_validate / predict_model).

Everything runs on the deterministic synthetic feature table, so the whole
file finishes in seconds and never touches demo data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from habit.api.exceptions import HABITAPIError
from habit.contracts.table import FeatureTable
from habit.datasets.synthetic import make_synthetic_feature_table
from habit.recipes.modeling import (
    CVResult,
    ModelResult,
    PredictionResult,
    cross_validate,
    predict_model,
    train_model,
)
from habit.spec.specs import MLSpec, Spec


def _spec(**overrides) -> MLSpec:
    """Build a modelling spec over registered, fast components."""
    fields = {
        "name": "test_ml",
        "classifier": Spec(
            name="LogisticRegression", params={"max_iter": 500}
        ),
        "table_preprocessors": (Spec(name="zscore"),),
        "metrics": (Spec(name="accuracy"), Spec(name="auc")),
        "random_seed": 0,
    }
    fields.update(overrides)
    return MLSpec(**fields)


@pytest.mark.unit
def test_train_model_fits_and_scores_training_table() -> None:
    """train_model returns a fitted pipeline plus the in-sample panel."""
    table = make_synthetic_feature_table(n_rows=40, n_features=6, rng=1)
    result = train_model(table, _spec())

    assert isinstance(result, ModelResult)
    assert set(result.train_metrics) == {"accuracy", "auc"}
    # The synthetic signal separates the classes, so in-sample accuracy is high.
    assert result.train_metrics["accuracy"] >= 0.9

    # The fitted pipeline predicts on fresh rows without refitting.
    fresh = make_synthetic_feature_table(n_rows=8, n_features=6, rng=2)
    labels = result.pipeline.predict(fresh)
    assert len(labels) == 8


@pytest.mark.unit
def test_train_model_seed_override_folds_into_manifest() -> None:
    """A call-site seed override changes the recorded spec, not just the run."""
    table = make_synthetic_feature_table(n_rows=20, n_features=4, rng=3)
    base = train_model(table, _spec())
    overridden = train_model(table, _spec(), seed=7)

    assert overridden.manifest.spec_payload["random_seed"] == 7
    assert overridden.manifest.provenance.random_seed == 7
    assert (
        overridden.manifest.provenance.spec_fingerprint
        != base.manifest.provenance.spec_fingerprint
    )


@pytest.mark.unit
def test_train_model_requires_an_outcome() -> None:
    """An unlabelled table is a clear error, not a crash inside sklearn."""
    table = make_synthetic_feature_table(n_rows=10, n_features=3, rng=4)
    unlabelled = FeatureTable(
        frame=table.frame.drop(columns=["label"]),
        id_columns=table.id_columns,
        feature_columns=table.feature_columns,
        outcome=None,
    )
    with pytest.raises(HABITAPIError, match="outcome"):
        train_model(unlabelled, _spec())


@pytest.mark.unit
def test_cross_validate_returns_per_fold_and_summary() -> None:
    """cross_validate refits per fold and summarises the panel."""
    table = make_synthetic_feature_table(n_rows=40, n_features=6, rng=5)
    result = cross_validate(table, _spec(), n_splits=4)

    assert isinstance(result, CVResult)
    assert result.n_splits == 4
    assert len(result.fold_metrics) == 4
    assert len(result.pipelines) == 4
    assert set(result.mean_metrics) == {"accuracy", "auc"}
    assert set(result.std_metrics) == {"accuracy", "auc"}
    # Folds are refit independently: distinct fitted objects.
    assert result.pipelines[0] is not result.pipelines[1]
    # The signal generalises, so cross-validated accuracy stays high.
    assert result.mean_metrics["accuracy"] >= 0.8
    # The summary is the across-fold mean of the per-fold panel.
    expected = sum(fold["accuracy"] for fold in result.fold_metrics) / 4
    assert result.mean_metrics["accuracy"] == pytest.approx(expected)


@pytest.mark.unit
def test_cross_validate_is_deterministic_for_a_fixed_seed() -> None:
    """Same table, same spec, same folds: identical panels."""
    table = make_synthetic_feature_table(n_rows=30, n_features=5, rng=6)
    first = cross_validate(table, _spec(), n_splits=3)
    second = cross_validate(table, _spec(), n_splits=3)
    assert first.fold_metrics == second.fold_metrics


@pytest.mark.unit
def test_cross_validate_rejects_too_few_splits() -> None:
    """n_splits below 2 is a usage error with a clear message."""
    with pytest.raises(HABITAPIError, match="n_splits"):
        cross_validate(make_synthetic_feature_table(rng=7), _spec(), n_splits=1)


@pytest.mark.unit
def test_recipes_fall_back_to_default_metric_panel() -> None:
    """A spec without metrics gets the recipe's default panel."""
    table = make_synthetic_feature_table(n_rows=30, n_features=5, rng=8)
    result = train_model(table, _spec(metrics=()))
    assert set(result.train_metrics) == {"accuracy", "auc"}


@pytest.mark.unit
def test_manifest_records_spec_provenance_and_rows() -> None:
    """The manifest ties the run to the effective spec and every row."""
    table = make_synthetic_feature_table(n_rows=20, n_features=4, rng=9)
    result = train_model(table, _spec())

    manifest = result.manifest
    assert manifest.spec_payload["classifier"]["name"] == "LogisticRegression"
    assert manifest.provenance.produced_by == "recipes.modeling.train_model"
    assert manifest.provenance.spec_fingerprint == _spec().fingerprint()
    assert len(manifest.subject_outcomes) == 20
    assert set(manifest.subject_outcomes.values()) == {"success"}
    assert manifest.started_at and manifest.finished_at


@pytest.mark.unit
def test_public_recipe_surface_includes_modeling() -> None:
    """The recipe package exports the ML assembly functions and results."""
    import habit.recipes as recipes

    assert {
        "train_model",
        "cross_validate",
        "predict_model",
        "ModelResult",
        "CVResult",
        "PredictionResult",
    } <= set(recipes.__all__)


# ---------------------------------------------------------------------------
# Hold-out splits in train_model
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_train_model_stratified_holdout_split() -> None:
    """A test_size hold-out fits on the training rows and scores both sides."""
    table = make_synthetic_feature_table(n_rows=40, n_features=6, rng=21)
    result = train_model(table, _spec(), test_size=0.25)

    assert result.test_metrics is not None
    assert set(result.test_metrics) == {"accuracy", "auc"}
    assert len(result.train_row_ids) == 30
    assert len(result.test_row_ids) == 10
    assert not set(result.train_row_ids) & set(result.test_row_ids)
    assert set(result.train_row_ids) | set(result.test_row_ids) == {
        str(value) for value in table.frame["subject"]
    }
    # The signal separates the classes, so the held-out readout stays high.
    assert result.test_metrics["accuracy"] >= 0.8


@pytest.mark.unit
def test_train_model_holdout_split_is_deterministic() -> None:
    """Same table, same spec seed: identical split and panels."""
    table = make_synthetic_feature_table(n_rows=30, n_features=5, rng=22)
    first = train_model(table, _spec(), test_size=0.3)
    second = train_model(table, _spec(), test_size=0.3)
    assert first.train_row_ids == second.train_row_ids
    assert first.test_row_ids == second.test_row_ids
    assert first.test_metrics == second.test_metrics


@pytest.mark.unit
def test_train_model_custom_id_split() -> None:
    """Explicit id lists define the split exactly; unlisted rows never fit."""
    table = make_synthetic_feature_table(n_rows=30, n_features=5, rng=23)
    all_ids = [str(value) for value in table.frame["subject"]]
    train_ids, test_ids = all_ids[:18], all_ids[18:24]
    result = train_model(table, _spec(), train_ids=train_ids, test_ids=test_ids)

    assert result.train_row_ids == tuple(train_ids)
    assert result.test_row_ids == tuple(test_ids)
    assert result.test_metrics is not None
    # Rows absent from both lists are left out of the fit entirely (v0.1 rule).
    assert set(all_ids[24:]).isdisjoint(result.train_row_ids)


@pytest.mark.unit
def test_train_model_rejects_ambiguous_split_arguments() -> None:
    """test_size and id lists are mutually exclusive; ids come in pairs."""
    table = make_synthetic_feature_table(n_rows=20, n_features=4, rng=24)
    ids = [str(value) for value in table.frame["subject"]]
    with pytest.raises(HABITAPIError, match="never both"):
        train_model(
            table, _spec(), test_size=0.3, train_ids=ids[:10], test_ids=ids[10:]
        )
    with pytest.raises(HABITAPIError, match="both train_ids and test_ids"):
        train_model(table, _spec(), train_ids=ids[:10])


@pytest.mark.unit
def test_train_model_rejects_unknown_or_overlapping_ids() -> None:
    """Id files naming missing rows or overlapping sides are usage errors."""
    table = make_synthetic_feature_table(n_rows=20, n_features=4, rng=25)
    ids = [str(value) for value in table.frame["subject"]]
    with pytest.raises(HABITAPIError, match="does not have"):
        train_model(
            table, _spec(), train_ids=ids[:10] + ["ghost"], test_ids=ids[10:]
        )
    with pytest.raises(HABITAPIError, match="overlap"):
        train_model(
            table, _spec(), train_ids=ids[:12], test_ids=ids[10:]
        )


# ---------------------------------------------------------------------------
# predict_model: the inference recipe
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_predict_model_replays_fitted_state() -> None:
    """predict_model applies the fitted pipeline to fresh rows, no refit."""
    table = make_synthetic_feature_table(n_rows=40, n_features=6, rng=26)
    trained = train_model(table, _spec())

    fresh = make_synthetic_feature_table(n_rows=8, n_features=6, rng=27)
    result = predict_model(trained.pipeline, fresh)

    assert isinstance(result, PredictionResult)
    assert len(result.predictions) == 8
    assert list(result.predictions.index) == [
        str(value) for value in fresh.frame["subject"]
    ]
    assert result.probabilities is not None
    assert set(result.probabilities.columns) == {"0", "1"}
    assert (
        result.manifest.provenance.produced_by == "recipes.modeling.predict_model"
    )
    assert len(result.manifest.subject_outcomes) == 8


@pytest.mark.unit
def test_predict_model_accepts_unlabelled_tables() -> None:
    """Inference is legitimate without an outcome column."""
    table = make_synthetic_feature_table(n_rows=30, n_features=5, rng=28)
    trained = train_model(table, _spec())
    unlabelled = FeatureTable(
        frame=table.frame.drop(columns=["label"]),
        id_columns=table.id_columns,
        feature_columns=table.feature_columns,
        outcome=None,
    )
    result = predict_model(trained.pipeline, unlabelled)
    assert len(result.predictions) == 30


@pytest.mark.unit
def test_predict_model_roundtrips_through_save_load(tmp_path: Path) -> None:
    """A saved ``.habitpipeline`` predicts exactly as the in-memory one."""
    table = make_synthetic_feature_table(n_rows=30, n_features=5, rng=29)
    trained = train_model(table, _spec())
    artefact = trained.pipeline.save(tmp_path / "model.habitpipeline")

    from habit.domain.pipeline import TablePipeline

    reloaded = TablePipeline.load(artefact)
    in_memory = predict_model(trained.pipeline, table)
    from_disk = predict_model(reloaded, table)
    assert list(in_memory.predictions) == list(from_disk.predictions)


# ---------------------------------------------------------------------------
# icc_precomputed: selection from a precomputed ICC results JSON
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_train_model_with_precomputed_icc_selector(tmp_path: Path) -> None:
    """The pipeline selects exactly the features stable in the JSON."""
    table = make_synthetic_feature_table(n_rows=30, n_features=5, rng=30)
    icc_path = tmp_path / "icc_results.json"
    icc_path.write_text(
        json.dumps(
            {
                "test_vs_retest": {
                    "signal": 0.93,
                    "noise0": 0.5,
                    "noise1": {"ICC3": {"value": 0.81}},
                    "noise2": {"ICC2": 0.1},
                }
            }
        ),
        encoding="utf-8",
    )
    spec = _spec(
        feature_selectors=(
            Spec(
                name="icc_precomputed",
                params={
                    "icc_results_path": str(icc_path),
                    "groups": ["test_vs_retest"],
                    "threshold": 0.75,
                },
            ),
        )
    )
    result = train_model(table, spec)

    # signal (simple format) and noise1 (nested value) clear 0.75; noise0
    # and noise2 do not.
    assert set(result.pipeline.transform(table).feature_columns) == {
        "signal",
        "noise1",
    }
    assert result.train_metrics["accuracy"] >= 0.9


@pytest.mark.unit
def test_precomputed_icc_selector_requires_groups_and_file(tmp_path: Path) -> None:
    """Missing file and unknown group are loud errors, not empty selections."""
    from habit.domain.feature_selection.selectors import PrecomputedIccSelector

    table = make_synthetic_feature_table(n_rows=10, n_features=3, rng=31)
    missing = PrecomputedIccSelector(
        icc_results_path=str(tmp_path / "nope.json"), groups=["g"]
    )
    with pytest.raises(HABITAPIError, match="no ICC results file"):
        missing.fit(table)

    icc_path = tmp_path / "icc.json"
    icc_path.write_text(json.dumps({"group_a": {"signal": 0.9}}), encoding="utf-8")
    unknown = PrecomputedIccSelector(
        icc_results_path=str(icc_path), groups=["group_b"]
    )
    with pytest.raises(HABITAPIError, match="group_b"):
        unknown.fit(table)
