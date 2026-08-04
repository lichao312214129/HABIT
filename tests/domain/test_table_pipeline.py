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
"""Tests for TablePipeline: the train/predict-consistent tabular flow."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from habit.api.exceptions import CompatibilityError, HABITAPIError
from habit.domain.classification import LogisticRegressionClassifier, RandomForestClassifier
from habit.domain.evaluation import AccuracyMetric, AucMetric, HosmerLemeshowPValueMetric
from habit.domain.feature_selection import IccSelector, VarianceSelector
from habit.domain.pipeline import TablePipeline
from habit.domain.table_preprocessing import ZScorePreprocessor
from habit.spec import Spec

from .conftest import make_feature_table


def _pipeline() -> TablePipeline:
    """A variance-then-zscore chain ending in logistic regression."""
    return TablePipeline(
        steps=[VarianceSelector(threshold=0.01), ZScorePreprocessor()],
        classifier=LogisticRegressionClassifier(max_iter=500),
    )


@pytest.mark.unit
def test_fit_transform_predict_roundtrip() -> None:
    """The fitted chain selects, scales and predicts new subjects."""
    train = make_feature_table(seed=1)
    pipeline = _pipeline().fit(train)
    new = make_feature_table(tuple(f"N{i}" for i in range(6)), seed=2)
    labels = pipeline.predict(new)
    assert list(labels.index) == list(new.frame["subject"])
    probabilities = pipeline.predict_proba(new)
    assert probabilities.shape == (6, 2)
    # transform exposes the classifier-ready table directly.
    transformed = pipeline.transform(new)
    assert transformed.feature_columns == pipeline.transform(train).feature_columns


@pytest.mark.unit
def test_prediction_uses_training_statistics() -> None:
    """New tables are scaled with TRAINING mean/std (no leakage)."""
    train = make_feature_table(seed=3)
    pipeline = TablePipeline(
        steps=[ZScorePreprocessor()], classifier=LogisticRegressionClassifier()
    ).fit(train)
    new = make_feature_table(["X1", "X2"], seed=4)
    transformed = pipeline.transform(new)
    train_block = train.frame[list(train.feature_columns)]
    expected = (new.frame[list(train.feature_columns)] - train_block.mean()) / train_block.std()
    np.testing.assert_allclose(
        transformed.frame[list(transformed.feature_columns)].to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
    )


@pytest.mark.unit
def test_unfitted_pipeline_and_missing_classifier_raise() -> None:
    """transform/predict before fit, and a missing classifier, are errors."""
    with pytest.raises(HABITAPIError):
        TablePipeline(steps=[], classifier=None)  # type: ignore[arg-type]
    pipeline = _pipeline()
    with pytest.raises(HABITAPIError):
        pipeline.transform(make_feature_table())
    with pytest.raises(HABITAPIError):
        pipeline.predict(make_feature_table())


@pytest.mark.unit
def test_evaluate_scores_label_and_probability_metrics() -> None:
    """evaluate returns one value per metric, keyed by the spec name."""
    train = make_feature_table(seed=5)
    pipeline = _pipeline().fit(train)
    results = pipeline.evaluate(
        train, [AccuracyMetric(), AucMetric(), HosmerLemeshowPValueMetric(n_groups=4)]
    )
    assert set(results) == {"accuracy", "auc", "hosmer_lemeshow_p_value"}
    assert results["accuracy"] >= 0.9
    assert 0.0 <= results["auc"] <= 1.0
    assert 0.0 <= results["hosmer_lemeshow_p_value"] <= 1.0


@pytest.mark.unit
def test_evaluate_requires_metrics_and_outcome() -> None:
    """An empty metric list or an outcome-less table is a clear error."""
    pipeline = _pipeline().fit(make_feature_table(seed=6))
    with pytest.raises(HABITAPIError):
        pipeline.evaluate(make_feature_table(seed=7), [])
    with pytest.raises(HABITAPIError):
        pipeline.evaluate(make_feature_table(seed=7, outcome=False), [AccuracyMetric()])


@pytest.mark.unit
def test_set_random_state_propagates_to_seedable_components() -> None:
    """One seeding call reaches every Seedable component of the pipeline."""
    train = make_feature_table(seed=8)
    first = TablePipeline(steps=[], classifier=RandomForestClassifier())
    second = TablePipeline(steps=[], classifier=RandomForestClassifier())
    first.set_random_state(17)
    second.set_random_state(17)
    np.testing.assert_array_equal(
        first.fit(train).predict(train).to_numpy(),
        second.fit(train).predict(train).to_numpy(),
    )


@pytest.mark.unit
def test_spec_describes_every_stage() -> None:
    """The composed spec lists the step specs and the model spec."""
    pipeline = _pipeline()
    spec = pipeline.spec
    assert spec.name == "table_pipeline"
    assert [s["name"] for s in spec.params["steps"]] == ["variance", "zscore"]
    assert spec.params["model"]["name"] == "LogisticRegression"


@pytest.mark.unit
def test_save_load_roundtrip_preserves_predictions(tmp_path) -> None:
    """A saved pipeline reloads with identical fitted behaviour."""
    train = make_feature_table(seed=9)
    pipeline = _pipeline().fit(train)
    destination = pipeline.save(tmp_path / "model.habitpipeline")
    with zipfile.ZipFile(destination, "r") as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    assert manifest["format"] == "habit.tablepipeline"
    assert manifest["format_version"] == 1
    assert manifest["is_fitted"] is True
    assert [record["spec"]["name"] for record in manifest["steps"]] == [
        "variance",
        "zscore",
    ]
    new = make_feature_table(tuple(f"N{i}" for i in range(5)), seed=10)
    loaded = TablePipeline.load(destination)
    np.testing.assert_allclose(
        loaded.predict_proba(new).to_numpy(), pipeline.predict_proba(new).to_numpy()
    )


@pytest.mark.unit
def test_load_rejects_foreign_and_newer_files(tmp_path) -> None:
    """Non-pipeline files and newer format versions raise CompatibilityError."""
    foreign = tmp_path / "foreign.zip"
    with zipfile.ZipFile(foreign, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"format": "other"}))
    with pytest.raises(CompatibilityError):
        TablePipeline.load(foreign)

    train = make_feature_table(seed=11)
    pipeline = _pipeline().fit(train)
    destination = pipeline.save(tmp_path / "newer.habitpipeline")
    with zipfile.ZipFile(destination, "r") as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        payload = archive.read("payload.pkl")
    manifest["format_version"] = 999
    with zipfile.ZipFile(destination, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("payload.pkl", payload)
    with pytest.raises(CompatibilityError):
        TablePipeline.load(destination)


@pytest.mark.unit
def test_load_rejects_internally_inconsistent_archives(tmp_path) -> None:
    """A manifest disagreeing with its payload signals archive corruption."""
    train = make_feature_table(seed=12)
    pipeline = _pipeline().fit(train)
    destination = pipeline.save(tmp_path / "tampered.habitpipeline")
    with zipfile.ZipFile(destination, "r") as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        payload = archive.read("payload.pkl")
    manifest["steps"] = [dict(record, spec=Spec(name="fake").to_dict()) for record in manifest["steps"]]
    with zipfile.ZipFile(destination, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("payload.pkl", payload)
    with pytest.raises(CompatibilityError):
        TablePipeline.load(destination)


@pytest.mark.unit
def test_repeat_tables_reach_icc_step_only() -> None:
    """Steps accepting repeat_tables receive them; the rest see the primary."""
    ids = tuple(f"S{i}" for i in range(16))
    primary = make_feature_table(ids, n_noise=1, seed=40)
    rng = np.random.RandomState(41)
    repeat = make_feature_table(ids, n_noise=1, seed=42)
    repeat.frame["signal"] = primary.frame["signal"] + rng.normal(scale=0.01, size=16)
    repeat.frame["noise0"] = rng.normal(size=16)
    pipeline = TablePipeline(
        steps=[IccSelector(threshold=0.75)],
        classifier=LogisticRegressionClassifier(),
    ).fit(primary, repeat_tables=[repeat])
    # The unstable noise column was filtered at fit time, so it is dropped
    # from prediction tables too.
    transformed = pipeline.transform(primary)
    assert "signal" in transformed.feature_columns
    assert "noise0" not in transformed.feature_columns
