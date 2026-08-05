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
"""Fast contract tests for the public sklearn-style HABIT API."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import joblib
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

import habit


def _ml_config(tmp_path: Path) -> dict[str, object]:
    """Build the smallest official-schema classification configuration."""
    return {
        "run_mode": "train",
        "input": [
            {
                "path": str(tmp_path / "unused.csv"),
                "subject_id_col": "subject",
                "label_col": "label",
            }
        ],
        "output": str(tmp_path / "output"),
        "models": {
            "LogisticRegression": {
                "params": {"solver": "liblinear", "random_state": 1},
            }
        },
        "feature_selection_methods": [],
        "is_visualize": False,
    }


@pytest.mark.unit
def test_estimator_symbols_are_lazily_public() -> None:
    """Estimator classes and their public error resolve from both facades."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, habit; "
                "assert 'habit.api.estimators' not in sys.modules; "
                "assert habit.HabitClassifier.__name__ == 'HabitClassifier'"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert habit.HabitClassifier is habit.api.HabitClassifier
    assert habit.HabitatClusterer is habit.api.HabitatClusterer
    assert habit.SubjectFeatureAggregator is habit.api.SubjectFeatureAggregator
    assert habit.HABITAPIError is habit.api.HABITAPIError


@pytest.mark.unit
def test_subject_feature_aggregator_learns_stable_schema() -> None:
    """Fitted aggregation preserves feature/habitat columns for later subjects."""
    train = pd.DataFrame(
        {
            "subject": ["a", "a", "b", "b"],
            "habitats": [1, 2, 1, 2],
            "count": [4, 5, 6, 7],
            "intensity": [1.0, 3.0, 2.0, 4.0],
            "metadata": ["x", "x", "y", "y"],
        }
    )
    estimator = habit.SubjectFeatureAggregator().fit(train)

    transformed = estimator.transform(
        pd.DataFrame(
            {
                "subject": ["c"],
                "habitats": [1],
                "count": [8],
                "intensity": [9.0],
            }
        )
    )

    assert transformed.columns.tolist() == estimator.get_feature_names_out().tolist()
    assert transformed.index.name == "subject"
    assert transformed.loc["c", "intensity__habitat_1"] == 9.0
    assert transformed.loc["c", "intensity__habitat_2"] == 0.0
    assert estimator.feature_names_in_.tolist() == ["intensity"]
    assert estimator.n_features_in_ == 1


@pytest.mark.unit
def test_subject_feature_aggregator_rejects_missing_contract_columns() -> None:
    """Long-table validation reports missing identifiers and fitted features."""
    estimator = habit.SubjectFeatureAggregator()
    with pytest.raises(habit.HABITAPIError, match="required columns"):
        estimator.fit(pd.DataFrame({"subject": ["a"], "feature": [1.0]}))

    estimator.fit(pd.DataFrame({"subject": ["a"], "habitats": [1], "feature": [1.0]}))
    with pytest.raises(habit.HABITAPIError, match="fitted feature columns"):
        estimator.transform(pd.DataFrame({"subject": ["a"], "habitats": [1]}))


@pytest.mark.unit
def test_habit_classifier_supports_clone_pipeline_and_persistence(
    tmp_path: Path,
) -> None:
    """Classifier remains cloneable and usable as the final sklearn Pipeline step."""
    features = pd.DataFrame(
        {
            "first": [0.0, 0.1, 0.9, 1.0, 0.2, 0.8],
            "second": [0.1, 0.0, 1.0, 0.9, 0.2, 0.8],
        }
    )
    labels = np.asarray([0, 0, 1, 1, 0, 1])
    classifier = habit.HabitClassifier(_ml_config(tmp_path))

    params = classifier.get_params(deep=True)
    assert params["model_name"] is None
    classifier.set_params(model_name="LogisticRegression")
    assert classifier.model_name == "LogisticRegression"

    cloned = clone(classifier)
    pipeline = Pipeline([("classifier", cloned)]).fit(features, labels)
    fitted = pipeline.named_steps["classifier"]

    assert fitted.n_features_in_ == 2
    assert fitted.feature_names_in_.tolist() == ["first", "second"]
    assert fitted.classes_.tolist() == [0, 1]
    assert fitted.predict_proba(features).shape == (6, 2)
    assert fitted.score(features, labels) >= 0.8

    path = tmp_path / "classifier.joblib"
    fitted.save(path)
    payload = joblib.load(path)
    assert payload["metadata"]["habit_version"] == habit.__version__
    assert payload["metadata"]["estimator_type"] == "HabitClassifier"
    loaded = habit.HabitClassifier.load(path)
    assert loaded.serialization_metadata_["schema_version"] == 1
    np.testing.assert_array_equal(loaded.predict(features), fitted.predict(features))


@pytest.mark.unit
def test_habit_classifier_rejects_unfitted_and_wrong_columns(tmp_path: Path) -> None:
    """Inference requires a fitted estimator and its exact DataFrame schema."""
    classifier = habit.HabitClassifier(_ml_config(tmp_path))
    with pytest.raises(NotFittedError):
        classifier.predict(pd.DataFrame({"first": [1.0], "second": [2.0]}))

    classifier.fit(
        pd.DataFrame({"first": [0.0, 1.0, 0.1, 0.9], "second": [0.0, 1.0, 0.2, 0.8]}),
        np.asarray([0, 1, 0, 1]),
    )
    with pytest.raises(habit.HABITAPIError, match="columns do not match"):
        classifier.predict(pd.DataFrame({"first": [0.1], "unexpected": [0.2]}))


@pytest.mark.unit
def test_habitat_clusterer_delegates_and_checks_fitted_state(tmp_path: Path) -> None:
    """Clusterer delegates to HabitatAnalysis without running image processing."""
    config = {
        "data_dir": str(tmp_path / "data"),
        "out_dir": str(tmp_path / "out"),
        "habitat_segmentation": {"clustering_mode": "one_step"},
    }
    results = pd.DataFrame({"subject": ["a"], "habitats": [1]})
    analysis = MagicMock()
    analysis.fit.return_value = results
    analysis.pipeline = MagicMock()

    clusterer = habit.HabitatClusterer(config)
    with pytest.raises(NotFittedError):
        clusterer.transform(["a"])

    with (
        patch.object(
            habit.HabitatClusterer,
            "_validated_config",
            return_value=MagicMock(out_dir=str(tmp_path / "out")),
        ),
        patch.object(
            habit.HabitatClusterer,
            "_create_analysis",
            return_value=analysis,
        ),
    ):
        assert clusterer.fit_transform(["a"]) is results

    analysis.fit.assert_called_once_with(subjects=["a"], save_results_csv=None)
    assert clusterer.pipeline_ is analysis.pipeline

    analysis.transform_with_pipeline.return_value = results
    transformed = clusterer.transform(["b"])

    assert transformed is results
    analysis.transform_with_pipeline.assert_called_once_with(
        pipeline=analysis.pipeline,
        subjects=["b"],
        save_results_csv=None,
    )
    assert clusterer.pipeline_ is analysis.pipeline


@pytest.mark.unit
def test_dicom_habitat_and_ml_runners_delegate_to_core() -> None:
    """Every remaining top-level workflow runner preserves its core delegation."""
    config = MagicMock()
    config.out_dir = "habitat-output"
    config.output_dir = "dicom-output"
    config.output = "ml-output"
    config.run_mode = "train"
    logger = MagicMock()
    with (
        patch("habit.api.dicom_sort.coerce_config", return_value=config),
        patch(
            "habit.api.habitat.coerce_config",
            return_value=config,
        ),
        patch(
            "habit.api.machine_learning.coerce_config",
            return_value=config,
        ),
        patch("habit.compat.dicom_sort_runner.run_dicom_sort") as dicom_run,
        patch(
            "habit.compat.habitat_runner.run_habitat_analysis_from_config",
            return_value=pd.DataFrame(),
        ) as habitat_run,
        patch(
            "habit.compat.ml_runner.run_ml_from_config",
            return_value=MagicMock(metrics={}),
        ) as ml_run,
        patch(
            "habit.compat.ml_runner.run_kfold_from_config",
            return_value=MagicMock(),
        ) as kfold_run,
        patch("habit.api.dicom_sort.create_run_manifest"),
        patch("habit.api.dicom_sort.write_run_manifest"),
        patch("habit.api.habitat.create_run_manifest"),
        patch("habit.api.habitat.write_run_manifest"),
        patch("habit.api.machine_learning.create_run_manifest"),
        patch("habit.api.machine_learning.write_run_manifest"),
    ):
        habit.run_dicom_sort(config)
        assert habit.run_habitat_analysis(config, logger=logger).data.empty
        habit.run_ml(config, logger=logger, output_dir="ml-output")
        habit.run_kfold(config, logger=logger, output_dir="kfold-output")

    dicom_run.assert_called_once_with(config, logger=None)
    habitat_run.assert_called_once_with(config, logger=logger)
    ml_run.assert_called_once_with(config, logger=logger, output_dir="ml-output")
    kfold_run.assert_called_once_with(
        config,
        logger=logger,
        output_dir="kfold-output",
    )
