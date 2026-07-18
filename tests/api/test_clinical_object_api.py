"""Tests for the high-level clinical object API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pandas as pd
import pytest

import habit
from habit.api.contracts import WorkflowResult


class _HabitatConfigStub:
    """Minimal immutable-like config stub used to isolate object API delegation."""

    def __init__(self, **values: Any) -> None:
        self.data_dir = values["data_dir"]
        self.out_dir = values["out_dir"]
        self.run_mode = values.get("run_mode", "train")
        self.pipeline_path = values.get("pipeline_path")

    def model_copy(self, *, update: Dict[str, Any]) -> "_HabitatConfigStub":
        """Return a replaced config without mutating the original test config."""
        values = {
            "data_dir": self.data_dir,
            "out_dir": self.out_dir,
            "run_mode": self.run_mode,
            "pipeline_path": self.pipeline_path,
        }
        values.update(update)
        return _HabitatConfigStub(**values)


@pytest.mark.unit
def test_cohort_normalizes_public_metadata() -> None:
    """Cohort exposes immutable, explicit study-level input metadata."""
    cohort = habit.Cohort.from_directory(
        "input/cohort",
        name="training",
        subject_ids=["subject_01", "subject_02"],
        metadata={"site": "hospital_a"},
    )

    assert cohort.data_dir == Path("input/cohort")
    assert cohort.subject_ids == ("subject_01", "subject_02")
    assert cohort.metadata["site"] == "hospital_a"
    with pytest.raises(TypeError):
        cohort.metadata["site"] = "hospital_b"  # type: ignore[index]


@pytest.mark.unit
def test_clinical_preprocessor_uses_existing_public_runner() -> None:
    """The object API delegates to the stable config-driven preprocessing runner."""
    config = {"data_dir": "configured_input", "out_dir": "prepared_output"}
    workflow_result = WorkflowResult(output_dir=Path("prepared_output"))
    cohort = habit.Cohort.from_directory("selected_input", name="baseline")

    with patch(
        "habit.api.preprocessing.run_preprocess",
        return_value=workflow_result,
    ) as mock_run:
        preprocessor = habit.ClinicalPreprocessor(config)
        prepared = preprocessor.fit_transform(cohort)

    delegated_config = mock_run.call_args.args[0]
    assert delegated_config.data_dir == "selected_input"
    assert prepared.data_dir == Path("prepared_output") / "processed_images"
    assert prepared.name == "baseline"
    assert prepared.preprocessing_result is workflow_result


@pytest.mark.unit
def test_clinical_preprocessor_uses_config_data_dir_when_cohort_omitted() -> None:
    """fit_transform() may rely solely on config['data_dir']."""
    config = {"data_dir": "configured_input", "out_dir": "prepared_output"}
    workflow_result = WorkflowResult(output_dir=Path("prepared_output"))

    with patch(
        "habit.api.preprocessing.run_preprocess",
        return_value=workflow_result,
    ) as mock_run:
        prepared = habit.ClinicalPreprocessor(config).fit_transform()

    delegated_config = mock_run.call_args.args[0]
    assert delegated_config.data_dir == "configured_input"
    assert prepared.data_dir == Path("prepared_output") / "processed_images"


@pytest.mark.unit
def test_habitat_segmenter_preserves_train_predict_lifecycle() -> None:
    """Segmenter converts cohort input into explicit train and predict configs."""
    training_config = _HabitatConfigStub(
        data_dir="configured_train",
        out_dir="training_output",
    )
    training_table = pd.DataFrame({"subject": ["train_01"], "habitats": [1]})
    prediction_table = pd.DataFrame({"subject": ["test_01"], "habitats": [2]})
    training_result = WorkflowResult(
        data=training_table,
        output_dir=Path("training_output"),
    )
    prediction_result = WorkflowResult(
        data=prediction_table,
        output_dir=Path("prediction_output"),
    )

    with (
        patch(
            "habit.api.clinical.coerce_config",
            return_value=training_config,
        ),
        patch(
            "habit.api.habitat.run_habitat_analysis",
            side_effect=[training_result, prediction_result],
        ) as mock_run,
    ):
        segmenter = habit.HabitatSegmenter(
            {"unused": "validated by the patched config boundary"},
            prediction_output_dir="prediction_output",
        )
        fitted = segmenter.fit(habit.Cohort.from_directory("training_input"))
        prediction = fitted.predict(habit.Cohort.from_directory("external_input"))

    train_config = mock_run.call_args_list[0].args[0]
    predict_config = mock_run.call_args_list[1].args[0]
    assert fitted is segmenter
    assert train_config.run_mode == "train"
    assert train_config.data_dir == "training_input"
    assert predict_config.run_mode == "predict"
    assert predict_config.data_dir == "external_input"
    assert predict_config.out_dir == "prediction_output"
    assert predict_config.pipeline_path == str(
        Path("training_output") / "habitat_pipeline.pkl"
    )
    pd.testing.assert_frame_equal(prediction.table, prediction_table)


@pytest.mark.unit
def test_habitat_segmenter_accepts_omitted_data_dir_with_cohort() -> None:
    """Passing a cohort supplies data_dir when it is absent from the config dict."""
    training_config = _HabitatConfigStub(
        data_dir="from_cohort",
        out_dir="training_output",
    )
    training_table = pd.DataFrame({"subject": ["train_01"], "habitats": [1]})
    training_result = WorkflowResult(
        data=training_table,
        output_dir=Path("training_output"),
    )

    with (
        patch(
            "habit.api.clinical.coerce_config",
            return_value=training_config,
        ) as mock_coerce,
        patch(
            "habit.api.habitat.run_habitat_analysis",
            return_value=training_result,
        ) as mock_run,
    ):
        segmenter = habit.HabitatSegmenter(
            {
                "run_mode": "train",
                "out_dir": "training_output",
            }
        )
        result = segmenter.fit_transform(
            habit.Cohort.from_directory("prepared/processed_images")
        )

    coerced_mapping = mock_coerce.call_args.args[0]
    assert coerced_mapping["data_dir"] == str(
        Path("prepared/processed_images")
    )
    assert mock_run.call_args.args[0].data_dir == str(
        Path("prepared/processed_images")
    )
    pd.testing.assert_frame_equal(result.table, training_table)


@pytest.mark.unit
def test_habitat_segmenter_requires_fit_before_prediction() -> None:
    """Segmenter follows sklearn's fitted-state contract."""
    segmenter = habit.HabitatSegmenter(
        {"data_dir": "input", "out_dir": "output"},
    )

    with pytest.raises(habit.NotFittedError):
        segmenter.predict(habit.Cohort.from_directory("external_input"))


@pytest.mark.unit
def test_outcome_classifier_is_a_clinically_named_sklearn_estimator() -> None:
    """OutcomeClassifier retains HabitClassifier's established estimator contract."""
    classifier = habit.OutcomeClassifier(
        config={
            "run_mode": "train",
            "input": [
                {
                    "path": "training.csv",
                    "subject_id_col": "subject",
                    "label_col": "label",
                }
            ],
            "output": "results",
            "models": {
                "LogisticRegression": {"params": {"solver": "liblinear"}},
            },
            "feature_selection_methods": [],
            "is_visualize": False,
        }
    )

    assert isinstance(classifier, habit.HabitClassifier)


@pytest.mark.integration
def test_clinical_preprocessor_runs_existing_preprocessing_workflow(
    synthetic_preprocess_dataset: tuple[Path, Dict[str, Any]],
) -> None:
    """Object API produces the same prepared cohort through the public runner."""
    data_dir, config = synthetic_preprocess_dataset
    preprocessor = habit.ClinicalPreprocessor(config)

    prepared = preprocessor.fit_transform(
        habit.Cohort.from_directory(data_dir, name="synthetic"),
    )

    assert prepared.data_dir == Path(config["out_dir"]) / "processed_images"
    assert (prepared.data_dir / "images").is_dir()
    assert preprocessor.result_.artifact("output_dir") == Path(config["out_dir"])
