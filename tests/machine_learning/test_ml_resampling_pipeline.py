"""Tests for configurable ML resampling pipeline placement."""

from __future__ import annotations

import pytest

from habit.core.machine_learning.config_schemas import MLConfig
from habit.core.machine_learning.pipeline_builder import PipelineBuilder


def _base_config_dict() -> dict:
    """Return the smallest valid training config needed to build pipelines."""
    return {
        "input": [
            {
                "path": "dummy.csv",
                "subject_id_col": "subjID",
                "label_col": "label",
            }
        ],
        "output": "./tmp",
        "feature_selection_methods": [],
        "models": {
            "LogisticRegression": {
                "params": {
                    "max_iter": 100,
                }
            }
        },
    }


def test_ml_config_accepts_resampling_position() -> None:
    """The public config should expose the new resampling block."""
    raw_config = _base_config_dict()
    raw_config["resampling"] = {
        "enabled": True,
        "method": "random_over",
        "position": "after_normalization",
        "ratio": 0.75,
        "random_state": 7,
    }

    config = MLConfig.model_validate(raw_config)

    assert config.resampling.enabled is True
    assert config.resampling.position == "after_normalization"
    assert config.resampling.ratio == 0.75
    assert config.resampling.random_state == 7


def test_ml_config_migrates_legacy_sampling_key() -> None:
    """Existing YAML files using sampling should keep loading."""
    raw_config = _base_config_dict()
    raw_config["sampling"] = {
        "enabled": True,
        "method": "random_under",
        "position": "before_model",
        "ratio": 0.5,
    }

    config = MLConfig.model_validate(raw_config)

    assert config.resampling.enabled is True
    assert config.resampling.method == "random_under"
    assert config.resampling.position == "before_model"
    assert config.resampling.ratio == 0.5


def test_pipeline_builder_inserts_resampling_before_model() -> None:
    """Enabled resampling should be part of the fitted pipeline, not the runner."""
    pytest.importorskip("imblearn")
    raw_config = _base_config_dict()
    raw_config["resampling"] = {
        "enabled": True,
        "method": "random_over",
        "position": "before_model",
    }
    config = MLConfig.model_validate(raw_config)

    pipeline = PipelineBuilder(config).build(
        "LogisticRegression",
        {"max_iter": 100},
        feature_names=["feature_1", "feature_2"],
    )

    assert [name for name, _ in pipeline.steps] == [
        "selector_before",
        "scaler",
        "selector_after",
        "resampler",
        "model",
    ]
