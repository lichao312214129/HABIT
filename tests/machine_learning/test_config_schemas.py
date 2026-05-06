"""
Unit tests for MLConfig (machine learning configuration schema).

Covers required fields, defaults, field validators, resampling config
migration from legacy 'sampling' key, and model-level cross-validators.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from habit.core.common.configs.base import ConfigValidationError
from habit.core.machine_learning.config_schemas import (
    InputFileConfig,
    MLConfig,
    NormalizationConfig,
    ResamplingConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_dict(tmp_path=None) -> dict:
    return {
        "input": [
            {
                "path": "dummy.csv",
                "subject_id_col": "sid",
                "label_col": "label",
            }
        ],
        "output": "./out",
        "feature_selection_methods": [],
        "models": {"LogisticRegression": {"params": {"max_iter": 100}}},
    }


# ---------------------------------------------------------------------------
# InputFileConfig
# ---------------------------------------------------------------------------


class TestInputFileConfig:
    def test_minimal_valid(self) -> None:
        cfg = InputFileConfig(path="a.csv", subject_id_col="sid", label_col="y")
        assert cfg.path == "a.csv"
        assert cfg.features is None

    def test_missing_path_raises(self) -> None:
        with pytest.raises(ValidationError):
            InputFileConfig(subject_id_col="sid", label_col="y")

    def test_optional_fields_default_none(self) -> None:
        cfg = InputFileConfig(path="a.csv", subject_id_col="sid", label_col="y")
        assert cfg.split_col is None
        assert cfg.pred_col is None
        assert cfg.features_from_log is None


# ---------------------------------------------------------------------------
# NormalizationConfig
# ---------------------------------------------------------------------------


class TestNormalizationConfig:
    def test_default_method_zscore(self) -> None:
        cfg = NormalizationConfig()
        assert cfg.method == "z_score"

    def test_valid_methods(self) -> None:
        for method in ["z_score", "min_max", "robust", "max_abs", "normalizer", "quantile", "power"]:
            cfg = NormalizationConfig(method=method)
            assert cfg.method == method

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValidationError):
            NormalizationConfig(method="unknown_method")


# ---------------------------------------------------------------------------
# ResamplingConfig
# ---------------------------------------------------------------------------


class TestResamplingConfig:
    def test_defaults(self) -> None:
        cfg = ResamplingConfig()
        assert cfg.enabled is False
        assert cfg.method == "random_over"
        assert cfg.ratio == 1.0

    def test_ratio_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="ratio must be > 0"):
            ResamplingConfig(ratio=0.0)

    def test_ratio_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            ResamplingConfig(ratio=-0.5)

    def test_valid_positions(self) -> None:
        for pos in [
            "before_feature_selection",
            "before_normalization",
            "after_normalization",
            "before_model",
        ]:
            cfg = ResamplingConfig(position=pos)
            assert cfg.position == pos

    def test_invalid_position_raises(self) -> None:
        with pytest.raises(ValidationError):
            ResamplingConfig(position="after_model")


# ---------------------------------------------------------------------------
# MLConfig
# ---------------------------------------------------------------------------


class TestMLConfig:
    def test_minimal_valid(self) -> None:
        cfg = MLConfig.model_validate(_minimal_dict())
        assert len(cfg.input) == 1
        assert "LogisticRegression" in cfg.models

    def test_output_required(self) -> None:
        d = _minimal_dict()
        d.pop("output")
        with pytest.raises((ValidationError, ConfigValidationError)):
            MLConfig.model_validate(d)

    def test_resampling_block_accepted(self) -> None:
        d = _minimal_dict()
        d["resampling"] = {
            "enabled": True,
            "method": "random_over",
            "position": "after_normalization",
            "ratio": 0.75,
            "random_state": 7,
        }
        cfg = MLConfig.model_validate(d)
        assert cfg.resampling.enabled is True
        assert cfg.resampling.position == "after_normalization"
        assert cfg.resampling.ratio == 0.75
        assert cfg.resampling.random_state == 7

    def test_legacy_sampling_key_migrated(self) -> None:
        """Existing YAML files using 'sampling' key should still load."""
        d = _minimal_dict()
        d["sampling"] = {
            "enabled": True,
            "method": "random_under",
            "position": "before_model",
            "ratio": 0.5,
        }
        cfg = MLConfig.model_validate(d)
        assert cfg.resampling.enabled is True
        assert cfg.resampling.method == "random_under"
        assert cfg.resampling.position == "before_model"

    def test_run_mode_default_train(self) -> None:
        cfg = MLConfig.model_validate(_minimal_dict())
        assert getattr(cfg, "run_mode", "train") == "train"

    def test_test_size_default(self) -> None:
        cfg = MLConfig.model_validate(_minimal_dict())
        assert 0 < cfg.test_size < 1

    def test_split_method_default(self) -> None:
        cfg = MLConfig.model_validate(_minimal_dict())
        assert cfg.split_method in ("stratified", "random", "custom")

    def test_multiple_input_files(self) -> None:
        d = _minimal_dict()
        d["input"].append({
            "path": "second.csv",
            "subject_id_col": "sid",
            "label_col": "label",
        })
        cfg = MLConfig.model_validate(d)
        assert len(cfg.input) == 2

    def test_feature_selection_methods_list(self) -> None:
        d = _minimal_dict()
        d["feature_selection_methods"] = [{"method": "variance", "params": {"threshold": 0.01}}]
        cfg = MLConfig.model_validate(d)
        assert len(cfg.feature_selection_methods) == 1

    def test_is_save_model_default_true(self) -> None:
        cfg = MLConfig.model_validate(_minimal_dict())
        assert getattr(cfg, "is_save_model", True) is True
