"""
Unit tests for PipelineBuilder and resampling position injection.

Migrated and extended from tests/test_ml_resampling_pipeline.py.
"""

from __future__ import annotations

import pytest

from habit.core.machine_learning.config_schemas import MLConfig
from habit.core.machine_learning.pipeline_builder import PipelineBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_config_dict() -> dict:
    return {
        "input": [{"path": "dummy.csv", "subject_id_col": "sid", "label_col": "label"}],
        "output": "./tmp",
        "feature_selection_methods": [],
        "models": {"LogisticRegression": {"params": {"max_iter": 100}}},
    }


def _build(config: MLConfig, model_name: str = "LogisticRegression",
           params: dict = None, n_features: int = 5):
    params = params or {"max_iter": 100}
    feature_names = [f"feature_{i}" for i in range(n_features)]
    return PipelineBuilder(config).build(model_name, params, feature_names=feature_names)


# ---------------------------------------------------------------------------
# Config schema – resampling acceptance
# ---------------------------------------------------------------------------


class TestMLConfigResampling:
    def test_resampling_block_accepted(self) -> None:
        d = _base_config_dict()
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
        d = _base_config_dict()
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


# ---------------------------------------------------------------------------
# PipelineBuilder – step order without resampling
# ---------------------------------------------------------------------------


class TestPipelineBuilderNoResampling:
    def test_base_pipeline_steps(self) -> None:
        cfg = MLConfig.model_validate(_base_config_dict())
        pipeline = _build(cfg)
        step_names = [name for name, _ in pipeline.steps]
        assert "model" in step_names
        assert "scaler" in step_names

    def test_returns_sklearn_pipeline(self) -> None:
        from sklearn.pipeline import Pipeline

        cfg = MLConfig.model_validate(_base_config_dict())
        pipeline = _build(cfg)
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_has_final_step_model(self) -> None:
        cfg = MLConfig.model_validate(_base_config_dict())
        pipeline = _build(cfg)
        assert pipeline.steps[-1][0] == "model"


# ---------------------------------------------------------------------------
# PipelineBuilder – resampling position: before_model
# ---------------------------------------------------------------------------


class TestPipelineBuilderResamplingBeforeModel:
    def test_resampler_inserted_before_model(self) -> None:
        pytest.importorskip("imblearn")
        d = _base_config_dict()
        d["resampling"] = {"enabled": True, "method": "random_over", "position": "before_model"}
        cfg = MLConfig.model_validate(d)
        pipeline = _build(cfg)
        step_names = [name for name, _ in pipeline.steps]
        resampler_idx = step_names.index("resampler")
        model_idx = step_names.index("model")
        assert resampler_idx < model_idx

    def test_expected_step_order(self) -> None:
        pytest.importorskip("imblearn")
        d = _base_config_dict()
        d["resampling"] = {"enabled": True, "method": "random_over", "position": "before_model"}
        cfg = MLConfig.model_validate(d)
        pipeline = _build(cfg)
        step_names = [name for name, _ in pipeline.steps]
        assert step_names == [
            "selector_before", "scaler", "selector_after", "resampler", "model"
        ]


# ---------------------------------------------------------------------------
# PipelineBuilder – resampling position: after_normalization
# ---------------------------------------------------------------------------


class TestPipelineBuilderResamplingAfterNormalization:
    def test_resampler_after_scaler(self) -> None:
        pytest.importorskip("imblearn")
        d = _base_config_dict()
        d["resampling"] = {
            "enabled": True, "method": "random_over", "position": "after_normalization"
        }
        cfg = MLConfig.model_validate(d)
        pipeline = _build(cfg)
        step_names = [name for name, _ in pipeline.steps]
        scaler_idx = step_names.index("scaler")
        resampler_idx = step_names.index("resampler")
        assert resampler_idx > scaler_idx


# ---------------------------------------------------------------------------
# PipelineBuilder – feature selection methods
# ---------------------------------------------------------------------------


class TestPipelineBuilderFeatureSelection:
    def test_variance_selector_in_pipeline(self) -> None:
        d = _base_config_dict()
        d["feature_selection_methods"] = [{"method": "variance", "params": {"threshold": 0.0}}]
        cfg = MLConfig.model_validate(d)
        pipeline = _build(cfg)
        step_names = [name for name, _ in pipeline.steps]
        # At least one selector step must be present
        has_selector = any("selector" in n for n in step_names)
        assert has_selector

    def test_disabled_resampling_no_resampler_step(self) -> None:
        pytest.importorskip("imblearn")
        d = _base_config_dict()
        d["resampling"] = {"enabled": False, "method": "random_over", "position": "before_model"}
        cfg = MLConfig.model_validate(d)
        pipeline = _build(cfg)
        step_names = [name for name, _ in pipeline.steps]
        assert "resampler" not in step_names
