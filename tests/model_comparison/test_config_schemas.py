"""
Unit tests for ModelComparisonConfig and its nested schemas.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from habit.core.common.configs.base import ConfigValidationError
from habit.core.machine_learning.config_schemas import (
    ComparisonFileConfig,
    ModelComparisonConfig,
)


# ---------------------------------------------------------------------------
# ComparisonFileConfig
# ---------------------------------------------------------------------------


class TestComparisonFileConfig:
    def test_minimal_valid(self) -> None:
        cfg = ComparisonFileConfig(
            path="preds.csv",
            subject_id_col="sid",
            label_col="label",
            prob_col="prob",
        )
        assert cfg.path == "preds.csv"

    def test_model_name_inferred_from_path(self) -> None:
        cfg = ComparisonFileConfig(
            path="/data/model_lr.csv",
            subject_id_col="sid",
            label_col="label",
            prob_col="prob",
        )
        assert cfg.model_name == "model_lr"

    def test_model_name_explicit_overrides_path(self) -> None:
        cfg = ComparisonFileConfig(
            path="/data/model_lr.csv",
            model_name="MyLR",
            subject_id_col="sid",
            label_col="label",
            prob_col="prob",
        )
        assert cfg.model_name == "MyLR"

    def test_name_alias_used_as_model_name(self) -> None:
        cfg = ComparisonFileConfig(
            path="/data/m.csv",
            name="AliasModel",
            subject_id_col="sid",
            label_col="label",
            prob_col="prob",
        )
        assert cfg.model_name == "AliasModel"

    def test_missing_required_fields_raises(self) -> None:
        with pytest.raises(ValidationError):
            ComparisonFileConfig(path="a.csv")  # missing subject_id_col, label_col, prob_col


# ---------------------------------------------------------------------------
# ModelComparisonConfig
# ---------------------------------------------------------------------------


class TestModelComparisonConfig:
    def _minimal(self) -> dict:
        return {
            "files_config": [
                {
                    "path": "m1.csv",
                    "subject_id_col": "sid",
                    "label_col": "label",
                    "prob_col": "prob",
                }
            ],
            "output_dir": "./out",
        }

    def test_minimal_valid(self) -> None:
        cfg = ModelComparisonConfig(**self._minimal())
        assert len(cfg.files_config) == 1
        assert cfg.output_dir == "./out"

    def test_multiple_files(self) -> None:
        d = self._minimal()
        d["files_config"].append({
            "path": "m2.csv",
            "subject_id_col": "sid",
            "label_col": "label",
            "prob_col": "prob",
        })
        cfg = ModelComparisonConfig(**d)
        assert len(cfg.files_config) == 2

    def test_delong_test_default_enabled(self) -> None:
        cfg = ModelComparisonConfig(**self._minimal())
        assert cfg.delong_test.enabled is True

    def test_split_default_disabled(self) -> None:
        cfg = ModelComparisonConfig(**self._minimal())
        assert cfg.split.enabled is False

    def test_target_metrics_values_must_be_in_0_1(self) -> None:
        d = self._minimal()
        d["metrics"] = {
            "target_metrics": {
                "enabled": True,
                "targets": {"sensitivity": 1.5},  # invalid: > 1
            }
        }
        with pytest.raises((ValidationError, ConfigValidationError), match="between 0 and 1"):
            ModelComparisonConfig(**d)

    def test_valid_target_metrics(self) -> None:
        d = self._minimal()
        d["metrics"] = {
            "target_metrics": {
                "enabled": True,
                "targets": {"sensitivity": 0.85, "specificity": 0.80},
            }
        }
        cfg = ModelComparisonConfig(**d)
        assert cfg.metrics.target_metrics.enabled is True
        assert cfg.metrics.target_metrics.targets["sensitivity"] == 0.85

    def test_visualization_defaults(self) -> None:
        cfg = ModelComparisonConfig(**self._minimal())
        assert cfg.visualization.roc.enabled is True
        assert cfg.visualization.dca.enabled is True
        assert cfg.visualization.calibration.enabled is True

    def test_missing_files_config_raises(self) -> None:
        with pytest.raises((ValidationError, ConfigValidationError, TypeError)):
            ModelComparisonConfig(output_dir="./out")

    def test_missing_output_dir_raises(self) -> None:
        with pytest.raises((ValidationError, ConfigValidationError, TypeError)):
            ModelComparisonConfig(files_config=[{
                "path": "m1.csv", "subject_id_col": "s", "label_col": "l", "prob_col": "p"
            }])
