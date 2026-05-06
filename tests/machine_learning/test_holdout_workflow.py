"""
Integration tests for HoldoutWorkflow (train + predict modes).

Uses demo_data/ml_data/breast_cancer_dataset.csv when available;
falls back to a temporary synthetic CSV when the demo file is absent.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from habit.core.common.configs.base import ConfigValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BREAST_CANCER_CSV = PROJECT_ROOT / "demo_data" / "ml_data" / "breast_cancer_dataset.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_synthetic_csv(path: Path, n: int = 80) -> None:
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "subject_id": [f"S{i:03d}" for i in range(n)],
            **{f"feature_{j}": rng.randn(n) for j in range(10)},
            "label": rng.randint(0, 2, n),
        }
    )
    df.to_csv(path, index=False)


def _get_csv_path(tmp_path: Path) -> str:
    if BREAST_CANCER_CSV.exists():
        return str(BREAST_CANCER_CSV)
    synthetic = tmp_path / "synth.csv"
    _write_synthetic_csv(synthetic)
    return str(synthetic)


def _subject_and_label_cols(csv_path: str) -> tuple[str, str]:
    """
    Demo breast cancer CSV uses ``subjID``; synthetic fixtures use ``subject_id``.
    """
    name = Path(csv_path).name
    if "breast_cancer_dataset" in name:
        return "subjID", "label"
    return "subject_id", "label"


def _make_config_dict(csv_path: str, output_dir: str) -> dict:
    sid_col, lbl_col = _subject_and_label_cols(csv_path)
    return {
        "input": [{"path": csv_path, "subject_id_col": sid_col, "label_col": lbl_col}],
        "output": output_dir,
        "feature_selection_methods": [{"method": "variance", "params": {"threshold": 0.0}}],
        "models": {"LogisticRegression": {"params": {"max_iter": 200}}},
        "is_save_model": False,
        "is_visualize": False,
        "test_size": 0.3,
        "split_method": "stratified",
        "random_state": 42,
    }


# ---------------------------------------------------------------------------
# HoldoutWorkflow – train mode
# ---------------------------------------------------------------------------


class TestHoldoutWorkflowTrain:
    def test_fit_completes_without_error(self, tmp_path: Path) -> None:
        from habit.core.machine_learning.config_schemas import MLConfig
        from habit.core.machine_learning.workflows.holdout_workflow import HoldoutWorkflow

        csv = _get_csv_path(tmp_path)
        cfg = MLConfig.model_validate(_make_config_dict(csv, str(tmp_path / "out")))
        wf = HoldoutWorkflow(config=cfg)
        wf.fit()  # must not raise

    def test_fit_populates_run_result(self, tmp_path: Path) -> None:
        from habit.core.machine_learning.config_schemas import MLConfig
        from habit.core.machine_learning.workflows.holdout_workflow import HoldoutWorkflow

        csv = _get_csv_path(tmp_path)
        cfg = MLConfig.model_validate(_make_config_dict(csv, str(tmp_path / "out")))
        wf = HoldoutWorkflow(config=cfg)
        wf.fit()
        assert wf._run_result is not None
        assert "LogisticRegression" in wf._run_result.models

    def test_fit_produces_auc_metrics(self, tmp_path: Path) -> None:
        from habit.core.machine_learning.config_schemas import MLConfig
        from habit.core.machine_learning.workflows.holdout_workflow import HoldoutWorkflow

        csv = _get_csv_path(tmp_path)
        cfg = MLConfig.model_validate(_make_config_dict(csv, str(tmp_path / "out")))
        wf = HoldoutWorkflow(config=cfg)
        wf.fit()
        model_result = wf._run_result.models["LogisticRegression"]
        assert "auc" in model_result.train_metrics
        assert "auc" in model_result.test_metrics
        assert 0.0 <= model_result.test_metrics["auc"] <= 1.0

    def test_fit_saves_model_pkl(self, tmp_path: Path) -> None:
        from habit.core.machine_learning.config_schemas import MLConfig
        from habit.core.machine_learning.workflows.holdout_workflow import HoldoutWorkflow

        csv = _get_csv_path(tmp_path)
        config_dict = _make_config_dict(csv, str(tmp_path / "out"))
        config_dict["is_save_model"] = True
        cfg = MLConfig.model_validate(config_dict)
        wf = HoldoutWorkflow(config=cfg)
        wf.fit()
        pkl_files = list(Path(cfg.output).rglob("*.pkl"))
        assert len(pkl_files) >= 1

    def test_fit_writes_summary_csv(self, tmp_path: Path) -> None:
        from habit.core.machine_learning.config_schemas import MLConfig
        from habit.core.machine_learning.workflows.holdout_workflow import HoldoutWorkflow

        csv = _get_csv_path(tmp_path)
        out = tmp_path / "out"
        cfg = MLConfig.model_validate(_make_config_dict(csv, str(out)))
        wf = HoldoutWorkflow(config=cfg)
        wf.fit()
        csv_files = list(out.rglob("*summary*.csv"))
        assert len(csv_files) >= 1

    def test_legacy_results_dict_populated(self, tmp_path: Path) -> None:
        from habit.core.machine_learning.config_schemas import MLConfig
        from habit.core.machine_learning.workflows.holdout_workflow import HoldoutWorkflow

        csv = _get_csv_path(tmp_path)
        cfg = MLConfig.model_validate(_make_config_dict(csv, str(tmp_path / "out")))
        wf = HoldoutWorkflow(config=cfg)
        wf.fit()
        assert isinstance(wf.results, dict)
        assert "LogisticRegression" in wf.results

    def test_run_dispatches_to_fit(self, tmp_path: Path) -> None:
        from habit.core.machine_learning.config_schemas import MLConfig
        from habit.core.machine_learning.workflows.holdout_workflow import HoldoutWorkflow

        csv = _get_csv_path(tmp_path)
        config_dict = _make_config_dict(csv, str(tmp_path / "out"))
        config_dict["run_mode"] = "train"
        cfg = MLConfig.model_validate(config_dict)
        wf = HoldoutWorkflow(config=cfg)
        wf.run()  # should call fit()
        assert wf._run_result is not None


# ---------------------------------------------------------------------------
# HoldoutWorkflow – predict mode
# ---------------------------------------------------------------------------


class TestHoldoutWorkflowPredict:
    def test_predict_mode_requires_pipeline_path(self, tmp_path: Path) -> None:
        from habit.core.machine_learning.config_schemas import MLConfig

        csv = _get_csv_path(tmp_path)
        config_dict = _make_config_dict(csv, str(tmp_path / "out"))
        config_dict["run_mode"] = "predict"
        # Missing pipeline_path: MLConfig rejects at validation time (BaseConfig wraps errors).
        with pytest.raises((ValidationError, ConfigValidationError)):
            MLConfig.model_validate(config_dict)

    def test_predict_after_train(self, tmp_path: Path) -> None:
        """Train first, then run predict mode using the saved pipeline."""
        from habit.core.machine_learning.config_schemas import MLConfig
        from habit.core.machine_learning.workflows.holdout_workflow import HoldoutWorkflow

        csv = _get_csv_path(tmp_path)
        out = tmp_path / "out"

        # Train
        config_dict = _make_config_dict(csv, str(out))
        config_dict["is_save_model"] = True
        cfg = MLConfig.model_validate(config_dict)
        wf = HoldoutWorkflow(config=cfg)
        wf.fit()

        # Locate saved pipeline
        pkl_files = list(out.rglob("*.pkl"))
        if not pkl_files:
            pytest.skip("No pkl file saved; model storage may be disabled")

        # Predict
        config_dict_pred = _make_config_dict(csv, str(tmp_path / "pred_out"))
        config_dict_pred["run_mode"] = "predict"
        config_dict_pred["pipeline_path"] = str(pkl_files[0])
        cfg_pred = MLConfig.model_validate(config_dict_pred)
        wf_pred = HoldoutWorkflow(config=cfg_pred)
        wf_pred.predict()  # must not raise
        assert wf_pred._inference_result is not None


# ---------------------------------------------------------------------------
# MachineLearningWorkflow deprecation shim
# ---------------------------------------------------------------------------


class TestMachineLearningWorkflowDeprecationShim:
    def test_deprecated_class_raises_warning(self, tmp_path: Path) -> None:
        import warnings

        from habit.core.machine_learning.config_schemas import MLConfig
        from habit.core.machine_learning.workflows.holdout_workflow import (
            MachineLearningWorkflow,
        )

        csv = _get_csv_path(tmp_path)
        config_dict = _make_config_dict(csv, str(tmp_path / "out"))
        config_dict["run_mode"] = "train"
        cfg = MLConfig.model_validate(config_dict)
        wf = MachineLearningWorkflow(config=cfg)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wf.run()
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
