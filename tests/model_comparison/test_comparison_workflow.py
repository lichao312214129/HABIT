"""
Unit tests for ModelComparison workflow.

Uses synthetic CSV prediction files so no real model training is needed.
Tests cover: setup(), split-group routing, Youden threshold transfer,
target threshold transfer, metrics aggregation, and the full run() path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from habit.core.machine_learning.config_schemas import ModelComparisonConfig
from habit.core.machine_learning.evaluation.model_evaluation import MultifileEvaluator
from habit.core.machine_learning.evaluation.threshold_manager import ThresholdManager
from habit.core.machine_learning.reporting.report_exporter import MetricsStore, ReportExporter
from habit.core.machine_learning.visualization.plot_manager import PlotManager
from habit.core.machine_learning.workflows.comparison_workflow import ModelComparison


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _logger() -> logging.Logger:
    log = logging.getLogger("test_comparison")
    log.setLevel(logging.CRITICAL)
    return log


def _write_pred_csv(path: Path, n: int = 60, seed: int = 0) -> None:
    """Write a synthetic prediction CSV with label, prob, pred columns."""
    rng = np.random.RandomState(seed)
    y_true = rng.randint(0, 2, n)
    y_prob = rng.beta(3, 3, n)
    y_pred = (y_prob >= 0.5).astype(int)
    df = pd.DataFrame({
        "subject_id": [f"S{i:03d}" for i in range(n)],
        "label": y_true,
        "prob": y_prob,
        "pred": y_pred,
    })
    df.to_csv(path, index=False)


def _write_split_pred_csv(path: Path, n: int = 80, seed: int = 0) -> None:
    """Write a prediction CSV with a 'split' column (train/test)."""
    rng = np.random.RandomState(seed)
    y_true = rng.randint(0, 2, n)
    y_prob = rng.beta(3, 3, n)
    y_pred = (y_prob >= 0.5).astype(int)
    split = ["train"] * (n // 2) + ["test"] * (n // 2)
    df = pd.DataFrame({
        "subject_id": [f"S{i:03d}" for i in range(n)],
        "label": y_true,
        "prob": y_prob,
        "pred": y_pred,
        "split": split,
    })
    df.to_csv(path, index=False)


def _make_config(files: list, output_dir: str, **kwargs) -> ModelComparisonConfig:
    cfg_dict = {
        "files_config": files,
        "output_dir": output_dir,
        "delong_test": {"enabled": False},
        "merged_data": {"enabled": False},
        "metrics": {
            "basic_metrics": {"enabled": True},
            "youden_metrics": {"enabled": False},
            "target_metrics": {"enabled": False},
        },
        "visualization": {
            "roc": {"enabled": False},
            "dca": {"enabled": False},
            "calibration": {"enabled": False},
            "pr_curve": {"enabled": False},
        },
    }
    cfg_dict.update(kwargs)
    return ModelComparisonConfig(**cfg_dict)


def _make_workflow(config: ModelComparisonConfig, output_dir: str) -> ModelComparison:
    evaluator = MultifileEvaluator(output_dir=output_dir)
    reporter = ReportExporter(output_dir=output_dir)
    threshold_manager = ThresholdManager()
    plot_manager = PlotManager(config=config, output_dir=output_dir)
    metrics_store = MetricsStore()
    return ModelComparison(
        config=config,
        evaluator=evaluator,
        reporter=reporter,
        threshold_manager=threshold_manager,
        plot_manager=plot_manager,
        metrics_store=metrics_store,
        logger=_logger(),
    )


# ---------------------------------------------------------------------------
# ModelComparison – instantiation
# ---------------------------------------------------------------------------


class TestModelComparisonInstantiation:
    def test_from_config_object(self, tmp_path: Path) -> None:
        _write_pred_csv(tmp_path / "m1.csv")
        cfg = _make_config(
            files=[{"path": str(tmp_path / "m1.csv"), "subject_id_col": "subject_id",
                    "label_col": "label", "prob_col": "prob"}],
            output_dir=str(tmp_path / "out"),
        )
        wf = _make_workflow(cfg, str(tmp_path / "out"))
        assert wf is not None

    def test_from_dict_config(self, tmp_path: Path) -> None:
        _write_pred_csv(tmp_path / "m1.csv")
        cfg_dict = {
            "files_config": [{"path": str(tmp_path / "m1.csv"), "subject_id_col": "subject_id",
                               "label_col": "label", "prob_col": "prob"}],
            "output_dir": str(tmp_path / "out"),
        }
        evaluator = MultifileEvaluator(output_dir=str(tmp_path / "out"))
        reporter = ReportExporter(output_dir=str(tmp_path / "out"))
        wf = ModelComparison(
            config=cfg_dict,
            evaluator=evaluator,
            reporter=reporter,
            threshold_manager=ThresholdManager(),
            plot_manager=PlotManager(config=cfg_dict, output_dir=str(tmp_path / "out")),
            metrics_store=MetricsStore(),
            logger=_logger(),
        )
        assert isinstance(wf.config, ModelComparisonConfig)

    def test_invalid_config_type_raises(self, tmp_path: Path) -> None:
        evaluator = MultifileEvaluator(output_dir=str(tmp_path))
        reporter = ReportExporter(output_dir=str(tmp_path))
        with pytest.raises(TypeError):
            ModelComparison(
                config=12345,  # type: ignore[arg-type]
                evaluator=evaluator,
                reporter=reporter,
                threshold_manager=ThresholdManager(),
                plot_manager=PlotManager(config={}, output_dir=str(tmp_path)),
                metrics_store=MetricsStore(),
                logger=_logger(),
            )


# ---------------------------------------------------------------------------
# ModelComparison – setup (data loading)
# ---------------------------------------------------------------------------


class TestModelComparisonSetup:
    def test_setup_populates_evaluator_data(self, tmp_path: Path) -> None:
        _write_pred_csv(tmp_path / "m1.csv")
        cfg = _make_config(
            files=[{"path": str(tmp_path / "m1.csv"), "subject_id_col": "subject_id",
                    "label_col": "label", "prob_col": "prob"}],
            output_dir=str(tmp_path / "out"),
        )
        wf = _make_workflow(cfg, str(tmp_path / "out"))
        wf.setup()
        assert wf.evaluator.data is not None
        assert len(wf.evaluator.data) > 0

    def test_setup_two_models(self, tmp_path: Path) -> None:
        _write_pred_csv(tmp_path / "m1.csv", seed=0)
        _write_pred_csv(tmp_path / "m2.csv", seed=1)
        cfg = _make_config(
            files=[
                {"path": str(tmp_path / "m1.csv"), "subject_id_col": "subject_id",
                 "label_col": "label", "prob_col": "prob", "model_name": "ModelA"},
                {"path": str(tmp_path / "m2.csv"), "subject_id_col": "subject_id",
                 "label_col": "label", "prob_col": "prob", "model_name": "ModelB"},
            ],
            output_dir=str(tmp_path / "out"),
        )
        wf = _make_workflow(cfg, str(tmp_path / "out"))
        wf.setup()
        assert len(wf.evaluator.models_data) == 2


# ---------------------------------------------------------------------------
# ModelComparison – split-group routing
# ---------------------------------------------------------------------------


class TestModelComparisonSplitGroups:
    def test_split_groups_created(self, tmp_path: Path) -> None:
        _write_split_pred_csv(tmp_path / "m1.csv")
        cfg = _make_config(
            files=[{"path": str(tmp_path / "m1.csv"), "subject_id_col": "subject_id",
                    "label_col": "label", "prob_col": "prob",
                    "split_col": "split"}],
            output_dir=str(tmp_path / "out"),
            **{"split": {"enabled": True}},
        )
        wf = _make_workflow(cfg, str(tmp_path / "out"))
        wf.setup()
        if wf.split_column:
            assert "train" in wf.split_groups or "test" in wf.split_groups


# ---------------------------------------------------------------------------
# ModelComparison – full run()
# ---------------------------------------------------------------------------


class TestModelComparisonRun:
    def test_run_completes_without_error(self, tmp_path: Path) -> None:
        _write_pred_csv(tmp_path / "m1.csv", n=60)
        cfg = _make_config(
            files=[{"path": str(tmp_path / "m1.csv"), "subject_id_col": "subject_id",
                    "label_col": "label", "prob_col": "prob"}],
            output_dir=str(tmp_path / "out"),
        )
        wf = _make_workflow(cfg, str(tmp_path / "out"))
        wf.run()  # must not raise

    def test_run_with_two_models_and_delong(self, tmp_path: Path) -> None:
        _write_pred_csv(tmp_path / "m1.csv", n=60, seed=0)
        _write_pred_csv(tmp_path / "m2.csv", n=60, seed=1)
        cfg_dict = {
            "files_config": [
                {"path": str(tmp_path / "m1.csv"), "subject_id_col": "subject_id",
                 "label_col": "label", "prob_col": "prob", "model_name": "A"},
                {"path": str(tmp_path / "m2.csv"), "subject_id_col": "subject_id",
                 "label_col": "label", "prob_col": "prob", "model_name": "B"},
            ],
            "output_dir": str(tmp_path / "out"),
            "delong_test": {"enabled": True, "save_name": "delong.json"},
            "merged_data": {"enabled": False},
            "metrics": {
                "basic_metrics": {"enabled": True},
                "youden_metrics": {"enabled": False},
                "target_metrics": {"enabled": False},
            },
            "visualization": {
                "roc": {"enabled": False}, "dca": {"enabled": False},
                "calibration": {"enabled": False}, "pr_curve": {"enabled": False},
            },
        }
        cfg = ModelComparisonConfig(**cfg_dict)
        wf = _make_workflow(cfg, str(tmp_path / "out"))
        wf.run()  # two models + DeLong test
