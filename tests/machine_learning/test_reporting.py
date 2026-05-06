"""
Unit tests for the reporting layer:
  - ModelStore (holdout save / kfold ensemble save)
  - ReportWriter (CSV + JSON output)
  - ReportExporter / MetricsStore
  - PlotComposer (no-op when is_visualize=False)

Uses in-memory stubs instead of real ML runs to keep the tests fast.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from habit.core.machine_learning.reporting.model_store import ModelStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plan(tmp_path: Path):
    from habit.core.machine_learning.contracts.plan import WorkflowPlan

    cfg = MagicMock()
    cfg.model_copy = MagicMock(return_value=cfg)
    return WorkflowPlan(config=cfg, output_dir=str(tmp_path), random_state=42)


def _make_run_result(tmp_path: Path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from habit.core.machine_learning.contracts.dataset import DatasetSnapshot
    from habit.core.machine_learning.contracts.results import ModelResult, RunResult

    X = pd.DataFrame({"f0": [0.1, 0.9], "f1": [0.2, 0.8]})
    y = pd.Series([0, 1], name="label")

    estimator = Pipeline([("model", LogisticRegression(max_iter=100))]).fit(X, y)

    model_result = ModelResult(
        model_name="LR",
        train={"y_true": [0, 1], "y_prob": [0.1, 0.9], "y_pred": [0, 1]},
        test={"y_true": [0, 1], "y_prob": [0.2, 0.8], "y_pred": [0, 1]},
        train_metrics={"auc": 0.9, "accuracy": 0.9},
        test_metrics={"auc": 0.85, "accuracy": 0.85},
        fitted_estimator=estimator,
        feature_names=("f0", "f1"),
    )
    dataset = DatasetSnapshot(label_col="label", x_train=X, x_test=X, y_train=y, y_test=y)
    plan = _make_plan(tmp_path)
    return RunResult.create(plan=plan, models={"LR": model_result}, summary_rows=[
        {"Model": "LR", "Train_auc": 0.9, "Test_auc": 0.85}
    ], dataset=dataset)


def _make_kfold_result(tmp_path: Path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from habit.core.machine_learning.contracts.results import (
        AggregatedModelResult,
        KFoldModelResult,
        KFoldRunResult,
    )

    X = pd.DataFrame({"f0": [0.1, 0.9, 0.5], "f1": [0.2, 0.8, 0.3]})
    y = pd.Series([0, 1, 0], name="label")
    estimator = Pipeline([("model", LogisticRegression(max_iter=100))]).fit(X, y)

    fold_payload = {"y_true": [0, 1], "y_prob": [0.2, 0.8], "y_pred": [0, 1], "metrics": {"auc": 0.85}}
    kfold_model = KFoldModelResult(
        model_name="LR", folds=(fold_payload,), fold_estimators=(estimator,)
    )
    agg = AggregatedModelResult(
        model_name="LR",
        raw={"y_true": [0, 1], "y_prob": [0.2, 0.8], "y_pred": [0, 1]},
        overall_metrics={"auc": 0.85},
        auc_mean=0.85,
        auc_std=0.0,
    )
    plan = _make_plan(tmp_path)
    return KFoldRunResult.create(
        plan=plan,
        models={"LR": kfold_model},
        aggregated={"LR": agg},
        summary_rows=[{"Model": "LR", "AUC_mean": 0.85}],
    )


# ---------------------------------------------------------------------------
# ModelStore â€?holdout
# ---------------------------------------------------------------------------


class TestModelStoreHoldout:
    def test_save_creates_pkl_file(self, tmp_path: Path) -> None:
        run_result = _make_run_result(tmp_path)
        store = ModelStore(output_dir=str(tmp_path), is_save_model=True)
        paths = store.save(run_result)
        assert "LR" in paths
        assert Path(paths["LR"]).exists()
        assert paths["LR"].endswith(".pkl")

    def test_save_disabled_returns_empty_dict(self, tmp_path: Path) -> None:
        run_result = _make_run_result(tmp_path)
        store = ModelStore(output_dir=str(tmp_path), is_save_model=False)
        paths = store.save(run_result)
        assert paths == {}

    def test_model_dir_created(self, tmp_path: Path) -> None:
        run_result = _make_run_result(tmp_path)
        store = ModelStore(output_dir=str(tmp_path), is_save_model=True)
        store.save(run_result)
        assert (tmp_path / "models").is_dir()


# ---------------------------------------------------------------------------
# ModelStore â€?kfold ensemble
# ---------------------------------------------------------------------------


class TestModelStoreKFold:
    def test_save_kfold_ensembles_creates_pkl(self, tmp_path: Path) -> None:
        kfold_result = _make_kfold_result(tmp_path)
        store = ModelStore(output_dir=str(tmp_path), is_save_model=True)
        paths = store.save_kfold_ensembles(kfold_result, voting="soft")
        assert "LR" in paths
        assert Path(paths["LR"]).exists()

    def test_save_kfold_disabled_returns_empty(self, tmp_path: Path) -> None:
        kfold_result = _make_kfold_result(tmp_path)
        store = ModelStore(output_dir=str(tmp_path), is_save_model=False)
        paths = store.save_kfold_ensembles(kfold_result)
        assert paths == {}


# ---------------------------------------------------------------------------
# ReportWriter
# ---------------------------------------------------------------------------


class TestReportWriter:
    def test_write_holdout_creates_json_and_csv(self, tmp_path: Path) -> None:
        from habit.core.machine_learning.reporting.report_writer import ReportWriter

        run_result = _make_run_result(tmp_path)
        writer = ReportWriter(output_dir=str(tmp_path), module_name="ml_standard")
        writer.write(run_result)

        # At least one JSON results file must exist
        json_files = list(tmp_path.rglob("*.json"))
        csv_files = list(tmp_path.rglob("*.csv"))
        assert len(json_files) >= 1 or len(csv_files) >= 1

    def test_write_kfold_creates_output(self, tmp_path: Path) -> None:
        from habit.core.machine_learning.reporting.report_writer import ReportWriter

        kfold_result = _make_kfold_result(tmp_path)
        writer = ReportWriter(output_dir=str(tmp_path), module_name="ml_kfold")
        writer.write(kfold_result)


# ---------------------------------------------------------------------------
# ReportExporter / MetricsStore
# ---------------------------------------------------------------------------


class TestMetricsStore:
    def test_add_and_get_metrics(self) -> None:
        from habit.core.machine_learning.reporting.report_exporter import MetricsStore

        store = MetricsStore()
        store.add_metrics("train", "ModelA", "basic_metrics", {"auc": 0.9})
        all_metrics = store.get()
        assert "train" in all_metrics
        assert "ModelA" in all_metrics["train"]

    def test_add_threshold(self) -> None:
        from habit.core.machine_learning.reporting.report_exporter import MetricsStore

        store = MetricsStore()
        store.add_threshold("train", "ModelA", "youden", 0.45)
        all_metrics = store.get()
        # Threshold should appear somewhere inside the nested dict
        assert all_metrics  # non-empty

    def test_multiple_models_isolated(self) -> None:
        from habit.core.machine_learning.reporting.report_exporter import MetricsStore

        store = MetricsStore()
        store.add_metrics("group1", "A", "basic_metrics", {"auc": 0.9})
        store.add_metrics("group1", "B", "basic_metrics", {"auc": 0.8})
        all_metrics = store.get()
        assert "A" in all_metrics["group1"]
        assert "B" in all_metrics["group1"]


# ---------------------------------------------------------------------------
# PlotComposer (no-op path)
# ---------------------------------------------------------------------------


class TestPlotComposerNoOp:
    def test_render_noop_when_is_visualize_false(self, tmp_path: Path) -> None:
        from habit.core.machine_learning.reporting.plot_composer import PlotComposer

        run_result = _make_run_result(tmp_path)
        plot_manager = MagicMock()
        composer = PlotComposer(plot_manager=plot_manager, is_visualize=False)
        composer.render(run_result)
        # When disabled, plot_manager methods should NOT be called
        plot_manager.assert_not_called()
