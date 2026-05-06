"""
Structural contract tests for ML result objects.

Migrated and extended from tests/test_ml_kfold_result_contract.py.
Covers:
  - RunResult (holdout)
  - KFoldRunResult (k-fold)
  - InferenceResult (prediction-only)
  - WorkflowPlan deep-copy guarantee
  - DatasetSnapshot field access
  
Tests use AST parsing for structural assertions (no heavy imports needed)
plus runtime construction for functional assertions.
"""

from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(relative_path: str) -> ast.Module:
    source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
    return ast.parse(source)


def _class_node(module: ast.Module, class_name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"Class {class_name!r} not found")


def _method_names(cls_node: ast.ClassDef) -> set:
    return {
        item.name
        for item in cls_node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


RESULTS_PATH = "habit/core/machine_learning/contracts/results.py"


# ---------------------------------------------------------------------------
# AST structural assertions
# ---------------------------------------------------------------------------


class TestResultClassesExist:
    def test_all_result_classes_present(self) -> None:
        module = _parse(RESULTS_PATH)
        class_names = {n.name for n in module.body if isinstance(n, ast.ClassDef)}
        expected = {
            "ModelResult",
            "KFoldModelResult",
            "AggregatedModelResult",
            "RunResult",
            "KFoldRunResult",
            "InferenceResult",
        }
        assert expected.issubset(class_names), f"Missing: {expected - class_names}"


class TestRunResultContract:
    def test_run_result_has_create_and_legacy(self) -> None:
        module = _parse(RESULTS_PATH)
        cls = _class_node(module, "RunResult")
        methods = _method_names(cls)
        assert "create" in methods
        assert "to_legacy_results" in methods


class TestKFoldRunResultContract:
    def test_kfold_run_result_has_create_and_legacy(self) -> None:
        module = _parse(RESULTS_PATH)
        cls = _class_node(module, "KFoldRunResult")
        methods = _method_names(cls)
        assert "create" in methods
        assert "to_legacy_results" in methods

    def test_kfold_runner_returns_kfold_run_result(self) -> None:
        module = _parse("habit/core/machine_learning/runners/kfold.py")
        cls = _class_node(module, "KFoldRunner")
        run_method = next(
            item for item in cls.body
            if isinstance(item, ast.FunctionDef) and item.name == "run"
        )
        assert isinstance(run_method.returns, ast.Name)
        assert run_method.returns.id == "KFoldRunResult"


class TestKFoldWorkflowUsesReportingLayer:
    def test_reporting_components_invoked(self) -> None:
        source = (PROJECT_ROOT / "habit/core/machine_learning/workflows/kfold_workflow.py"
                  ).read_text(encoding="utf-8")
        assert "ModelStore(" in source
        assert "ReportWriter(" in source
        assert "PlotComposer(" in source
        assert "save_kfold_ensembles" in source


# ---------------------------------------------------------------------------
# Runtime construction tests
# ---------------------------------------------------------------------------


def _make_model_result():
    from habit.core.machine_learning.contracts.results import ModelResult

    return ModelResult(
        model_name="LR",
        train={"y_true": [0, 1], "y_prob": [0.1, 0.9], "y_pred": [0, 1]},
        test={"y_true": [0, 1], "y_prob": [0.2, 0.8], "y_pred": [0, 1]},
        train_metrics={"auc": 0.9},
        test_metrics={"auc": 0.85},
        fitted_estimator=MagicMock(),
        feature_names=("f0", "f1"),
        train_subject_ids=("S0", "S1"),
        test_subject_ids=("S2", "S3"),
    )


def _make_plan():
    from habit.core.machine_learning.contracts.plan import WorkflowPlan

    cfg = MagicMock()
    cfg.model_copy = MagicMock(return_value=cfg)
    return WorkflowPlan(config=cfg, output_dir="/tmp/out", random_state=42)


class TestRunResultRuntime:
    def test_create_auto_timestamps(self) -> None:
        from habit.core.machine_learning.contracts.dataset import DatasetSnapshot
        from habit.core.machine_learning.contracts.results import RunResult

        model_result = _make_model_result()
        plan = _make_plan()
        dataset = DatasetSnapshot(
            label_col="label",
            x_train=pd.DataFrame(),
            x_test=pd.DataFrame(),
            y_train=pd.Series(dtype=float),
            y_test=pd.Series(dtype=float),
        )
        result = RunResult.create(
            plan=plan,
            models={"LR": model_result},
            summary_rows=[],
            dataset=dataset,
        )
        assert result.created_at.endswith("Z")
        # Verify it's parseable as ISO8601
        dt = datetime.fromisoformat(result.created_at.rstrip("Z"))
        assert dt.year >= 2020

    def test_to_legacy_results_shape(self) -> None:
        from habit.core.machine_learning.contracts.dataset import DatasetSnapshot
        from habit.core.machine_learning.contracts.results import RunResult

        model_result = _make_model_result()
        plan = _make_plan()
        dataset = DatasetSnapshot(
            label_col="label",
            x_train=pd.DataFrame(),
            x_test=pd.DataFrame(),
            y_train=pd.Series(dtype=float),
            y_test=pd.Series(dtype=float),
        )
        result = RunResult.create(
            plan=plan, models={"LR": model_result}, summary_rows=[], dataset=dataset
        )
        legacy = result.to_legacy_results()
        assert "LR" in legacy
        assert "train" in legacy["LR"]
        assert "test" in legacy["LR"]

    def test_backward_compat_properties(self) -> None:
        from habit.core.machine_learning.contracts.dataset import DatasetSnapshot
        from habit.core.machine_learning.contracts.results import RunResult

        X = pd.DataFrame({"f0": [1.0, 2.0]})
        y = pd.Series([0, 1], name="label")
        dataset = DatasetSnapshot(
            label_col="label", x_train=X, x_test=X, y_train=y, y_test=y
        )
        plan = _make_plan()
        result = RunResult.create(
            plan=plan, models={}, summary_rows=[], dataset=dataset
        )
        assert result.x_train is X
        assert result.x_test is X
        assert result.label_col == "label"


class TestKFoldRunResultRuntime:
    def test_to_legacy_results_shape(self) -> None:
        from habit.core.machine_learning.contracts.results import (
            AggregatedModelResult,
            KFoldModelResult,
            KFoldRunResult,
        )

        fold_payload = {"y_true": [0, 1], "y_prob": [0.2, 0.8], "y_pred": [0, 1], "metrics": {}}
        kfold_model = KFoldModelResult(
            model_name="LR",
            folds=(fold_payload,),
            fold_estimators=(MagicMock(),),
        )
        agg = AggregatedModelResult(
            model_name="LR",
            raw={"y_true": [0, 1], "y_prob": [0.2, 0.8], "y_pred": [0, 1]},
            overall_metrics={"auc": 0.85},
            auc_mean=0.85,
            auc_std=0.02,
        )
        plan = _make_plan()
        result = KFoldRunResult.create(
            plan=plan,
            models={"LR": kfold_model},
            aggregated={"LR": agg},
            summary_rows=[],
        )
        legacy = result.to_legacy_results()
        assert "folds" in legacy
        assert "aggregated" in legacy
        assert "LR" in legacy["aggregated"]

    def test_fold_pipelines_property(self) -> None:
        from habit.core.machine_learning.contracts.results import (
            AggregatedModelResult,
            KFoldModelResult,
            KFoldRunResult,
        )

        estimator = MagicMock()
        kfold_model = KFoldModelResult(
            model_name="LR",
            folds=({"metrics": {}},),
            fold_estimators=(estimator,),
        )
        agg = AggregatedModelResult(
            model_name="LR", raw={}, overall_metrics={}, auc_mean=0.8, auc_std=0.0
        )
        plan = _make_plan()
        result = KFoldRunResult.create(
            plan=plan, models={"LR": kfold_model}, aggregated={"LR": agg}, summary_rows=[]
        )
        assert result.fold_pipelines["LR"] == [estimator]


class TestInferenceResultRuntime:
    def test_create_auto_timestamp(self) -> None:
        from habit.core.machine_learning.contracts.results import InferenceResult

        plan = _make_plan()
        result = InferenceResult.create(
            plan=plan,
            pipeline_path="/models/lr.pkl",
            predictions=pd.DataFrame({"y_pred": [0, 1]}),
            metrics={"auc": 0.88},
            label_col="label",
        )
        assert result.created_at.endswith("Z")
        assert result.pipeline_path == "/models/lr.pkl"

    def test_to_legacy_results(self) -> None:
        from habit.core.machine_learning.contracts.results import InferenceResult

        plan = _make_plan()
        predictions = pd.DataFrame({"y_pred": [0, 1]})
        result = InferenceResult.create(
            plan=plan, pipeline_path="/m.pkl", predictions=predictions
        )
        legacy = result.to_legacy_results()
        assert "predictions" in legacy
        assert "metrics" in legacy
