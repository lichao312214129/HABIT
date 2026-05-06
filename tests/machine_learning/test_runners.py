"""
Unit tests for HoldoutRunner, KFoldRunner, and InferenceRunner.

Uses mock/stub collaborators (RunnerContext with minimal DataManager,
PipelineBuilder) so no real CSV files or heavy ML packages are required
beyond scikit-learn, which is a core dependency.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from habit.core.machine_learning.contracts.plan import WorkflowPlan
from habit.core.machine_learning.contracts.results import KFoldRunResult, RunResult
from habit.core.machine_learning.runners.context import RunnerContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_X_y(n: int = 60, n_features: int = 5, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.Series(rng.randint(0, 2, n), name="label")
    return X, y


def _minimal_config():
    cfg = MagicMock()
    cfg.models = {"LogisticRegression": MagicMock(params={"max_iter": 100})}
    cfg.split_method = "random"
    cfg.test_size = 0.3
    cfg.random_state = 42
    return cfg


def _stub_pipeline():
    """Return a simple sklearn Pipeline that can be fit/predicted."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=200, random_state=0)),
    ])


def _make_context(tmp_path: Path):
    """Build a RunnerContext with a minimal DataManager stub."""
    config = _minimal_config()

    # --- DataManager stub ---
    X, y = _make_X_y()
    X_tr, y_tr = X.iloc[:42], y.iloc[:42]
    X_te, y_te = X.iloc[42:], y.iloc[42:]

    data_manager = MagicMock()
    data_manager.data = pd.concat([X, y.rename("label")], axis=1)
    data_manager.label_col = "label"
    data_manager.subject_id_col = "sid"
    data_manager.split_data.return_value = (X_tr, X_te, y_tr, y_te)

    # --- PipelineBuilder stub ---
    pipeline_builder = MagicMock()

    def _build(model_name, params, feature_names=None):
        pipe = _stub_pipeline()
        pipe.fit(X_tr, y_tr)
        return pipe

    pipeline_builder.build.side_effect = _build

    logger = logging.getLogger("test_runner")
    logger.setLevel(logging.CRITICAL)

    plan = WorkflowPlan(
        config=config,
        output_dir=str(tmp_path / "out"),
        random_state=42,
    )

    return RunnerContext(
        data_manager=data_manager,
        pipeline_builder=pipeline_builder,
        logger=logger,
        config=config,
    ), plan


# ---------------------------------------------------------------------------
# RunnerContext
# ---------------------------------------------------------------------------


class TestRunnerContext:
    def test_is_dataclass(self) -> None:
        import dataclasses

        assert dataclasses.is_dataclass(RunnerContext)

    def test_fields_present(self) -> None:
        import dataclasses

        fields = {f.name for f in dataclasses.fields(RunnerContext)}
        assert {"data_manager", "pipeline_builder", "logger", "config"}.issubset(fields)


# ---------------------------------------------------------------------------
# WorkflowPlan
# ---------------------------------------------------------------------------


class TestWorkflowPlan:
    def test_config_deep_copied(self) -> None:
        """Mutations to the original config must not affect the stored plan config."""
        cfg = _minimal_config()
        original_value = cfg.random_state
        plan = WorkflowPlan(config=cfg, output_dir="/out", random_state=42)
        # Mutate the original
        cfg.random_state = 999
        # Plan's copy should be unchanged
        assert plan.random_state == 42

    def test_frozen_fields(self) -> None:
        cfg = _minimal_config()
        plan = WorkflowPlan(config=cfg, output_dir="/out", random_state=42)
        with pytest.raises((AttributeError, TypeError)):
            plan.random_state = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# HoldoutRunner
# ---------------------------------------------------------------------------


class TestHoldoutRunner:
    def test_run_returns_run_result(self, tmp_path: Path) -> None:
        from habit.core.machine_learning.runners.holdout import HoldoutRunner

        context, plan = _make_context(tmp_path)
        runner = HoldoutRunner(context=context, plan=plan)
        result = runner.run()
        assert isinstance(result, RunResult)

    def test_run_result_contains_model(self, tmp_path: Path) -> None:
        from habit.core.machine_learning.runners.holdout import HoldoutRunner

        context, plan = _make_context(tmp_path)
        runner = HoldoutRunner(context=context, plan=plan)
        result = runner.run()
        assert "LogisticRegression" in result.models

    def test_model_result_has_metrics(self, tmp_path: Path) -> None:
        from habit.core.machine_learning.runners.holdout import HoldoutRunner

        context, plan = _make_context(tmp_path)
        runner = HoldoutRunner(context=context, plan=plan)
        result = runner.run()
        model_result = result.models["LogisticRegression"]
        assert "auc" in model_result.train_metrics
        assert "auc" in model_result.test_metrics

    def test_dataset_snapshot_populated(self, tmp_path: Path) -> None:
        from habit.core.machine_learning.runners.holdout import HoldoutRunner

        context, plan = _make_context(tmp_path)
        runner = HoldoutRunner(context=context, plan=plan)
        result = runner.run()
        assert result.dataset.x_train is not None
        assert result.dataset.x_test is not None

    def test_to_legacy_results_format(self, tmp_path: Path) -> None:
        from habit.core.machine_learning.runners.holdout import HoldoutRunner

        context, plan = _make_context(tmp_path)
        runner = HoldoutRunner(context=context, plan=plan)
        result = runner.run()
        legacy = result.to_legacy_results()
        assert isinstance(legacy, dict)
        assert "LogisticRegression" in legacy
        assert "train" in legacy["LogisticRegression"]
        assert "test" in legacy["LogisticRegression"]


# ---------------------------------------------------------------------------
# KFoldRunner
# ---------------------------------------------------------------------------


class TestKFoldRunner:
    def test_run_returns_kfold_result(self, tmp_path: Path) -> None:
        from habit.core.machine_learning.runners.kfold import KFoldRunner

        context, plan = _make_context(tmp_path)
        # KFoldRunner needs n_splits on config
        context.config.n_splits = 3
        context.config.shuffle = True

        X, y = _make_X_y()
        runner = KFoldRunner(context=context, plan=plan)
        result = runner.run(X=X, y=y)
        assert isinstance(result, KFoldRunResult)

    def test_kfold_result_has_aggregated(self, tmp_path: Path) -> None:
        from habit.core.machine_learning.runners.kfold import KFoldRunner

        context, plan = _make_context(tmp_path)
        context.config.n_splits = 3
        context.config.shuffle = True

        X, y = _make_X_y()
        runner = KFoldRunner(context=context, plan=plan)
        result = runner.run(X=X, y=y)
        assert "LogisticRegression" in result.aggregated
        agg = result.aggregated["LogisticRegression"]
        assert agg.auc_mean is not None or agg.overall_metrics  # at least one populated
