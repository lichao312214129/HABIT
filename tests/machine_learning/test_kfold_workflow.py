"""
Integration tests for KFoldWorkflow.

Uses a synthetic CSV to keep the test suite self-contained (no demo data
files required).  KFoldWorkflow is the thin orchestration shell; the heavy
logic lives in KFoldRunner which is tested separately in test_runners.py.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_synthetic_csv(path: Path, n: int = 80) -> None:
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "subject_id": [f"P{i:03d}" for i in range(n)],
            **{f"feat_{j}": rng.randn(n) for j in range(8)},
            "label": rng.randint(0, 2, n),
        }
    )
    df.to_csv(path, index=False)


def _make_config_dict(csv_path: str, output_dir: str, n_splits: int = 3) -> dict:
    return {
        "input": [{"path": csv_path, "subject_id_col": "subject_id", "label_col": "label"}],
        "output": output_dir,
        "feature_selection_methods": [],
        "models": {"LogisticRegression": {"params": {"max_iter": 200}}},
        "n_splits": n_splits,
        "is_save_model": False,
        "is_visualize": False,
        "random_state": 42,
    }


# ---------------------------------------------------------------------------
# KFoldWorkflow
# ---------------------------------------------------------------------------


class TestKFoldWorkflow:
    def _workflow(self, tmp_path: Path, n_splits: int = 3):
        from habit.core.machine_learning.config_schemas import MLConfig
        from habit.core.machine_learning.workflows.kfold_workflow import KFoldWorkflow

        csv = tmp_path / "data.csv"
        _write_synthetic_csv(csv)
        cfg = MLConfig.model_validate(
            _make_config_dict(str(csv), str(tmp_path / "out"), n_splits=n_splits)
        )
        return KFoldWorkflow(config=cfg)

    def test_run_completes_without_error(self, tmp_path: Path) -> None:
        wf = self._workflow(tmp_path)
        wf.run()

    def test_run_result_populated(self, tmp_path: Path) -> None:
        wf = self._workflow(tmp_path)
        wf.run()
        assert wf._run_result is not None

    def test_results_contains_aggregated(self, tmp_path: Path) -> None:
        wf = self._workflow(tmp_path)
        wf.run()
        assert "aggregated" in wf.results
        assert "LogisticRegression" in wf.results["aggregated"]

    def test_results_contains_folds(self, tmp_path: Path) -> None:
        wf = self._workflow(tmp_path, n_splits=3)
        wf.run()
        assert "folds" in wf.results
        assert len(wf.results["folds"]) == 3

    def test_fold_pipelines_populated(self, tmp_path: Path) -> None:
        wf = self._workflow(tmp_path, n_splits=3)
        wf.run()
        assert "LogisticRegression" in wf.fold_pipelines
        assert len(wf.fold_pipelines["LogisticRegression"]) == 3

    def test_auc_in_aggregated_metrics(self, tmp_path: Path) -> None:
        wf = self._workflow(tmp_path)
        wf.run()
        agg = wf.results["aggregated"]["LogisticRegression"]
        metrics = agg.get("metrics") or agg.get("overall_metrics") or {}
        assert "auc" in metrics or wf._run_result.aggregated["LogisticRegression"].auc_mean is not None

    def test_summary_csv_written(self, tmp_path: Path) -> None:
        wf = self._workflow(tmp_path)
        wf.run()
        out = tmp_path / "out"
        csv_files = list(out.rglob("*summary*.csv"))
        assert len(csv_files) >= 1

    def test_different_n_splits(self, tmp_path: Path) -> None:
        for n_splits in [3, 5]:
            sub = tmp_path / f"splits_{n_splits}"
            sub.mkdir()
            wf = self._workflow(sub, n_splits=n_splits)
            wf.run()
            assert len(wf.results["folds"]) == n_splits


# ---------------------------------------------------------------------------
# MachineLearningKFoldWorkflow deprecation shim
# ---------------------------------------------------------------------------


class TestKFoldDeprecationShim:
    def test_deprecated_class_emits_deprecation_warning(self, tmp_path: Path) -> None:
        from habit.core.machine_learning.config_schemas import MLConfig
        from habit.core.machine_learning.workflows.kfold_workflow import (
            MachineLearningKFoldWorkflow,
        )

        csv = tmp_path / "data.csv"
        _write_synthetic_csv(csv)
        cfg = MLConfig.model_validate(
            _make_config_dict(str(csv), str(tmp_path / "out"))
        )
        wf = MachineLearningKFoldWorkflow(config=cfg)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wf.run()
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
