"""
Report file writer for machine-learning run results.

The writer accepts every concrete result variant produced by the runner
layer (:class:`RunResult`, :class:`KFoldRunResult`,
:class:`InferenceResult`).  Routing by result type keeps each path explicit
and avoids the previous duplication where the K-Fold workflow handled
``save_json``/``save_csv`` on its own.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd

from habit.utils.io_utils import save_csv, save_json

from ..core.results import InferenceResult, KFoldRunResult, RunResult


class ReportWriter:
    """Generate CSV/JSON reports from a structured run result."""

    def __init__(self, output_dir: str, module_name: str) -> None:
        self.output_dir = output_dir
        self.module_name = module_name

    # ------------------------------------------------------------------
    # Public dispatcher
    # ------------------------------------------------------------------

    def write(self, run_result: Any) -> None:
        """
        Persist artefacts based on the runtime type of ``run_result``.

        Parameters
        ----------
        run_result:
            One of :class:`RunResult`, :class:`KFoldRunResult`,
            :class:`InferenceResult`.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        if isinstance(run_result, RunResult):
            self._write_holdout(run_result)
        elif isinstance(run_result, KFoldRunResult):
            self._write_kfold(run_result)
        elif isinstance(run_result, InferenceResult):
            self._write_inference(run_result)
        else:  # pragma: no cover - defensive
            raise TypeError(
                f"ReportWriter cannot handle result type: {type(run_result).__name__}"
            )

    # ------------------------------------------------------------------
    # Holdout
    # ------------------------------------------------------------------

    def _write_holdout(self, run_result: RunResult) -> None:
        """Write summary, results JSON, and merged prediction CSV."""
        legacy_results: Dict[str, Dict[str, Any]] = run_result.to_legacy_results()

        # JSON omits the fitted estimator (not serialisable, not useful in JSON).
        results_for_json: Dict[str, Dict[str, Any]] = {}
        for model_name, payload in legacy_results.items():
            payload_copy = dict(payload)
            payload_copy.pop("pipeline", None)
            results_for_json[model_name] = payload_copy

        save_json(
            results_for_json,
            os.path.join(self.output_dir, f"{self.module_name}_results.json"),
        )

        summary_df = pd.DataFrame(run_result.summary_rows)
        save_csv(
            summary_df,
            os.path.join(self.output_dir, f"{self.module_name}_summary.csv"),
        )

        self._write_holdout_prediction_table(run_result=run_result)

    def _write_holdout_prediction_table(self, run_result: RunResult) -> None:
        """Write ``all_prediction_results.csv`` using train/test splits."""
        dataset = run_result.dataset
        if dataset.x_train is None or dataset.x_test is None:
            return
        if dataset.y_train is None or dataset.y_test is None:
            return

        train_df = pd.DataFrame(
            {
                "subject_id": dataset.x_train.index,
                "label": dataset.y_train.values,
                "dataset": "train",
            }
        )
        test_df = pd.DataFrame(
            {
                "subject_id": dataset.x_test.index,
                "label": dataset.y_test.values,
                "dataset": "test",
            }
        )

        for model_name, model in run_result.models.items():
            train_df[f"{model_name}_prob"] = model.train["y_prob"]
            train_df[f"{model_name}_pred"] = model.train["y_pred"]
            test_df[f"{model_name}_prob"] = model.test["y_prob"]
            test_df[f"{model_name}_pred"] = model.test["y_pred"]

        all_df = pd.concat([train_df, test_df], ignore_index=True)
        all_df.to_csv(
            os.path.join(self.output_dir, "all_prediction_results.csv"), index=False
        )

    # ------------------------------------------------------------------
    # K-Fold
    # ------------------------------------------------------------------

    def _write_kfold(self, run_result: KFoldRunResult) -> None:
        """Write summary CSV and full results JSON for a K-Fold run."""
        save_json(
            run_result.to_legacy_results(),
            os.path.join(self.output_dir, f"{self.module_name}_results.json"),
        )
        summary_df = pd.DataFrame(run_result.summary_rows)
        save_csv(
            summary_df,
            os.path.join(self.output_dir, f"{self.module_name}_summary.csv"),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _write_inference(self, run_result: InferenceResult) -> None:
        """Write the prediction CSV and (optionally) evaluation metrics."""
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, "prediction_results.csv")
        run_result.predictions.to_csv(output_path, index=False)

        if run_result.metrics:
            metrics_path = os.path.join(self.output_dir, "evaluation_metrics.csv")
            pd.DataFrame([run_result.metrics]).to_csv(metrics_path, index=False)
