"""
Report file writer for machine-learning run results.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd

from habit.utils.io_utils import save_csv, save_json

from ..core.results import RunResult


class ReportWriter:
    """
    Generate CSV/JSON reports from a run result.
    """

    def __init__(self, output_dir: str, module_name: str) -> None:
        self.output_dir = output_dir
        self.module_name = module_name

    def write(self, run_result: RunResult) -> None:
        """
        Write summary, result JSON, and merged prediction CSV.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        legacy_results: Dict[str, Dict[str, Any]] = run_result.to_legacy_results()

        # Save JSON without estimator objects.
        results_for_json: Dict[str, Dict[str, Any]] = {}
        for model_name, result_payload in legacy_results.items():
            payload_copy = dict(result_payload)
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

        self._write_holdout_prediction_table(run_result=run_result, legacy_results=legacy_results)

    def _write_holdout_prediction_table(
        self,
        run_result: RunResult,
        legacy_results: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        Write ``all_prediction_results.csv`` using holdout train/test splits.
        """
        train_df = pd.DataFrame(
            {
                "subject_id": run_result.x_train.index,
                "label": run_result.y_train.values,
                "dataset": "train",
            }
        )
        test_df = pd.DataFrame(
            {
                "subject_id": run_result.x_test.index,
                "label": run_result.y_test.values,
                "dataset": "test",
            }
        )

        for model_name, payload in legacy_results.items():
            train_df[f"{model_name}_prob"] = payload["train"]["y_prob"]
            train_df[f"{model_name}_pred"] = payload["train"]["y_pred"]
            test_df[f"{model_name}_prob"] = payload["test"]["y_prob"]
            test_df[f"{model_name}_pred"] = payload["test"]["y_pred"]

        all_df = pd.concat([train_df, test_df], ignore_index=True)
        all_df.to_csv(os.path.join(self.output_dir, "all_prediction_results.csv"), index=False)
