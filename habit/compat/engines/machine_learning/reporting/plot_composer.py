# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Visualization trigger for machine-learning run results.

Like :class:`ReportWriter`, the composer routes by runtime type so the same
component can render figures for both holdout and K-Fold runs.  Inference
runs produce evaluation plots when ground-truth labels were available during
prediction (``evaluate=true``); otherwise they produce predictions only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from habit.utils.log_utils import get_module_logger

from ..contracts.results import InferenceResult, KFoldRunResult, RunResult
from ..visualization.plot_manager import PlotManager


class PlotComposer:
    """Render figures from a structured workflow result."""

    def __init__(self, plot_manager: PlotManager, is_visualize: bool = True) -> None:
        self.plot_manager = plot_manager
        self.is_visualize = is_visualize
        self.logger = get_module_logger("ml.plot_composer")

    def render(self, run_result: Any) -> None:
        """
        Render figures based on the result variant.

        Args:
        run_result:
            One of :class:`RunResult`, :class:`KFoldRunResult`,
            :class:`InferenceResult`.  Inference plots require evaluation
            arrays (``y_true`` / ``y_prob``) collected when labels exist.
        """
        if not self.is_visualize:
            return

        if isinstance(run_result, RunResult):
            self._render_holdout(run_result)
        elif isinstance(run_result, KFoldRunResult):
            self._render_kfold(run_result)
        elif isinstance(run_result, InferenceResult):
            self._render_inference(run_result)
        else:  # pragma: no cover - defensive
            raise TypeError(
                f"PlotComposer cannot handle result type: {type(run_result).__name__}"
            )

    # ------------------------------------------------------------------
    # Holdout
    # ------------------------------------------------------------------

    def _render_holdout(self, run_result: RunResult) -> None:
        """Render train and test plots from a holdout run."""
        legacy_results = run_result.to_legacy_results()
        self.plot_manager.run_workflow_plots(
            legacy_results,
            prefix="standard_train_",
            X_test=run_result.dataset.x_train,
            dataset_type="train",
        )
        self.plot_manager.run_workflow_plots(
            legacy_results,
            prefix="standard_test_",
            X_test=run_result.dataset.x_test,
            dataset_type="test",
        )

    # ------------------------------------------------------------------
    # K-Fold
    # ------------------------------------------------------------------

    def _render_kfold(self, run_result: KFoldRunResult) -> None:
        """Render aggregated K-Fold plots."""
        legacy_results = run_result.to_legacy_results()
        aggregated_payload = legacy_results.get("aggregated", {})
        self.plot_manager.run_workflow_plots(
            aggregated_payload,
            prefix="kfold_",
        )

    # ------------------------------------------------------------------
    # Inference / predict
    # ------------------------------------------------------------------

    def _render_inference(self, run_result: InferenceResult) -> None:
        """
        Render evaluation plots for a prediction run.

        Requires ``evaluate=true`` with resolved labels so ``y_true`` and
        ``y_prob`` are present.  Without labels, prediction remains CSV-only.
        """
        if run_result.y_true is None or run_result.y_prob is None:
            self.logger.info(
                "Skipping prediction plots: no ground-truth labels or "
                "probabilities available (set evaluate=true and provide "
                "label_col when evaluation figures are needed)."
            )
            return

        model_name = self._infer_model_name(run_result.pipeline_path)
        plot_payload: Dict[str, Any] = {
            model_name: {
                "raw": {
                    "y_true": run_result.y_true,
                    "y_prob": run_result.y_prob,
                    "y_pred": run_result.y_pred,
                },
                "pipeline": run_result.fitted_estimator,
            }
        }
        self.plot_manager.run_workflow_plots(
            plot_payload,
            prefix="predict_",
            X_test=run_result.feature_frame,
            dataset_type="raw",
        )

    @staticmethod
    def _infer_model_name(pipeline_path: str) -> str:
        """Derive a readable model name from a saved pipeline filename."""
        stem = Path(pipeline_path).stem
        for suffix in ("_final_pipeline", "_pipeline"):
            if stem.endswith(suffix):
                return stem[: -len(suffix)] or "model"
        return stem or "model"
