"""
Visualization trigger for machine-learning run results.

Like :class:`ReportWriter`, the composer routes by runtime type so the same
component can render figures for both holdout and K-Fold runs.  Inference
runs intentionally produce no plots (predictions only).
"""

from __future__ import annotations

from typing import Any

from ..core.results import InferenceResult, KFoldRunResult, RunResult
from ..visualization.plot_manager import PlotManager


class PlotComposer:
    """Render figures from a structured workflow result."""

    def __init__(self, plot_manager: PlotManager, is_visualize: bool = True) -> None:
        self.plot_manager = plot_manager
        self.is_visualize = is_visualize

    def render(self, run_result: Any) -> None:
        """
        Render figures based on the result variant.

        Parameters
        ----------
        run_result:
            One of :class:`RunResult`, :class:`KFoldRunResult`,
            :class:`InferenceResult`.  Inference runs produce no figures.
        """
        if not self.is_visualize:
            return

        if isinstance(run_result, RunResult):
            self._render_holdout(run_result)
        elif isinstance(run_result, KFoldRunResult):
            self._render_kfold(run_result)
        elif isinstance(run_result, InferenceResult):
            return
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
