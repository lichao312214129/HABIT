"""
Visualization trigger for machine-learning run results.
"""

from __future__ import annotations

from ..core.results import RunResult
from ..visualization.plot_manager import PlotManager


class PlotComposer:
    """
    Render holdout plots from a structured run result.
    """

    def __init__(self, plot_manager: PlotManager, is_visualize: bool = True) -> None:
        self.plot_manager = plot_manager
        self.is_visualize = is_visualize

    def render(self, run_result: RunResult) -> None:
        """
        Generate train/test workflow plots.
        """
        if not self.is_visualize:
            return

        legacy_results = run_result.to_legacy_results()
        self.plot_manager.run_workflow_plots(
            legacy_results,
            prefix="standard_train_",
            X_test=run_result.x_train,
            dataset_type="train",
        )
        self.plot_manager.run_workflow_plots(
            legacy_results,
            prefix="standard_test_",
            X_test=run_result.x_test,
            dataset_type="test",
        )
