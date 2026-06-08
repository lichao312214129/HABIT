# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
"""
K-Fold cross-validation workflow.

The class is a thin orchestration shell on top of
:class:`KFoldRunner`.  Persistence (model ensembles, summary CSV, results
JSON, plots) is delegated to the same :mod:`reporting` components that the
holdout workflow uses, so the two workflows share one reporting seam.

The class was previously named ``MachineLearningKFoldWorkflow`` - the new
preferred name is :class:`KFoldWorkflow`.  The old name is kept as a
deprecation subclass for backward compatibility with external scripts and
the static API tests.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ..config_schemas import MLConfig
from ..contracts.plan import WorkflowPlan
from ..contracts.results import KFoldRunResult
from ..reporting.model_store import ModelStore
from ..reporting.plot_composer import PlotComposer
from ..reporting.report_writer import ReportWriter
from ..runners.kfold import KFoldRunner
from .base import BaseWorkflow


class KFoldWorkflow(BaseWorkflow):
    """
    K-Fold cross-validation workflow.

    The public API matches the historical class while the internals delegate
    to :class:`KFoldRunner` and the reporting components.
    """

    def __init__(self, config: MLConfig) -> None:
        super().__init__(config, module_name="ml_kfold")
        self.results: Dict[str, Any] = {"folds": [], "aggregated": {}}
        self.fold_pipelines: Dict[str, List[Any]] = {}
        self._plan = WorkflowPlan(
            config=self.config_obj,
            output_dir=self.output_dir,
            random_state=self.random_state,
        )
        self._run_result: Optional[KFoldRunResult] = None
        self.runner = KFoldRunner(context=self.runner_context, plan=self._plan)

    def run(self) -> None:
        """Execute the full K-Fold pipeline."""
        self.logger.info("Starting K-Fold pipeline...")

        # 1. Load data.
        X, y = self._load_and_prepare_data()

        # 2. Delegate fold execution to the runner.
        self._run_result = self.runner.run(X=X, y=y)
        self.results = self._run_result.to_legacy_results()
        self.fold_pipelines = self._run_result.fold_pipelines

        # 3. Persist artefacts via the reporting layer.
        model_store = ModelStore(
            output_dir=self.output_dir,
            is_save_model=bool(getattr(self.config_obj, "is_save_model", True)),
        )
        report_writer = ReportWriter(
            output_dir=self.output_dir,
            module_name=self.module_name,
        )
        plot_composer = PlotComposer(
            plot_manager=self.plot_manager,
            is_visualize=bool(getattr(self.config_obj, "is_visualize", True)),
        )

        model_store.save_kfold_ensembles(self._run_result, voting="soft")
        report_writer.write(self._run_result)
        plot_composer.render(self._run_result)

        self.logger.info(
            "K-Fold workflow completed. Results saved to %s", self.output_dir
        )


# ---------------------------------------------------------------------------
# Backward-compatible deprecation shim
# ---------------------------------------------------------------------------


class MachineLearningKFoldWorkflow(KFoldWorkflow):
    """
    Deprecated alias for :class:`KFoldWorkflow`.

    Kept as a thin subclass so external scripts and the configurator that
    import ``MachineLearningKFoldWorkflow`` keep working.  New code should
    use :class:`KFoldWorkflow` directly.
    """

    def run(self) -> None:
        """Run the workflow.  Identical to :meth:`KFoldWorkflow.run`."""
        import warnings

        warnings.warn(
            "MachineLearningKFoldWorkflow is deprecated; use KFoldWorkflow instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().run()
