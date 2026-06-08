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
MLConfigurator: factory for machine-learning-domain services.

Owns the assembly of:
    * :class:`HoldoutWorkflow` (holdout train + predict),
    * :class:`KFoldWorkflow` (k-fold CV),
    * :class:`ModelComparison` (multi-file evaluator + plotter + reporter).

Heavy imports (sklearn / matplotlib / shap) are deferred to the factory
methods so importing this module is cheap.
"""

from __future__ import annotations

from typing import Any, Optional

from habit.core.common.configurators.base import BaseConfigurator


class MLConfigurator(BaseConfigurator):
    """Factory for ML training, prediction, k-fold and model comparison."""

    logger_name = 'ml_configurator'

    def create_evaluator(self, output_dir: Optional[str] = None) -> Any:
        """Return a :class:`MultifileEvaluator` rooted at ``output_dir``."""
        from habit.core.machine_learning.evaluation.model_evaluation import (
            MultifileEvaluator,
        )

        return MultifileEvaluator(
            output_dir=output_dir or self._ensure_output_dir()
        )

    def create_reporter(self, output_dir: Optional[str] = None) -> Any:
        """Return a :class:`ReportExporter` rooted at ``output_dir``."""
        from habit.core.machine_learning.reporting.report_exporter import ReportExporter

        return ReportExporter(
            output_dir=output_dir or self._ensure_output_dir(),
            logger=self.logger,
        )

    def create_threshold_manager(self) -> Any:
        """Return a fresh :class:`ThresholdManager`."""
        from habit.core.machine_learning.evaluation.threshold_manager import (
            ThresholdManager,
        )

        return ThresholdManager()

    def create_plot_manager(self, config: Optional[Any] = None) -> Any:
        """Return a :class:`PlotManager` bound to the active output dir."""
        from habit.core.machine_learning.visualization.plot_manager import PlotManager

        return PlotManager(
            config=config if config is not None else self.config,
            output_dir=self._ensure_output_dir(),
        )

    def create_metrics_store(self) -> Any:
        """Return a fresh :class:`MetricsStore`."""
        from habit.core.machine_learning.reporting.report_exporter import MetricsStore

        return MetricsStore()

    def create_model_comparison(
        self,
        config: Optional[Any] = None,
        output_dir: Optional[str] = None,
    ) -> Any:
        """Return a fully wired :class:`ModelComparison` instance."""
        from habit.core.machine_learning.workflows.comparison_workflow import (
            ModelComparison,
        )

        cfg = config if config is not None else self.config
        out_dir = output_dir or self._ensure_output_dir()

        return ModelComparison(
            config=cfg,
            evaluator=self.create_evaluator(out_dir),
            reporter=self.create_reporter(out_dir),
            threshold_manager=self.create_threshold_manager(),
            plot_manager=self.create_plot_manager(cfg),
            metrics_store=self.create_metrics_store(),
            logger=self.logger,
        )

    def _coerce_ml_config(self, config: Optional[Any], purpose: str) -> Any:
        """Coerce a raw config to :class:`MLConfig`, raising on failure."""
        from habit.core.machine_learning.config_schemas import MLConfig

        cfg = config if config is not None else self.config
        if isinstance(cfg, MLConfig):
            return cfg
        if isinstance(cfg, dict):
            return MLConfig.model_validate(cfg)
        if hasattr(cfg, 'to_dict'):
            return MLConfig.model_validate(cfg.to_dict())
        raise ValueError(
            f"Invalid configuration type for {purpose}: {type(cfg)}"
        )

    def create_ml_workflow(self, config: Optional[Any] = None) -> Any:
        """
        Return a :class:`HoldoutWorkflow` instance.

        The workflow's ``run`` method dispatches to ``fit()`` / ``predict()``
        based on ``config.run_mode``.
        """
        from habit.core.machine_learning.workflows.holdout_workflow import (
            HoldoutWorkflow,
        )

        cfg = self._coerce_ml_config(config, 'ML workflow')
        return HoldoutWorkflow(cfg)

    def create_kfold_workflow(self, config: Optional[Any] = None) -> Any:
        """Return a :class:`KFoldWorkflow` instance."""
        from habit.core.machine_learning.workflows.kfold_workflow import (
            KFoldWorkflow,
        )

        cfg = self._coerce_ml_config(config, 'K-Fold workflow')
        return KFoldWorkflow(cfg)
