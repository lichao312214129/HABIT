"""
Machine Learning module for the HABIT package.

This module aggregates the public entry points used by the CLI and external
scripts.  Heavy imports (workflows, runners, plotting / ``shap``) are lazy;
lightweight contracts and :class:`MLConfigurator` remain eager.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from habit.utils.lazy_exports import lazy_getattr

from .configurator import MLConfigurator
from .contracts import (
    AggregatedModelResult,
    DatasetSnapshot,
    InferenceResult,
    KFoldModelResult,
    KFoldRunResult,
    ModelResult,
    RunResult,
    WorkflowPlan,
    WorkflowResult,
)

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "run_selector": (".feature_selectors", "run_selector"),
    "ModelFactory": (".models.factory", "ModelFactory"),
    "ModelStore": (".reporting.model_store", "ModelStore"),
    "PlotComposer": (".reporting.plot_composer", "PlotComposer"),
    "ReportWriter": (".reporting.report_writer", "ReportWriter"),
    "HoldoutRunner": (".runners.holdout", "HoldoutRunner"),
    "InferenceRunner": (".runners.inference", "InferenceRunner"),
    "KFoldRunner": (".runners.kfold", "KFoldRunner"),
    "RunnerContext": (".runners.context", "RunnerContext"),
    "Plotter": (".visualization.plotting", "Plotter"),
    "KMSurvivalPlotter": (".visualization.km_survival", "KMSurvivalPlotter"),
    "HoldoutWorkflow": (".workflows.holdout_workflow", "HoldoutWorkflow"),
    "MachineLearningWorkflow": (".workflows.holdout_workflow", "HoldoutWorkflow"),
    "KFoldWorkflow": (".workflows.kfold_workflow", "KFoldWorkflow"),
    "MachineLearningKFoldWorkflow": (".workflows.kfold_workflow", "KFoldWorkflow"),
}

__all__ = [
    "HoldoutWorkflow",
    "KFoldWorkflow",
    "MLConfigurator",
    "MachineLearningWorkflow",
    "MachineLearningKFoldWorkflow",
    "HoldoutRunner",
    "KFoldRunner",
    "InferenceRunner",
    "RunnerContext",
    "WorkflowPlan",
    "WorkflowResult",
    "DatasetSnapshot",
    "RunResult",
    "ModelResult",
    "KFoldRunResult",
    "KFoldModelResult",
    "AggregatedModelResult",
    "InferenceResult",
    "ModelStore",
    "ReportWriter",
    "PlotComposer",
    "ModelFactory",
    "run_selector",
    "Plotter",
    "KMSurvivalPlotter",
]


def __getattr__(name: str) -> Any:
    """Resolve heavy ML exports on first access."""
    return lazy_getattr(name, globals(), _LAZY_EXPORTS)
