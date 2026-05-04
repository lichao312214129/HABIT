"""
Machine Learning module for the HABIT package.

This module aggregates the public entry points used by the CLI and external
scripts.  The naming convention follows the runner/workflow split:

* :class:`HoldoutWorkflow`  - holdout train/test workflow.
* :class:`KFoldWorkflow`    - K-Fold cross-validation workflow.
* :class:`HoldoutRunner`,
  :class:`KFoldRunner`,
  :class:`InferenceRunner`  - execution-only components.
* :class:`ModelStore`,
  :class:`ReportWriter`,
  :class:`PlotComposer`     - reporting components.

The legacy class names ``MachineLearningWorkflow`` /
``MachineLearningKFoldWorkflow`` are kept as deprecation subclasses so
existing scripts keep working; new code should prefer the new names.
"""

from .core import (
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
from .feature_selectors import run_selector
from .models import ModelFactory
from .reporting.model_store import ModelStore
from .reporting.plot_composer import PlotComposer
from .reporting.report_writer import ReportWriter
from .runners import HoldoutRunner, InferenceRunner, KFoldRunner, RunnerContext
from .visualization import KMSurvivalPlotter, Plotter
from .workflows.holdout_workflow import HoldoutWorkflow, MachineLearningWorkflow
from .workflows.kfold_workflow import KFoldWorkflow, MachineLearningKFoldWorkflow

__all__ = [
    # Workflows (preferred names).
    "HoldoutWorkflow",
    "KFoldWorkflow",
    # Workflows (deprecated aliases).
    "MachineLearningWorkflow",
    "MachineLearningKFoldWorkflow",
    # Runners.
    "HoldoutRunner",
    "KFoldRunner",
    "InferenceRunner",
    "RunnerContext",
    # Core data contracts.
    "WorkflowPlan",
    "WorkflowResult",
    "DatasetSnapshot",
    "RunResult",
    "ModelResult",
    "KFoldRunResult",
    "KFoldModelResult",
    "AggregatedModelResult",
    "InferenceResult",
    # Reporting.
    "ModelStore",
    "ReportWriter",
    "PlotComposer",
    # Other.
    "ModelFactory",
    "run_selector",
    "Plotter",
    "KMSurvivalPlotter",
]
