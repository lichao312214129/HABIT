"""
Machine Learning module for HABIT package.

This module aggregates key components for ML workflows, including
model training, evaluation, and feature selection.
"""

# Import main workflow entry points
from .workflows.holdout_workflow import MachineLearningWorkflow
from .workflows.kfold_workflow import MachineLearningKFoldWorkflow
from .runners import HoldoutRunner, KFoldRunner
from .core import WorkflowPlan, RunResult, ModelResult
from .reporting.model_store import ModelStore
from .reporting.report_writer import ReportWriter
from .reporting.plot_composer import PlotComposer

# Import the model factory
from .models import ModelFactory

# Import the main feature selection runner
from .feature_selectors import run_selector

# Import the primary visualization tools
from .visualization import Plotter, KMSurvivalPlotter

__all__ = [
    "MachineLearningWorkflow",
    "MachineLearningKFoldWorkflow",
    "HoldoutRunner",
    "KFoldRunner",
    "WorkflowPlan",
    "RunResult",
    "ModelResult",
    "ModelStore",
    "ReportWriter",
    "PlotComposer",
    "ModelFactory",
    "run_selector",
    "Plotter",
    "KMSurvivalPlotter",
]
