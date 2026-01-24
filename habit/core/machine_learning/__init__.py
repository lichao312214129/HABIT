"""
Machine Learning module for HABIT package.

This module aggregates key components for ML workflows, including
model training, evaluation, and feature selection.
"""

# Import main workflow entry points
from .workflows.holdout_workflow import MachineLearningWorkflow
from .workflows.kfold_workflow import MachineLearningKFoldWorkflow

# Import the model factory
from .models import ModelFactory

# Import the main feature selection runner
from .feature_selectors import run_selector

# Import the primary visualization tools
from .visualization import Plotter, KMSurvivalPlotter

__all__ = [
    "MachineLearningWorkflow",
    "MachineLearningKFoldWorkflow",
    "ModelFactory",
    "run_selector",
    "Plotter",
    "KMSurvivalPlotter",
]
