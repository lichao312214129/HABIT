"""
Core machine-learning data contracts.
"""

from .dataset import DatasetSnapshot
from .plan import WorkflowPlan
from .protocols import WorkflowResult
from .results import (
    AggregatedModelResult,
    InferenceResult,
    KFoldModelResult,
    KFoldRunResult,
    ModelResult,
    RunResult,
)

__all__ = [
    "DatasetSnapshot",
    "WorkflowPlan",
    "WorkflowResult",
    "ModelResult",
    "KFoldModelResult",
    "AggregatedModelResult",
    "RunResult",
    "KFoldRunResult",
    "InferenceResult",
]
