"""
Core machine-learning data contracts.
"""

from .plan import WorkflowPlan
from .results import ModelResult, RunResult

__all__ = ["WorkflowPlan", "ModelResult", "RunResult"]
