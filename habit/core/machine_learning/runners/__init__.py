"""
Runner layer for machine-learning workflows.

Concrete runners encapsulate the training/evaluation/prediction loops.
Workflows delegate heavy logic to runners so the public workflow classes
stay thin orchestration shells.
"""

from .context import RunnerContext
from .holdout import HoldoutRunner
from .inference import InferenceRunner
from .kfold import KFoldRunner

__all__ = [
    "RunnerContext",
    "HoldoutRunner",
    "InferenceRunner",
    "KFoldRunner",
]
