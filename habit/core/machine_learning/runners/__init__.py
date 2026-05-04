"""
Runner layer for machine-learning workflows.

This package hosts execution-focused classes that encapsulate
training/evaluation loops. Workflows can delegate heavy logic here to keep
public workflow classes thin and easier to maintain.
"""

from .kfold import KFoldRunner
from .holdout import HoldoutRunner

__all__ = ["KFoldRunner", "HoldoutRunner"]
