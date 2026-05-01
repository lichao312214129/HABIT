"""
Core modules for HABIT package.

V1: imports are fail-fast. If a core import is broken, you get a real
``ImportError`` immediately rather than a silent ``None`` attribute.
For genuinely optional third-party dependencies, use the
``habit.is_available`` / ``habit.import_error`` helpers exposed at the
package root.
"""

from .habitat_analysis import HabitatAnalysis, HabitatFeatureExtractor
from .machine_learning.workflows.holdout_workflow import (
    MachineLearningWorkflow as Modeling,
)

__all__ = [
    "HabitatAnalysis",
    "HabitatFeatureExtractor",
    "Modeling",
]
