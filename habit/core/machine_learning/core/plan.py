"""
Workflow plan definitions for machine-learning execution.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config_schemas import MLConfig


@dataclass(frozen=True)
class WorkflowPlan:
    """
    Immutable snapshot of workflow execution settings.

    Attributes
    ----------
    config:
        Validated ML configuration object used for the run.
    output_dir:
        Output directory where artifacts are written.
    random_state:
        Global seed for split and model reproducibility.
    """

    config: MLConfig
    output_dir: str
    random_state: int
