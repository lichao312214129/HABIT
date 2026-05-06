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

    The dataclass itself is frozen (re-binding the ``config`` reference is
    forbidden) but :class:`MLConfig` is a Pydantic model whose fields stay
    mutable. To live up to the "immutable snapshot" promise we deep-copy the
    configuration in :meth:`__post_init__` so downstream code that mutates
    ``plan.config.foo`` cannot leak back into the caller's instance.

    Attributes
    ----------
    config:
        Validated ML configuration object used for the run.  A deep copy of
        the caller-provided config is stored.
    output_dir:
        Output directory where artifacts are written.
    random_state:
        Global seed for split and model reproducibility.
    """

    config: MLConfig
    output_dir: str
    random_state: int

    def __post_init__(self) -> None:
        """Defensive copy so the plan is a true snapshot of the config."""
        # ``object.__setattr__`` is required because the dataclass is frozen.
        copied: MLConfig
        if hasattr(self.config, "model_copy"):
            copied = self.config.model_copy(deep=True)
        elif hasattr(self.config, "copy"):
            # Pydantic v1 fallback path.
            copied = self.config.copy(deep=True)
        else:
            copied = self.config
        object.__setattr__(self, "config", copied)
