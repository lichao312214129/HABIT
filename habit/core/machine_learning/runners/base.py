"""
Base runner definitions for machine-learning workflows.

The base class only owns the ``context`` and ``plan`` references.  Concrete
runners (:class:`HoldoutRunner`, :class:`KFoldRunner`, :class:`InferenceRunner`)
implement their own ``run`` signature because their inputs/outputs differ.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from ..core.plan import WorkflowPlan
from .context import RunnerContext


class BaseRunner:
    """
    Shared runner helper that depends on a :class:`RunnerContext`.

    Concrete runners extend this class and add their own ``run`` method with
    the signature appropriate to their workflow mode (no shared abstract
    ``run`` is enforced because the input shapes differ).
    """

    def __init__(self, context: RunnerContext, plan: WorkflowPlan) -> None:
        """
        Parameters
        ----------
        context:
            Bundle of collaborators (data manager, pipeline builder,
            resampler, logger, config).
        plan:
            Immutable execution-plan snapshot.
        """
        self.context = context
        self.plan = plan

    def load_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load features/labels via the context's data manager.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            Feature matrix (without label column) and target vector.
        """
        data_manager = self.context.data_manager
        if data_manager.data is None:
            data_manager.load_data()
        if data_manager.data is None or data_manager.label_col is None:
            raise ValueError(
                "DataManager returned empty dataset or missing label column."
            )
        X: pd.DataFrame = data_manager.data.drop(columns=[data_manager.label_col])
        y: pd.Series = data_manager.data[data_manager.label_col]
        return X, y
