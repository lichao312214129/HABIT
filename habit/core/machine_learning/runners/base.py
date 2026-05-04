"""
Base runner definitions for machine-learning workflows.
"""

from __future__ import annotations

from typing import Any, Tuple

import pandas as pd

from ..core.plan import WorkflowPlan


class BaseRunner:
    """
    Shared runner helper that reads data through workflow-owned DataManager.
    """

    def __init__(self, workflow: Any, plan: WorkflowPlan) -> None:
        self.workflow = workflow
        self.plan = plan

    def load_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load features/labels via workflow data manager.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            Feature matrix and target vector.
        """
        self.workflow.data_manager.load_data()
        if self.workflow.data_manager.data is None or self.workflow.data_manager.label_col is None:
            raise ValueError("DataManager returned empty dataset or missing label column.")
        X: pd.DataFrame = self.workflow.data_manager.data.drop(columns=[self.workflow.data_manager.label_col])
        y: pd.Series = self.workflow.data_manager.data[self.workflow.data_manager.label_col]
        return X, y
