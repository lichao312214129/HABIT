"""
Structured result objects for machine-learning workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd

from .plan import WorkflowPlan


@dataclass(frozen=True)
class ModelResult:
    """
    Per-model holdout result payload.

    Attributes
    ----------
    model_name:
        Name defined in ML config.
    train:
        Serialized prediction payload on training split.
    test:
        Serialized prediction payload on test split.
    train_metrics:
        Metric dictionary computed on train split.
    test_metrics:
        Metric dictionary computed on test split.
    fitted_estimator:
        Trained sklearn-compatible estimator/pipeline.
    feature_names:
        Ordered input feature names used for training.
    """

    model_name: str
    train: Dict[str, Any]
    test: Dict[str, Any]
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    fitted_estimator: Any
    feature_names: Tuple[str, ...]

    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Convert to legacy workflow result format.

        Returns
        -------
        Dict[str, Any]
            Result structure compatible with existing plot/report components.
        """
        train_payload: Dict[str, Any] = dict(self.train)
        test_payload: Dict[str, Any] = dict(self.test)
        train_payload["metrics"] = self.train_metrics
        test_payload["metrics"] = self.test_metrics
        return {
            "train": train_payload,
            "test": test_payload,
            "pipeline": self.fitted_estimator,
            "features": list(self.feature_names),
        }


@dataclass(frozen=True)
class RunResult:
    """
    Full holdout workflow output package.

    Attributes
    ----------
    plan:
        Workflow plan snapshot.
    models:
        Mapping from model name to model result object.
    summary_rows:
        Summary rows used for summary CSV generation.
    x_train:
        Training feature matrix.
    x_test:
        Test feature matrix.
    y_train:
        Training labels.
    y_test:
        Test labels.
    label_col:
        Label column name from DataManager.
    created_at:
        UTC timestamp in ISO8601 format.
    """

    plan: WorkflowPlan
    models: Dict[str, ModelResult]
    summary_rows: List[Dict[str, Any]]
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    label_col: str
    created_at: str

    @classmethod
    def create(
        cls,
        plan: WorkflowPlan,
        models: Dict[str, ModelResult],
        summary_rows: List[Dict[str, Any]],
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        label_col: str,
    ) -> "RunResult":
        """
        Build a run result with auto timestamp.
        """
        return cls(
            plan=plan,
            models=models,
            summary_rows=summary_rows,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            label_col=label_col,
            created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        )

    def to_legacy_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Convert to the historical ``workflow.results`` format.
        """
        return {name: model.to_legacy_dict() for name, model in self.models.items()}
