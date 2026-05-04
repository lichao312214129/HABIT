"""
Holdout runner used by ``MachineLearningWorkflow``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from ..core.plan import WorkflowPlan
from ..core.results import ModelResult, RunResult
from ..evaluation.metrics import calculate_metrics
from ..evaluation.prediction_container import PredictionContainer
from .base import BaseRunner


class HoldoutRunner(BaseRunner):
    """
    Execute holdout train/test logic and return a structured result object.
    """

    def run(self) -> RunResult:
        """
        Train configured models on holdout split.

        Returns
        -------
        RunResult
            Structured run output consumed by writers and plotting components.
        """
        if self.workflow.data_manager.data is None:
            self.load_dataset()
        X_train, X_test, y_train, y_test = self.workflow.data_manager.split_data()
        models: Dict[str, ModelResult] = {}
        summary_rows: List[Dict[str, Any]] = []

        models_config = self.workflow.config_obj.models or {}
        for model_name, model_params in models_config.items():
            model_params_dict = (
                model_params.params if hasattr(model_params, "params") else model_params
            )
            trained_estimator = self._train_one_model(
                model_name=model_name,
                model_params=model_params_dict,
                X_train=X_train,
                y_train=y_train,
            )
            model_result = self._evaluate_one_model(
                model_name=model_name,
                trained_estimator=trained_estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
            models[model_name] = model_result
            summary_rows.append(self._build_summary_row(model_result=model_result))

        if self.workflow.data_manager.label_col is None:
            raise ValueError("DataManager.label_col is required for reporting.")

        return RunResult.create(
            plan=self.plan,
            models=models,
            summary_rows=summary_rows,
            x_train=X_train,
            x_test=X_test,
            y_train=y_train,
            y_test=y_test,
            label_col=self.workflow.data_manager.label_col,
        )

    def _train_one_model(
        self,
        model_name: str,
        model_params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Any:
        """
        Build and fit one model pipeline.
        """
        self.workflow.logger.info("Training Model: %s", model_name)
        pipeline = self.workflow.pipeline_builder.build(
            model_name,
            model_params,
            feature_names=list(X_train.columns),
        )
        return self.workflow._train_with_optional_sampling(pipeline, X_train, y_train)

    def _evaluate_one_model(
        self,
        model_name: str,
        trained_estimator: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> ModelResult:
        """
        Evaluate one trained estimator on train/test splits.
        """
        train_container = PredictionContainer(
            y_true=y_train.values,
            y_prob=trained_estimator.predict_proba(X_train),
            y_pred=trained_estimator.predict(X_train),
        )
        test_container = PredictionContainer(
            y_true=y_test.values,
            y_prob=trained_estimator.predict_proba(X_test),
            y_pred=trained_estimator.predict(X_test),
        )

        train_metrics = calculate_metrics(train_container)
        test_metrics = calculate_metrics(test_container)

        return ModelResult(
            model_name=model_name,
            train=train_container.to_dict(),
            test=test_container.to_dict(),
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            fitted_estimator=trained_estimator,
            feature_names=tuple(X_train.columns.tolist()),
        )

    @staticmethod
    def _build_summary_row(model_result: ModelResult) -> Dict[str, Any]:
        """
        Build one summary CSV row from a model result.
        """
        row: Dict[str, Any] = {"Model": model_result.model_name}
        row.update({f"Train_{k}": v for k, v in model_result.train_metrics.items()})
        row.update({f"Test_{k}": v for k, v in model_result.test_metrics.items()})
        return row
