"""
Holdout runner used by :class:`HoldoutWorkflow`.

The runner owns *only* the training/evaluation loop.  Persistence, plotting
and CSV/JSON writing happen in the reporting layer once a :class:`RunResult`
is returned.  The runner depends on a :class:`RunnerContext` rather than the
workflow itself, so it can be unit-tested with stub collaborators.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from ..core.dataset import DatasetSnapshot
from ..core.results import ModelResult, RunResult
from ..evaluation.metrics import calculate_metrics
from ..evaluation.prediction_container import PredictionContainer
from .base import BaseRunner


class HoldoutRunner(BaseRunner):
    """
    Execute holdout train/test logic and return a structured run result.
    """

    def run(self) -> RunResult:
        """
        Train every configured model on the holdout split and evaluate.

        Returns
        -------
        RunResult
            Structured output consumed by writers and plotting components.
        """
        data_manager = self.context.data_manager
        if data_manager.data is None:
            self.load_dataset()
        X_train, X_test, y_train, y_test = data_manager.split_data()

        models: Dict[str, ModelResult] = {}
        summary_rows: List[Dict[str, Any]] = []

        models_config = self.context.config.models or {}
        for model_name, model_params in models_config.items():
            model_params_dict = self._extract_params(model_params)
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

        if data_manager.label_col is None:
            raise ValueError("DataManager.label_col is required for reporting.")

        dataset = DatasetSnapshot(
            label_col=data_manager.label_col,
            x_train=X_train,
            x_test=X_test,
            y_train=y_train,
            y_test=y_test,
            subject_id_col=getattr(data_manager, "subject_id_col", None),
        )

        return RunResult.create(
            plan=self.plan,
            models=models,
            summary_rows=summary_rows,
            dataset=dataset,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_params(model_params: Any) -> Dict[str, Any]:
        """Return a plain ``dict`` of model params from a Pydantic field."""
        if hasattr(model_params, "params"):
            return dict(model_params.params)
        if isinstance(model_params, dict):
            return dict(model_params)
        return {}

    def _train_one_model(
        self,
        model_name: str,
        model_params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Any:
        """Build and fit one model pipeline using the context resampler."""
        self.context.logger.info("Training Model: %s", model_name)
        pipeline = self.context.pipeline_builder.build(
            model_name,
            model_params,
            feature_names=list(X_train.columns),
        )
        return self.context.resampler.fit_with_resampling(pipeline, X_train, y_train)

    def _evaluate_one_model(
        self,
        model_name: str,
        trained_estimator: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> ModelResult:
        """Evaluate one fitted estimator on both train and test splits."""
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
            train_subject_ids=tuple(X_train.index.tolist()),
            test_subject_ids=tuple(X_test.index.tolist()),
        )

    @staticmethod
    def _build_summary_row(model_result: ModelResult) -> Dict[str, Any]:
        """Build one summary CSV row from a model result."""
        row: Dict[str, Any] = {"Model": model_result.model_name}
        row.update({f"Train_{k}": v for k, v in model_result.train_metrics.items()})
        row.update({f"Test_{k}": v for k, v in model_result.test_metrics.items()})
        return row
