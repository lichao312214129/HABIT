# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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

from ..contracts.dataset import DatasetSnapshot
from ..contracts.results import ModelResult, RunResult
from ..evaluation.metrics import (
    bootstrap_metrics,
    calculate_metrics,
    format_metric_ci,
)
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
            self.context.logger.info(
                "[%s] Train AUC=%.4f | Test AUC=%.4f",
                model_name,
                model_result.train_metrics.get("auc", float("nan")),
                model_result.test_metrics.get("auc", float("nan")),
            )

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
        """Build and fit one model pipeline from the configured steps."""
        self.context.logger.info("Training Model: %s", model_name)
        pipeline = self.context.pipeline_builder.build(
            model_name,
            model_params,
            feature_names=list(X_train.columns),
        )
        self.context.logger.info("Fitting pipeline for %s ...", model_name)
        pipeline.fit(X_train, y_train)
        self.context.logger.info("Pipeline fit complete for %s", model_name)
        return pipeline

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

        bootstrap_kwargs = self.bootstrap_options()
        train_metrics_ci: Dict[str, Dict[str, float]] = {}
        test_metrics_ci: Dict[str, Dict[str, float]] = {}
        if bootstrap_kwargs is not None:
            self.context.logger.info(
                "Bootstrapping %s confidence intervals (%s replicates) ...",
                model_name,
                bootstrap_kwargs["n_iterations"],
            )
            train_metrics_ci = bootstrap_metrics(train_container, **bootstrap_kwargs)
            test_metrics_ci = bootstrap_metrics(test_container, **bootstrap_kwargs)

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
            train_metrics_ci=train_metrics_ci,
            test_metrics_ci=test_metrics_ci,
        )

    @staticmethod
    def _build_summary_row(model_result: ModelResult) -> Dict[str, Any]:
        """Build one summary CSV row from a model result."""
        row: Dict[str, Any] = {"Model": model_result.model_name}
        row.update({f"Train_{k}": v for k, v in model_result.train_metrics.items()})
        row.update({f"Test_{k}": v for k, v in model_result.test_metrics.items()})
        # Formatted intervals sit next to the point estimates so the summary CSV
        # is directly quotable; the machine-readable bounds go to the tidy
        # metrics-CI table written by ReportWriter.
        row.update(
            {
                f"Train_{k}_ci": format_metric_ci(v)
                for k, v in model_result.train_metrics_ci.items()
            }
        )
        row.update(
            {
                f"Test_{k}_ci": format_metric_ci(v)
                for k, v in model_result.test_metrics_ci.items()
            }
        )
        return row
