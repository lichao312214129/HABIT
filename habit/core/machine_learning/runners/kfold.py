"""
K-Fold runner used by :class:`KFoldWorkflow`.

The runner owns the fold/model loops and produces a structured
:class:`KFoldRunResult`.  Reporting (CSV/JSON/plots/ensembles) is delegated
to the ``reporting`` layer which consumes the same result object.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from ..core.results import (
    AggregatedModelResult,
    KFoldModelResult,
    KFoldRunResult,
)
from ..evaluation.metrics import calculate_metrics
from ..evaluation.prediction_container import PredictionContainer
from .base import BaseRunner


class KFoldRunner(BaseRunner):
    """
    Execute K-Fold training/evaluation and return a structured result.
    """

    def run(self, X: pd.DataFrame, y: pd.Series) -> KFoldRunResult:
        """
        Run the full K-Fold loop and collect artefacts.

        Parameters
        ----------
        X:
            Feature matrix.
        y:
            Target vector aligned with ``X``.

        Returns
        -------
        KFoldRunResult
            Structured K-Fold output (per-fold + aggregated payloads).
        """
        splitter = self._build_splitter()

        fold_payloads_per_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        fold_estimators_per_model: Dict[str, List[Any]] = defaultdict(list)

        for fold_idx, (train_index, val_index) in enumerate(splitter.split(X, y)):
            fold_id = fold_idx + 1
            self.context.logger.info("--- Processing Fold %s ---", fold_id)

            X_train: pd.DataFrame = X.iloc[train_index]
            X_val: pd.DataFrame = X.iloc[val_index]
            y_train: pd.Series = y.iloc[train_index]
            y_val: pd.Series = y.iloc[val_index]

            self._train_and_evaluate_fold(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                fold_payloads_per_model=fold_payloads_per_model,
                fold_estimators_per_model=fold_estimators_per_model,
            )

        # Promote the per-model collections to immutable result objects.
        models: Dict[str, KFoldModelResult] = {}
        for model_name, fold_payloads in fold_payloads_per_model.items():
            models[model_name] = KFoldModelResult(
                model_name=model_name,
                folds=tuple(fold_payloads),
                fold_estimators=tuple(fold_estimators_per_model[model_name]),
            )

        aggregated, summary_rows = self._aggregate_models(models=models)

        return KFoldRunResult.create(
            plan=self.plan,
            models=models,
            aggregated=aggregated,
            summary_rows=summary_rows,
        )

    # ------------------------------------------------------------------
    # Splitter
    # ------------------------------------------------------------------

    def _build_splitter(self) -> Union[KFold, StratifiedKFold]:
        """Build the splitter from the run plan's config."""
        cfg = self.context.config
        stratified: bool = bool(cfg.stratified)
        n_splits: int = int(cfg.n_splits)
        splitter_cls = StratifiedKFold if stratified else KFold
        return splitter_cls(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.plan.random_state,
        )

    # ------------------------------------------------------------------
    # Per-fold training/evaluation
    # ------------------------------------------------------------------

    def _train_and_evaluate_fold(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        fold_payloads_per_model: Dict[str, List[Dict[str, Any]]],
        fold_estimators_per_model: Dict[str, List[Any]],
    ) -> None:
        """Train every configured model on one fold and capture predictions."""
        models_config = self.context.config.models

        for model_name, model_params in models_config.items():
            model_params_dict = (
                model_params.params if hasattr(model_params, "params") else model_params
            )

            pipeline = self.context.pipeline_builder.build(
                model_name,
                model_params_dict,
                feature_names=list(X_train.columns),
            )
            trained_estimator = self.context.resampler.fit_with_resampling(
                pipeline, X_train, y_train
            )
            fold_estimators_per_model[model_name].append(trained_estimator)

            container = PredictionContainer(
                y_true=y_val.values,
                y_prob=trained_estimator.predict_proba(X_val),
                y_pred=trained_estimator.predict(X_val),
            )
            metrics_dict = calculate_metrics(container)
            payload = container.to_dict()
            payload["metrics"] = metrics_dict
            fold_payloads_per_model[model_name].append(payload)

    # ------------------------------------------------------------------
    # Aggregation (pure functions)
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_models(
        models: Dict[str, KFoldModelResult],
    ) -> Tuple[Dict[str, AggregatedModelResult], List[Dict[str, Any]]]:
        """
        Aggregate fold-level outputs into model-level results and summary rows.

        Returns
        -------
        Tuple[Dict[str, AggregatedModelResult], List[Dict[str, Any]]]
            The aggregated result map and the summary rows used by the writer.
        """
        aggregated: Dict[str, AggregatedModelResult] = {}
        summary_rows: List[Dict[str, Any]] = []

        for model_name, model in models.items():
            if not model.folds:
                continue

            fold_aucs: List[Optional[float]] = []
            y_true: List[Any] = []
            y_prob: List[Any] = []
            y_pred: List[Any] = []

            for fold_payload in model.folds:
                y_true.extend(fold_payload["y_true"])
                y_prob.extend(fold_payload["y_prob"])
                y_pred.extend(fold_payload["y_pred"])
                fold_aucs.append(fold_payload.get("metrics", {}).get("auc"))

            aggregated_container = PredictionContainer(
                np.array(y_true), np.array(y_prob), np.array(y_pred)
            )
            overall_metrics = calculate_metrics(aggregated_container)

            valid_aucs = [value for value in fold_aucs if value is not None]
            auc_mean: Optional[float] = (
                float(np.mean(valid_aucs)) if valid_aucs else None
            )
            auc_std: Optional[float] = (
                float(np.std(valid_aucs)) if valid_aucs else None
            )

            aggregated[model_name] = AggregatedModelResult(
                model_name=model_name,
                raw=aggregated_container.to_dict(),
                overall_metrics=overall_metrics,
                auc_mean=auc_mean,
                auc_std=auc_std,
            )

            row: Dict[str, Any] = {"Model": model_name}
            if auc_mean is not None:
                row["AUC_Mean"] = auc_mean
            if auc_std is not None:
                row["AUC_Std"] = auc_std
            row.update({f"Overall_{k}": v for k, v in overall_metrics.items()})
            summary_rows.append(row)

        return aggregated, summary_rows
