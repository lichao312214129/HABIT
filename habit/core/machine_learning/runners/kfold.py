"""
K-Fold runner used by ``MachineLearningKFoldWorkflow``.

The runner extracts fold/model loops from the workflow class while keeping
the existing runtime behavior unchanged. In particular, callback hooks are
still triggered in the same places so downstream integrations continue to work.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from ..evaluation.metrics import calculate_metrics
from ..evaluation.prediction_container import PredictionContainer


class KFoldRunner:
    """
    Execute K-Fold training/evaluation for an existing workflow instance.

    Parameters
    ----------
    workflow:
        The owner workflow that provides configuration, builders, logger, and
        callback hooks.
    """

    def __init__(self, workflow: Any) -> None:
        self.workflow = workflow

    def run(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[Dict[str, Any], Dict[str, List[Any]], List[Dict[str, Any]]]:
        """
        Run the full K-Fold loop and return collected artifacts.

        Parameters
        ----------
        X:
            Feature matrix.
        y:
            Target vector.

        Returns
        -------
        Tuple[Dict[str, Any], Dict[str, List[Any]], List[Dict[str, Any]]]
            - results: fold-level and aggregated prediction payloads.
            - fold_pipelines: trained estimators grouped by model name.
            - summary_results: tabular summary rows for report writers.
        """
        results: Dict[str, Any] = {"folds": [], "aggregated": {}}
        fold_pipelines: Dict[str, List[Any]] = defaultdict(list)

        splitter = self._build_splitter()

        for fold_idx, (train_index, val_index) in enumerate(splitter.split(X, y)):
            fold_id = fold_idx + 1
            self.workflow.logger.info(f"--- Processing Fold {fold_id} ---")

            X_train: pd.DataFrame = X.iloc[train_index]
            X_val: pd.DataFrame = X.iloc[val_index]
            y_train: pd.Series = y.iloc[train_index]
            y_val: pd.Series = y.iloc[val_index]

            fold_result: Dict[str, Any] = self._process_fold(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                fold_id=fold_id,
                fold_pipelines=fold_pipelines,
            )
            results["folds"].append(fold_result)

        summary_results = self._aggregate_results(results=results)
        return results, fold_pipelines, summary_results

    def _build_splitter(self) -> Union[KFold, StratifiedKFold]:
        """
        Build the splitter from workflow config.

        Returns
        -------
        KFold | StratifiedKFold
            A deterministic splitter using workflow random_state.
        """
        stratified: bool = bool(self.workflow.config_obj.stratified)
        n_splits: int = int(self.workflow.config_obj.n_splits)
        splitter_cls = StratifiedKFold if stratified else KFold
        return splitter_cls(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.workflow.random_state,
        )

    def _process_fold(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        fold_id: int,
        fold_pipelines: Dict[str, List[Any]],
    ) -> Dict[str, Any]:
        """
        Train/evaluate all configured models for one fold.

        Parameters
        ----------
        X_train:
            Training features for the current fold.
        y_train:
            Training labels for the current fold.
        X_val:
            Validation features for the current fold.
        y_val:
            Validation labels for the current fold.
        fold_id:
            One-based fold index used in logs and callback payloads.
        fold_pipelines:
            Mutable model->estimators mapping updated in place.

        Returns
        -------
        Dict[str, Any]
            Serialized fold payload compatible with existing reports.
        """
        fold_data: Dict[str, Any] = {"fold": fold_id, "models": {}}
        models_config = self.workflow.config_obj.models

        for model_name, model_params in models_config.items():
            model_params_dict = (
                model_params.params if hasattr(model_params, "params") else model_params
            )

            pipeline = self.workflow.pipeline_builder.build(
                model_name,
                model_params_dict,
                feature_names=list(X_train.columns),
            )
            trained_estimator = self.workflow._train_with_optional_sampling(
                pipeline, X_train, y_train
            )
            fold_pipelines[model_name].append(trained_estimator)

            container = PredictionContainer(
                y_true=y_val.values,
                y_prob=trained_estimator.predict_proba(X_val),
                y_pred=trained_estimator.predict(X_val),
            )
            metrics_dict = calculate_metrics(container)
            fold_data["models"][model_name] = container.to_dict()
            fold_data["models"][model_name]["metrics"] = metrics_dict

        return fold_data

    def _aggregate_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Aggregate fold-level outputs into model-level metrics.

        Parameters
        ----------
        results:
            Mutable results dictionary with ``folds`` and ``aggregated`` keys.

        Returns
        -------
        List[Dict[str, Any]]
            Summary rows, one row per model, used by report writers.
        """
        folds: List[Dict[str, Any]] = results["folds"]
        if not folds:
            return []

        model_names = folds[0]["models"].keys()
        summary_rows: List[Dict[str, Any]] = []

        for model_name in model_names:
            y_true: List[Any] = []
            y_prob: List[Any] = []
            y_pred: List[Any] = []
            fold_aucs: List[Optional[float]] = []

            for fold_data in folds:
                if model_name not in fold_data["models"]:
                    continue
                model_data = fold_data["models"][model_name]
                y_true.extend(model_data["y_true"])
                y_prob.extend(model_data["y_prob"])
                y_pred.extend(model_data["y_pred"])
                fold_aucs.append(model_data["metrics"].get("auc"))

            if not y_true:
                continue

            aggregated_container = PredictionContainer(
                np.array(y_true), np.array(y_prob), np.array(y_pred)
            )
            overall_metrics = calculate_metrics(aggregated_container)

            valid_aucs = [value for value in fold_aucs if value is not None]
            aggregated_payload: Dict[str, Any] = {"metrics": overall_metrics}
            summary_row: Dict[str, Any] = {"Model": model_name}

            if valid_aucs:
                auc_mean = float(np.mean(valid_aucs))
                auc_std = float(np.std(valid_aucs))
                aggregated_payload["auc_mean"] = auc_mean
                aggregated_payload["auc_std"] = auc_std
                summary_row["AUC_Mean"] = auc_mean
                summary_row["AUC_Std"] = auc_std

            results["aggregated"][model_name] = aggregated_container.to_dict()
            results["aggregated"][model_name].update(aggregated_payload)
            summary_row.update({f"Overall_{key}": value for key, value in overall_metrics.items()})
            summary_rows.append(summary_row)

        return summary_rows
