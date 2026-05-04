"""
Standard Machine Learning Workflow (Train / Test Split).

Inherits from :class:`BaseWorkflow` for consistent infrastructure. V1
unifies training and prediction here: ``run`` dispatches on
``config.run_mode`` so a single workflow class handles both paths and the
old ``PredictionWorkflow`` is gone.
"""

import os
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from .base import BaseWorkflow
from ..config_schemas import MLConfig
from ..evaluation.metrics import calculate_metrics
from ..evaluation.prediction_container import PredictionContainer


class MachineLearningWorkflow(BaseWorkflow):
    """
    Hold-out train/test workflow.

    Public entry points:
        - :meth:`fit`      : train + persist pipelines (one per model).
        - :meth:`predict`  : load a saved ``*_final_pipeline.pkl`` and predict
          on ``config.input[0].path``.
        - :meth:`run` : dispatcher; routes to :meth:`fit` /
          :meth:`predict` based on ``config.run_mode``. This is the V1 entry
          used by the CLI (`habit model`).
    """

    def __init__(self, config: MLConfig):
        super().__init__(config, module_name="ml_standard")
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None

    # ---------------------------------------------------------------------
    # Dispatcher
    # ---------------------------------------------------------------------

    def run(self) -> None:
        """Route to :meth:`fit` or :meth:`predict` according to ``run_mode``."""
        run_mode = getattr(self.config_obj, "run_mode", "train")
        if run_mode == "train":
            self.fit()
        elif run_mode == "predict":
            self.predict()
        else:
            raise ValueError(
                f"MachineLearningWorkflow: unsupported run_mode={run_mode!r}; "
                f"expected 'train' or 'predict'."
            )

    # ---------------------------------------------------------------------
    # Train path
    # ---------------------------------------------------------------------

    def fit(self) -> None:
        """Train every configured model on the train split and persist outputs."""
        self.logger.info("Starting Standard ML Pipeline (mode=train)...")
        self.callbacks.on_pipeline_start()

        # 1. Load Data.
        X, y = self._load_and_prepare_data()

        # 2. Split Data.
        self.X_train, self.X_test, y_train, y_test = self.data_manager.split_data()

        models_config = self.config_obj.models or {}
        summary_results = []

        # 3. Process Models.
        for m_name, m_params in models_config.items():
            model_params_dict = m_params.params if hasattr(m_params, 'params') else m_params

            self.logger.info(f"Training Model: {m_name}")
            self.callbacks.on_model_start(m_name)

            pipeline = self.pipeline_builder.build(
                m_name,
                model_params_dict,
                feature_names=list(self.X_train.columns),
            )
            trained_estimator = self._train_with_optional_sampling(
                pipeline, self.X_train, y_train
            )

            train_container = PredictionContainer(
                y_true=y_train.values,
                y_prob=trained_estimator.predict_proba(self.X_train),
                y_pred=trained_estimator.predict(self.X_train),
            )
            test_container = PredictionContainer(
                y_true=y_test.values,
                y_prob=trained_estimator.predict_proba(self.X_test),
                y_pred=trained_estimator.predict(self.X_test),
            )

            train_metrics = calculate_metrics(train_container)
            test_metrics = calculate_metrics(test_container)

            self.results[m_name] = {
                'train': train_container.to_dict(),
                'test': test_container.to_dict(),
                'pipeline': trained_estimator,
                'features': list(self.X_train.columns),
            }
            self.results[m_name]['train']['metrics'] = train_metrics
            self.results[m_name]['test']['metrics'] = test_metrics

            self.callbacks.on_model_end(m_name, logs={'pipeline': trained_estimator})

            row = {'Model': m_name}
            row.update({f'Train_{k}': v for k, v in train_metrics.items()})
            row.update({f'Test_{k}': v for k, v in test_metrics.items()})
            summary_results.append(row)

        # 4. End Pipeline (callbacks handle final reports / plotting).
        self.callbacks.on_pipeline_end(logs={'summary_results': summary_results})
        self.logger.info(
            f"Standard ML workflow completed (train). Results saved to {self.output_dir}"
        )

    # ---------------------------------------------------------------------
    # Predict path
    # ---------------------------------------------------------------------

    def predict(self) -> None:
        """
        Load a saved pipeline and predict on the input CSV.

        Reads:
            - ``config_obj.pipeline_path``: the ``*_final_pipeline.pkl``.
            - ``config_obj.input[0].path``: input CSV.
            - ``config_obj.input[0].label_col``: ground-truth label column,
              used only when ``evaluate=True``.

        Writes:
            - ``<output>/prediction_results.csv``
            - ``<output>/evaluation_metrics.csv`` (only if ``evaluate=True``).
        """
        self.logger.info("Starting Standard ML Pipeline (mode=predict)...")
        self.callbacks.on_pipeline_start(logs={"mode": "predict"})

        pipeline_path = self.config_obj.pipeline_path
        if not pipeline_path or not os.path.exists(pipeline_path):
            raise FileNotFoundError(
                f"Saved pipeline not found: {pipeline_path}"
            )

        if not self.config_obj.input:
            raise ValueError(
                "MachineLearningWorkflow.predict requires at least one entry "
                "in config.input."
            )
        input_cfg = self.config_obj.input[0]
        data_path = input_cfg.path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Input data file not found: {data_path}")

        self.logger.info(f"Loading data from {data_path}")
        df = self.data_manager.load_inference_data(data_path)
        self.logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        self.logger.info(f"Loading pipeline from {pipeline_path}")
        pipeline = joblib.load(pipeline_path)

        # Find ground-truth column when evaluation is requested.
        label_col = self._find_predict_label_column(df, input_cfg.label_col)

        evaluate = bool(getattr(self.config_obj, "evaluate", False))
        if evaluate and label_col is None:
            self.logger.warning(
                "evaluate=True but no label column resolved; skipping evaluation."
            )
            evaluate = False

        y_true = df[label_col].values if (evaluate and label_col) else None
        X = df  # FeatureSelectTransformer in the saved pipeline does column selection.

        # Predictions.
        preds = pipeline.predict(X)

        probs_raw = None
        if hasattr(pipeline, "predict_proba"):
            try:
                probs_raw = pipeline.predict_proba(X)
            except Exception as exc:
                self.logger.warning(f"Failed to predict probabilities: {exc}")

        # Evaluation + probability extraction via PredictionContainer.
        metrics_results: dict = {}
        probs_for_output = None
        if probs_raw is not None:
            y_for_container = y_true if y_true is not None else preds
            try:
                container = PredictionContainer(
                    y_true=y_for_container,
                    y_prob=probs_raw,
                    y_pred=preds,
                )
                probs_for_output = container.get_eval_probs()

                if evaluate and y_true is not None:
                    metrics_results = calculate_metrics(container)
                    self.logger.info("Evaluation Metrics:")
                    for k, v in metrics_results.items():
                        self.logger.info(f"  {k}: {v}")
            except Exception as exc:
                self.logger.error(f"Failed to use PredictionContainer: {exc}")
                probs_for_output = probs_raw

        # Save results.
        results = df.copy()
        results[self.config_obj.output_label_col] = preds
        if probs_for_output is not None:
            if probs_for_output.ndim > 1:
                # Multiclass: store as a list (CSV-friendly).
                results[self.config_obj.output_prob_col] = list(probs_for_output)
            else:
                results[self.config_obj.output_prob_col] = probs_for_output

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, 'prediction_results.csv')
        results.to_csv(output_path, index=False)
        self.logger.info(f"Saved prediction results to {output_path}")

        if evaluate and metrics_results:
            metrics_path = os.path.join(self.output_dir, 'evaluation_metrics.csv')
            pd.DataFrame([metrics_results]).to_csv(metrics_path, index=False)
            self.logger.info(f"Saved evaluation metrics to {metrics_path}")

        # Keep callback lifecycle aligned with train mode; pass lightweight
        # summary so reporting callbacks can remain mode-agnostic.
        summary_results = []
        if metrics_results:
            row = {"Model": "inference_pipeline"}
            row.update({f"Predict_{key}": value for key, value in metrics_results.items()})
            summary_results.append(row)

        self.callbacks.on_pipeline_end(
            logs={
                "mode": "predict",
                "summary_results": summary_results,
            }
        )

        self.logger.info(
            f"Standard ML workflow completed (predict). Output dir: {self.output_dir}"
        )

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    def _find_predict_label_column(
        self,
        df: pd.DataFrame,
        configured_label_col: Optional[str],
    ) -> Optional[str]:
        """
        Resolve the ground-truth column for the predict path.

        Priority:
            1. The label column declared on ``config.input[0].label_col``
               (if it actually exists in the dataframe).
            2. A small set of conventional fallbacks.

        Args:
            df: input dataframe.
            configured_label_col: the value from ``input[0].label_col``.

        Returns:
            Column name if a match is found, else ``None``.
        """
        if configured_label_col and configured_label_col in df.columns:
            return configured_label_col
        if configured_label_col:
            self.logger.warning(
                f"Configured label column '{configured_label_col}' not found "
                "in data; trying fallback candidates."
            )
        for col in ('label', 'Target', 'class', 'diagnosis', 'outcome', 'y'):
            if col in df.columns:
                return col
        return None
