"""
Standard Machine Learning Workflow (Train / Test Split).

Inherits from :class:`BaseWorkflow` for consistent infrastructure. V1
unifies training and prediction here: ``run`` dispatches on
``config.run_mode`` so a single workflow class handles both paths and the
old ``PredictionWorkflow`` is gone.
"""

import os
from typing import Optional

import joblib
import pandas as pd

from .base import BaseWorkflow
from ..config_schemas import MLConfig
from ..core.plan import WorkflowPlan
from ..core.results import RunResult
from ..evaluation.metrics import calculate_metrics
from ..evaluation.prediction_container import PredictionContainer
from ..reporting.model_store import ModelStore
from ..reporting.plot_composer import PlotComposer
from ..reporting.report_writer import ReportWriter
from ..runners.holdout import HoldoutRunner


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
        self._plan = WorkflowPlan(
            config=self.config_obj,
            output_dir=self.output_dir,
            random_state=self.random_state,
        )
        self._runner = HoldoutRunner(workflow=self, plan=self._plan)
        self._run_result: Optional[RunResult] = None

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
        """
        Train every configured model on the train split and persist outputs.

        The training path uses explicit reporting components instead of callback-
        driven side effects, while preserving the historical output files.
        """
        self.logger.info("Starting Standard ML Pipeline (mode=train)...")
        self.data_manager.load_data()
        self._run_result = self._runner.run()

        # Keep public attributes available for compatibility.
        self.X_train = self._run_result.x_train
        self.X_test = self._run_result.x_test
        self.results = self._run_result.to_legacy_results()

        # Persist outputs in explicit order: models -> reports -> plots.
        model_store = ModelStore(
            output_dir=self.output_dir,
            is_save_model=bool(getattr(self.config_obj, "is_save_model", True)),
        )
        report_writer = ReportWriter(
            output_dir=self.output_dir,
            module_name=self.module_name,
        )
        plot_composer = PlotComposer(
            plot_manager=self.plot_manager,
            is_visualize=bool(getattr(self.config_obj, "is_visualize", True)),
        )

        model_store.save(self._run_result)
        report_writer.write(self._run_result)
        plot_composer.render(self._run_result)

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
