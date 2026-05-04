"""
Inference runner for the prediction-only path.

The :class:`InferenceRunner` reuses the same Plan/Result/Reporting seam as
the training runners, so the historical "predict" flow no longer bypasses
the runner layer.  It loads a saved pipeline, runs predictions and produces
an :class:`InferenceResult` that the reporting layer can persist.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from ..core.results import InferenceResult
from ..evaluation.metrics import calculate_metrics
from ..evaluation.prediction_container import PredictionContainer
from .base import BaseRunner


# Conventional fallback columns inspected when the configured label column
# cannot be found in the inference dataframe.
_LABEL_FALLBACK_COLUMNS = ("label", "Target", "class", "diagnosis", "outcome", "y")


class InferenceRunner(BaseRunner):
    """
    Execute the predict path: load -> predict -> optional evaluation.
    """

    def run(self) -> InferenceResult:
        """
        Run prediction on the inference input declared in the config.

        Returns
        -------
        InferenceResult
            Structured output with the prediction dataframe and (optionally)
            evaluation metrics.

        Raises
        ------
        FileNotFoundError
            If the saved pipeline file or the input dataframe file is missing.
        ValueError
            If ``config.input`` is empty.
        """
        cfg = self.plan.config
        logger = self.context.logger

        pipeline_path = self._require_pipeline_path(cfg)
        input_path, configured_label_col = self._require_input(cfg)

        logger.info("Loading data from %s", input_path)
        df = self.context.data_manager.load_inference_data(input_path)
        logger.info("Data loaded: %s rows, %s columns", df.shape[0], df.shape[1])

        logger.info("Loading pipeline from %s", pipeline_path)
        pipeline = joblib.load(pipeline_path)

        label_col = self._resolve_label_column(df, configured_label_col)
        evaluate = bool(getattr(cfg, "evaluate", False))
        if evaluate and label_col is None:
            logger.warning(
                "evaluate=True but no label column resolved; skipping evaluation."
            )
            evaluate = False

        y_true = df[label_col].values if (evaluate and label_col) else None
        preds = pipeline.predict(df)

        probs_raw = self._safe_predict_proba(pipeline, df)
        probs_for_output, metrics = self._extract_probs_and_metrics(
            preds=preds,
            probs_raw=probs_raw,
            y_true=y_true,
            evaluate=evaluate,
        )

        predictions_df = self._build_predictions_dataframe(
            df=df,
            preds=preds,
            probs_for_output=probs_for_output,
            output_label_col=cfg.output_label_col,
            output_prob_col=cfg.output_prob_col,
        )

        return InferenceResult.create(
            plan=self.plan,
            pipeline_path=pipeline_path,
            predictions=predictions_df,
            metrics=metrics,
            label_col=label_col if evaluate else None,
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_pipeline_path(cfg: Any) -> str:
        """Return ``cfg.pipeline_path`` or raise a descriptive error."""
        pipeline_path = getattr(cfg, "pipeline_path", None)
        if not pipeline_path or not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Saved pipeline not found: {pipeline_path}")
        return str(pipeline_path)

    @staticmethod
    def _require_input(cfg: Any) -> tuple:
        """Return (input_path, configured_label_col) from cfg.input[0]."""
        if not getattr(cfg, "input", None):
            raise ValueError(
                "InferenceRunner requires at least one entry in config.input."
            )
        input_cfg = cfg.input[0]
        data_path = input_cfg.path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Input data file not found: {data_path}")
        return data_path, getattr(input_cfg, "label_col", None)

    def _resolve_label_column(
        self,
        df: pd.DataFrame,
        configured_label_col: Optional[str],
    ) -> Optional[str]:
        """Resolve ground-truth column with a fallback list."""
        if configured_label_col and configured_label_col in df.columns:
            return configured_label_col
        if configured_label_col:
            self.context.logger.warning(
                "Configured label column '%s' not found in data; trying fallback.",
                configured_label_col,
            )
        for col in _LABEL_FALLBACK_COLUMNS:
            if col in df.columns:
                return col
        return None

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _safe_predict_proba(self, pipeline: Any, df: pd.DataFrame):
        """Call ``predict_proba`` defensively, logging any failure."""
        if not hasattr(pipeline, "predict_proba"):
            return None
        try:
            return pipeline.predict_proba(df)
        except Exception as exc:  # pragma: no cover - defensive
            self.context.logger.warning("Failed to predict probabilities: %s", exc)
            return None

    def _extract_probs_and_metrics(
        self,
        preds,
        probs_raw,
        y_true,
        evaluate: bool,
    ):
        """
        Use :class:`PredictionContainer` to derive the output probabilities
        and (optionally) compute evaluation metrics.
        """
        if probs_raw is None:
            return None, {}

        try:
            y_for_container = y_true if y_true is not None else preds
            container = PredictionContainer(
                y_true=y_for_container,
                y_prob=probs_raw,
                y_pred=preds,
            )
            probs_for_output = container.get_eval_probs()

            metrics: Dict[str, float] = {}
            if evaluate and y_true is not None:
                metrics = dict(calculate_metrics(container))
                self.context.logger.info("Evaluation Metrics:")
                for k, v in metrics.items():
                    self.context.logger.info("  %s: %s", k, v)
            return probs_for_output, metrics
        except Exception as exc:  # pragma: no cover - defensive
            self.context.logger.error("Failed to use PredictionContainer: %s", exc)
            return probs_raw, {}

    @staticmethod
    def _build_predictions_dataframe(
        df: pd.DataFrame,
        preds,
        probs_for_output,
        output_label_col: str,
        output_prob_col: str,
    ) -> pd.DataFrame:
        """Append predicted labels (and probabilities, when available)."""
        results = df.copy()
        results[output_label_col] = preds
        if probs_for_output is not None:
            if hasattr(probs_for_output, "ndim") and probs_for_output.ndim > 1:
                # Multiclass path - persist as list cells (CSV-friendly).
                results[output_prob_col] = list(probs_for_output)
            else:
                results[output_prob_col] = probs_for_output
        return results
