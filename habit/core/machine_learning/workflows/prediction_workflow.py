"""
Prediction Workflow Module
Handles loading trained models and making predictions on new data.
"""

import os
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Dict, Any, List

from habit.core.machine_learning.config_schemas import PredictionConfig
from habit.core.machine_learning.evaluation.prediction_container import PredictionContainer
from habit.core.machine_learning.evaluation.metrics import calculate_metrics

class PredictionWorkflow:
    """
    Workflow for running model predictions.
    
    Attributes:
        config (PredictionConfig): Configuration object.
        logger (logging.Logger): Logger instance.
    """

    def __init__(self, config: PredictionConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the prediction workflow.

        Args:
            config: Prediction configuration.
            logger: Logger instance.
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def _find_label_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find the ground truth label column based on configuration.
        
        Args:
            df: Input dataframe
            
        Returns:
            Column name if found, None otherwise
        """
        # First, try explicitly configured label column
        if self.config.label_col:
            if self.config.label_col in df.columns:
                return self.config.label_col
            else:
                self.logger.warning(
                    f"Configured label column '{self.config.label_col}' not found in data. "
                    f"Trying fallback candidates."
                )
        
        # Fallback to candidates
        for col in self.config.label_col_candidates:
            if col in df.columns:
                return col
        
        return None

    def run_pipeline(self) -> None:
        """Run the full prediction pipeline."""
        try:
            self.logger.info("Starting prediction pipeline")
            
            # 1. Load Data
            self.logger.info(f"Loading data from {self.config.data_path}")
            if not os.path.exists(self.config.data_path):
                raise FileNotFoundError(f"Data file not found: {self.config.data_path}")
            
            df = pd.read_csv(self.config.data_path)
            self.logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # 2. Load Model
            self.logger.info(f"Loading model from {self.config.model_path}")
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(f"Model file not found: {self.config.model_path}")
            
            pipeline = joblib.load(self.config.model_path)
            
            # 3. Make Predictions
            self.logger.info("Making predictions...")
            
            # Separate features (X) from labels (y) if evaluation is requested
            label_col = self._find_label_column(df)
            
            if self.config.evaluate and label_col:
                self.logger.info(f"Evaluation enabled. Using column '{label_col}' as ground truth.")
                y_true = df[label_col].values
                # We assume the pipeline handles column selection (FeatureSelectTransformer), 
                # so passing the full DF including label col is usually fine, 
                # BUT to be safe, we should drop it if the model wasn't trained with it.
                # However, our FeatureSelectTransformer selects by name, so extra columns are ignored.
                X = df
            elif self.config.evaluate and not label_col:
                self.logger.warning("Evaluation enabled but no common label column found. Skipping evaluation.")
                X = df
                y_true = None
            else:
                X = df
                y_true = None

            # Get Predictions
            preds = pipeline.predict(X)
            
            # Extract probabilities - use raw predict_proba output
            probs_raw = None
            if hasattr(pipeline, "predict_proba"):
                try:
                    probs_raw = pipeline.predict_proba(X)
                except Exception as e:
                    self.logger.warning(f"Failed to predict probabilities: {e}")
            
            # 4. Evaluation and probability extraction using PredictionContainer
            metrics_results = {}
            probs_for_output = None
            
            if probs_raw is not None:
                # Always use PredictionContainer for consistent probability handling
                # If y_true is available, use it; otherwise use predictions as pseudo y_true
                y_for_container = y_true if y_true is not None else preds
                
                try:
                    container = PredictionContainer(
                        y_true=y_for_container,  # Use y_true if available, otherwise use preds to infer class structure
                        y_prob=probs_raw,
                        y_pred=preds
                    )
                    
                    # Extract probabilities using container's logic
                    probs_for_output = container.get_eval_probs()
                    
                    # Calculate metrics if evaluation is enabled and y_true is available
                    if self.config.evaluate and y_true is not None:
                        metrics_results = calculate_metrics(container)
                        self.logger.info("Evaluation Metrics:")
                        for k, v in metrics_results.items():
                            self.logger.info(f"  {k}: {v}")
                            
                except Exception as e:
                    self.logger.error(f"Failed to use PredictionContainer: {e}")
                    # Fallback to raw probabilities
                    probs_for_output = probs_raw

            # 5. Save Results
            results = df.copy()
            
            # Append results using configured column names
            results[self.config.output_label_col] = preds
            if probs_for_output is not None:
                # Handle multiclass probs for CSV
                if probs_for_output.ndim > 1:
                    # For multiclass, save as list (pandas can handle this)
                    # Alternatively, could save as separate columns per class
                    results[self.config.output_prob_col] = list(probs_for_output)
                else:
                    results[self.config.output_prob_col] = probs_for_output
            
            output_path = os.path.join(self.config.output_dir, 'prediction_results.csv')
            results.to_csv(output_path, index=False)
            
            self.logger.info(f"Saved prediction results to {output_path}")
            
            if self.config.evaluate and metrics_results:
                 metrics_path = os.path.join(self.config.output_dir, 'evaluation_metrics.csv')
                 pd.DataFrame([metrics_results]).to_csv(metrics_path, index=False)
                 self.logger.info(f"Saved evaluation metrics to {metrics_path}")
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}", exc_info=True)
            raise
