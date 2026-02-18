import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Union
from abc import ABC, abstractmethod

from .data_manager import DataManager
from .visualization.plot_manager import PlotManager
from .pipeline_utils import PipelineBuilder
from .config_schemas import MLConfig, validate_config
from .callbacks.base import CallbackList
from .callbacks.model_checkpoint import ModelCheckpoint
from .callbacks.visualization_callback import VisualizationCallback
from .callbacks.report_callback import ReportCallback
from habit.utils.log_utils import get_module_logger, setup_logger, LoggerManager
from habit.utils.io_utils import save_json, save_csv
from habit.core.common.config_validator import ConfigValidator
from habit.core.common.config_base import ConfigAccessor

class BaseWorkflow(ABC):
    """
    Abstract Base Class for all Machine Learning Workflows.
    Handles infrastructure like logging, data loading, and basic results persistence.
    """
    def __init__(self, config: Union[MLConfig, Dict[str, Any]], module_name: str):
        """
        Initialize BaseWorkflow.
        
        Args:
            config: MLConfig Pydantic object or dict (dict will be validated and converted).
            module_name: Name of the workflow module.
        """
        self.module_name = module_name
        # Validate configuration early using unified validator
        # Ensure we always have a valid MLConfig object
        if isinstance(config, MLConfig):
            self.config_obj = config
        elif isinstance(config, dict):
            # Validate dictionary and convert to MLConfig
            try:
                self.config_obj = ConfigValidator.validate_dict(config, MLConfig, strict=True)
            except Exception as e:
                raise ValueError(
                    f"Configuration validation failed for {module_name}: {e}. "
                    f"Please ensure your configuration matches the MLConfig schema. "
                    f"Use MLConfig.from_file() or ConfigValidator.validate_and_load() to load configuration."
                ) from e
        else:
            raise TypeError(
                f"Invalid configuration type for {module_name}: expected MLConfig or dict, "
                f"got {type(config)}"
            )
        
        # Use ConfigAccessor for unified access
        self.config_accessor = ConfigAccessor(self.config_obj)
        # Keep dict access only for backward compatibility (deprecated, prefer config_obj)
        self.config = self.config_obj.to_dict()
            
        # Get output directory from Pydantic object
        self.output_dir = getattr(self.config_obj, 'output', f'./results/{module_name}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Logging
        manager = LoggerManager()
        if manager.get_log_file() is not None:
            self.logger = get_module_logger(module_name)
        else:
            self.logger = setup_logger(module_name, self.output_dir, f'{module_name}.log')
            
        # Common Components - pass Pydantic config object
        # DataManager and PlotManager now expect Pydantic objects
        self.data_manager = DataManager(self.config_obj, self.logger)
        self.plot_manager = PlotManager(self.config_obj, self.output_dir)
        # Pass Pydantic model to PipelineBuilder
        self.pipeline_builder = PipelineBuilder(self.config_obj, self.output_dir)
        self.random_state = getattr(self.config_obj, 'random_state', 42)
        
        # Callbacks
        self.callbacks = CallbackList([
            ModelCheckpoint(),
            ReportCallback(),
            VisualizationCallback()
        ], workflow=self)
        
        # Results storage
        self.results = {}

    def _load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Common data loading entry point."""
        self.logger.info("Loading and preparing data...")
        self.data_manager.load_data()
        X = self.data_manager.data.drop(columns=[self.data_manager.label_col])
        y = self.data_manager.data[self.data_manager.label_col]
        return X, y

    def _resample_training_data(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply optional resampling on training data only.

        Supported methods:
        - random_over: random oversampling of minority class
        - random_under: random undersampling of majority class
        - smote: SMOTE oversampling (requires imbalanced-learn)

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Resampled (or original) training data.
        """
        sampling_cfg = getattr(self.config_obj, "sampling", None)
        if sampling_cfg is None or not getattr(sampling_cfg, "enabled", False):
            return X_train, y_train

        X_df: pd.DataFrame = X_train.copy()
        y_series: pd.Series = pd.Series(y_train).copy()
        y_series.index = X_df.index

        class_counts = y_series.value_counts()
        if class_counts.shape[0] != 2:
            self.logger.warning(
                "Sampling is designed for binary labels. Found %d classes; skipping sampling.",
                class_counts.shape[0],
            )
            return X_df, y_series

        majority_label = class_counts.idxmax()
        minority_label = class_counts.idxmin()
        majority_count = int(class_counts[majority_label])
        minority_count = int(class_counts[minority_label])

        method = str(getattr(sampling_cfg, "method", "random_over")).strip().lower()
        ratio = float(getattr(sampling_cfg, "ratio", 1.0))
        random_state = int(getattr(sampling_cfg, "random_state", self.random_state))

        self.logger.info(
            "Sampling enabled: method=%s, ratio=%.4f, before_counts={%s: %d, %s: %d}",
            method,
            ratio,
            str(majority_label),
            majority_count,
            str(minority_label),
            minority_count,
        )

        if method == "random_over":
            target_minority_count = int(np.ceil(majority_count * ratio))
            if target_minority_count <= minority_count:
                self.logger.info("random_over skipped: minority already >= target.")
                return X_df, y_series

            minority_indices = y_series[y_series == minority_label].index
            sampled_indices = pd.Series(minority_indices).sample(
                n=target_minority_count - minority_count,
                replace=True,
                random_state=random_state,
            ).values

            X_add = X_df.loc[sampled_indices]
            y_add = y_series.loc[sampled_indices]
            X_res = pd.concat([X_df, X_add], axis=0)
            y_res = pd.concat([y_series, y_add], axis=0)

        elif method == "random_under":
            if ratio > 1.0:
                raise ValueError("For random_under, sampling.ratio must be <= 1.0")

            target_majority_count = int(np.floor(minority_count / ratio))
            if target_majority_count >= majority_count:
                self.logger.info("random_under skipped: majority already <= target.")
                return X_df, y_series

            majority_indices = y_series[y_series == majority_label].index
            minority_indices = y_series[y_series == minority_label].index

            kept_majority_indices = pd.Series(majority_indices).sample(
                n=target_majority_count,
                replace=False,
                random_state=random_state,
            ).values
            kept_indices = np.concatenate([kept_majority_indices, minority_indices.values])

            X_res = X_df.loc[kept_indices]
            y_res = y_series.loc[kept_indices]

        elif method == "smote":
            try:
                from imblearn.over_sampling import SMOTE  # type: ignore
            except Exception as e:
                raise ImportError(
                    "SMOTE requires imbalanced-learn. Install with `pip install imbalanced-learn`."
                ) from e

            smote = SMOTE(sampling_strategy=ratio, random_state=random_state)
            X_sm, y_sm = smote.fit_resample(X_df, y_series)
            X_res = (
                pd.DataFrame(X_sm, columns=X_df.columns)
                if not isinstance(X_sm, pd.DataFrame)
                else X_sm
            )
            y_res = pd.Series(y_sm)
        else:
            raise ValueError(f"Unsupported sampling method: {method}")

        # Shuffle after resampling for stable downstream training.
        permutation = pd.Series(range(len(y_res))).sample(
            frac=1.0, random_state=random_state
        ).values
        X_res = X_res.reset_index(drop=True).iloc[permutation].reset_index(drop=True)
        y_res = y_res.reset_index(drop=True).iloc[permutation].reset_index(drop=True)

        res_counts = y_res.value_counts()
        self.logger.info(
            "Sampling completed: after_counts={%s: %d, %s: %d}",
            str(majority_label),
            int(res_counts.get(majority_label, 0)),
            str(minority_label),
            int(res_counts.get(minority_label, 0)),
        )
        return X_res, y_res

    def _train_with_optional_sampling(
        self, estimator: Any, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Any:
        """
        Unified training entry with optional sampling only.

        Args:
            estimator: Model pipeline/estimator to train.
            X_train: Training features.
            y_train: Training labels.

        Returns:
            Any: Fitted estimator object.
        """
        X_fit, y_fit = self._resample_training_data(X_train, y_train)
        estimator.fit(X_fit, y_fit)
        return estimator

    @abstractmethod
    def run_pipeline(self):
        """Main entry point to be implemented by subclasses."""
        pass

    def _save_common_results(self, summary_df: pd.DataFrame, prefix: str = ""):
        """Saves common result artifacts."""
        save_json(self.results, os.path.join(self.output_dir, f'{prefix}results.json'))
        save_csv(summary_df, os.path.join(self.output_dir, f'{prefix}summary.csv'))