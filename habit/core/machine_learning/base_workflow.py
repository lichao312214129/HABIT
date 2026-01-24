import os
import pandas as pd
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

    @abstractmethod
    def run_pipeline(self):
        """Main entry point to be implemented by subclasses."""
        pass

    def _save_common_results(self, summary_df: pd.DataFrame, prefix: str = ""):
        """Saves common result artifacts."""
        save_json(self.results, os.path.join(self.output_dir, f'{prefix}results.json'))
        save_csv(summary_df, os.path.join(self.output_dir, f'{prefix}summary.csv'))