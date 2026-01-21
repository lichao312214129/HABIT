import os
import pandas as pd
from typing import Dict, Any, List, Tuple
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
    def __init__(self, config: Dict[str, Any], module_name: str):
        self.module_name = module_name
        # Validate configuration early using unified validator
        try:
            # Try to validate as MLConfig
            if isinstance(config, MLConfig):
                self.config_obj = config
            else:
                self.config_obj = ConfigValidator.validate_dict(config, MLConfig, strict=True)
            # Use ConfigAccessor for unified access
            self.config_accessor = ConfigAccessor(self.config_obj)
            # For backward compatibility, also provide dict access
            self.config = self.config_obj.to_dict()
        except Exception as e:
            # Fallback to dict for backward compatibility (with warning)
            import warnings
            warnings.warn(
                f"Configuration validation failed: {e}. "
                f"Falling back to dictionary access. "
                f"Please update your configuration to use proper schema.",
                UserWarning
            )
            self.config = config
            self.config_obj = None
            self.config_accessor = ConfigAccessor(config)
            
        # Get output directory using unified accessor
        self.output_dir = self.config_accessor.get('output', f'./results/{module_name}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Logging
        manager = LoggerManager()
        if manager.get_log_file() is not None:
            self.logger = get_module_logger(module_name)
        else:
            self.logger = setup_logger(module_name, self.output_dir, f'{module_name}.log')
            
        # Common Components - pass config dict for backward compatibility
        self.data_manager = DataManager(self.config, self.logger)
        self.plot_manager = PlotManager(self.config, self.output_dir)
        self.pipeline_builder = PipelineBuilder(self.config, self.output_dir)
        self.random_state = self.config_accessor.get('random_state', 42)
        
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