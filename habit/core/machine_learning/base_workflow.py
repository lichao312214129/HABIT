import os
import logging
import pandas as pd
from typing import Dict, Any, List, Tuple
from abc import ABC, abstractmethod

from .data_manager import DataManager
from .visualization.plot_manager import PlotManager
from habit.utils.log_utils import get_module_logger, setup_logger, LoggerManager
from habit.utils.io_utils import save_json, save_csv

class BaseWorkflow(ABC):
    """
    Abstract Base Class for all Machine Learning Workflows.
    Handles infrastructure like logging, data loading, and basic results persistence.
    """
    def __init__(self, config: Dict[str, Any], module_name: str):
        self.config = config
        self.output_dir = config.get('output', f'./results/{module_name}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Logging
        manager = LoggerManager()
        if manager.get_log_file() is not None:
            self.logger = get_module_logger(module_name)
        else:
            self.logger = setup_logger(module_name, self.output_dir, f'{module_name}.log')
            
        # Common Components
        self.data_manager = DataManager(config, self.logger)
        self.plot_manager = PlotManager(config, self.output_dir)
        self.random_state = config.get('random_state', 42)
        
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
