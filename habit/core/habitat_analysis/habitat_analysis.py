"""
Habitat Clustering Analysis Module

This module implements a two-step (or one-step) clustering approach for 
tumor habitat analysis:
1. Individual-level clustering: Divide each tumor into supervoxels
2. Population-level clustering: Cluster supervoxels across patients to obtain habitats
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union

# Suppress warnings
warnings.simplefilter('ignore')

# Internal imports
from habit.utils.io_utils import (
    get_image_and_mask_paths,
    detect_image_names,
    check_data_structure
)
from habit.utils.log_utils import setup_logger, get_module_logger, LoggerManager
from habit.utils.parallel_utils import parallel_map

# Local imports
from .config_schemas import HabitatAnalysisConfig
from .modes import create_mode
from .strategies import get_strategy
from .managers import FeatureManager, ClusteringManager, ResultManager

class HabitatAnalysis:
    """
    Habitat Analysis class for performing clustering analysis on medical images.
    
    Acts as a coordinator for FeatureManager, ClusteringManager, and ResultManager.
    Supports dependency injection for better testability and flexibility.
    """

    def __init__(
        self,
        config: Union[Dict[str, Any], HabitatAnalysisConfig],
        feature_manager: Optional[FeatureManager] = None,
        clustering_manager: Optional[ClusteringManager] = None,
        result_manager: Optional[ResultManager] = None,
        logger: Optional[Any] = None,
    ):
        """
        Initialize HabitatAnalysis from a configuration dictionary or a Pydantic model.
        
        Args:
            config: A dictionary conforming to HabitatAnalysisConfig schema or an instance of it.
            feature_manager: Optional FeatureManager instance (will be created if not provided)
            clustering_manager: Optional ClusteringManager instance (will be created if not provided)
            result_manager: Optional ResultManager instance (will be created if not provided)
            logger: Optional logger instance (will be created if not provided)
        """
        if isinstance(config, HabitatAnalysisConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = HabitatAnalysisConfig.model_validate(config)
        else:
            raise TypeError("config must be a dict or HabitatAnalysisConfig")

        self._setup_logging(logger)
        
        # Initialize Managers - use injected instances or create new ones
        self.feature_manager = feature_manager or FeatureManager(self.config, self.logger)
        self.clustering_manager = clustering_manager or ClusteringManager(self.config, self.logger)
        self.result_manager = result_manager or ResultManager(self.config, self.logger)
        
        # Setup Data
        self._setup_data_paths()
        
        # Initialize Mode Handler
        self.mode_handler = create_mode(self.config, self.logger)
        
        # Pass logging info to managers for subprocesses
        self.feature_manager.set_logging_info(self._log_file_path, self._log_level)
        self.result_manager.set_logging_info(self._log_file_path, self._log_level)

    
    def _setup_logging(self, logger: Optional[Any] = None) -> None:
        """Setup logging configuration."""
        if logger is not None:
            self.logger = logger
            manager = LoggerManager()
            self._log_file_path = manager.get_log_file()
            self._log_level = logging.INFO
            return

        manager = LoggerManager()
        
        if manager.get_log_file() is not None:
            # Logging already configured by CLI
            self.logger = get_module_logger('habitat')
            self.logger.info("Using existing logging configuration from CLI entry point")
            self._log_file_path = manager.get_log_file()
            self._log_level = (
                manager._root_logger.getEffectiveLevel() 
                if manager._root_logger else logging.INFO
            )
        else:
            # Setup new logging
            level = logging.DEBUG if self.config.debug else logging.INFO
            self.logger = setup_logger(
                name='habitat',
                output_dir=self.config.out_dir,
                log_filename='habitat_analysis.log',
                level=level
            )
            self._log_file_path = manager.get_log_file()
            self._log_level = level
    
    def _setup_data_paths(self) -> None:
        """Setup data paths and create output directory."""
        os.makedirs(self.config.out_dir, exist_ok=True)
        
        # Get image and mask paths. Assuming image/mask dir names are now part of file list yaml or default structure
        # This part of logic might need to be adapted if images_dir/masks_dir were important
        images_paths, mask_paths = get_image_and_mask_paths(
            self.config.data_dir,
        )
        
        # Auto-detect image names if not provided
        # This logic needs to adapt as `image_names` is not in the new schema, but part of the `method` string
        # For now, assuming the logic in FeatureManager can handle the `method` string directly
        
        # Pass paths to FeatureManager
        self.feature_manager.set_data_paths(images_paths, mask_paths)
    
    def run(
        self, 
        subjects: Optional[List[str]] = None, 
        save_results_csv: bool = True
    ) -> pd.DataFrame:
        """
        Run the habitat clustering pipeline.
        
        Args:
            subjects: List of subjects to process (None = all subjects)
            save_results_csv: Whether to save results as CSV files
            
        Returns:
            DataFrame with habitat clustering results
        """
        strategy_class = get_strategy(self.config.HabitatsSegmention.clustering_mode)
        strategy = strategy_class(self)
        return strategy.run(subjects=subjects, save_results_csv=save_results_csv)
    
    # =========================================================================
    # Public Facade Methods for Strategies
    # =========================================================================

    # Properties to maintain compatibility and ease access
    @property
    def results_df(self):
        return self.result_manager.results_df
    
    @results_df.setter
    def results_df(self, value):
        self.result_manager.results_df = value
    
    @property
    def supervoxel2habitat_clustering(self):
        return self.clustering_manager.supervoxel2habitat_clustering
    
    @property
    def images_paths(self):
        return self.feature_manager.images_paths
    
    @property
    def mask_paths(self):
        return self.feature_manager.mask_paths