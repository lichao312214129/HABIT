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
from .strategies import get_strategy
from .managers import FeatureManager, ClusteringManager, ResultManager

class HabitatAnalysis:
    """
    Habitat Analysis class for performing clustering analysis on medical images.
    
    Acts as a coordinator for FeatureManager, ClusteringManager, and ResultManager.
    
    Note: Dependencies should be provided via ServiceConfigurator or explicitly.
    """

    def __init__(
        self,
        config: Union[Dict[str, Any], HabitatAnalysisConfig],
        feature_manager: FeatureManager,
        clustering_manager: ClusteringManager,
        result_manager: ResultManager,
        logger: Any,
    ):
        """
        Initialize HabitatAnalysis.
        
        Args:
            config: Configuration dictionary or HabitatAnalysisConfig instance.
            feature_manager: FeatureManager instance (required).
            clustering_manager: ClusteringManager instance (required).
            result_manager: ResultManager instance (required).
            logger: Logger instance (required).
        """
        if isinstance(config, HabitatAnalysisConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = HabitatAnalysisConfig.model_validate(config)
        else:
            raise TypeError("config must be a dict or HabitatAnalysisConfig")

        self.feature_manager = feature_manager
        self.clustering_manager = clustering_manager
        self.result_manager = result_manager
        self.logger = logger
        
        self._setup_logging_info()
        self._setup_data_paths()
        
        self.feature_manager.set_logging_info(self._log_file_path, self._log_level)
        self.result_manager.set_logging_info(self._log_file_path, self._log_level)

    def _setup_logging_info(self) -> None:
        """Get logging info from injected logger or create defaults."""
        manager = LoggerManager()
        
        log_file = manager.get_log_file()
        if log_file:
            self._log_file_path = log_file
        elif hasattr(self.logger, 'log_file'):
            self._log_file_path = self.logger.log_file
        else:
            self._log_file_path = os.path.join(self.config.out_dir, 'habitat_analysis.log')
        
        if manager._root_logger:
            self._log_level = manager._root_logger.getEffectiveLevel()
        else:
            self._log_level = logging.INFO
    
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
        save_results_csv: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Run the habitat clustering pipeline.
        
        Args:
            subjects: List of subjects to process (None = all subjects)
            save_results_csv: Whether to save results as CSV files (defaults to config.save_results_csv)
            
        Returns:
            DataFrame with habitat clustering results
        """
        # Use config value if parameter not provided, allowing runtime override
        if save_results_csv is None:
            save_results_csv = self.config.save_results_csv
        
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