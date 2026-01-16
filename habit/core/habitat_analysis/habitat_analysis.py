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
from .config import HabitatConfig, ResultColumns, ClusteringConfig, IOConfig, RuntimeConfig
from .strategies.clustering_pipeline import create_pipeline
from .strategies import get_strategy
from .managers import FeatureManager, ClusteringManager, ResultManager

class HabitatAnalysis:
    """
    Habitat Analysis class for performing clustering analysis on medical images.
    
    Acts as a coordinator for FeatureManager, ClusteringManager, and ResultManager.
    """

    def __init__(
        self,
        config: Optional[HabitatConfig] = None,
        # Legacy parameters for backward compatibility
        root_folder: Optional[str] = None,
        out_folder: Optional[str] = None,
        feature_config: Optional[Dict[str, Any]] = None,
        clustering_strategy: str = "two_step",
        supervoxel_clustering_method: str = "kmeans",
        habitat_clustering_method: str = "kmeans",
        mode: str = "training",
        n_clusters_supervoxel: int = 50,
        n_clusters_habitats_max: int = 10,
        n_clusters_habitats_min: int = 2,
        habitat_cluster_selection_method: Optional[Union[str, List[str]]] = None,
        best_n_clusters: Optional[int] = None,
        one_step_settings: Optional[Dict[str, Any]] = None,
        n_processes: int = 1,
        random_state: int = 42,
        verbose: bool = True,
        images_dir: str = "images",
        masks_dir: str = "masks",
        plot_curves: bool = True,
        progress_callback: Optional[callable] = None,
        save_intermediate_results: bool = False,
        config_file: Optional[str] = None,
        log_level: str = "INFO"
    ):
        """Initialize HabitatAnalysis."""
        if config is None:
            config = self._build_config_from_legacy(
                root_folder, out_folder, feature_config, clustering_strategy,
                supervoxel_clustering_method, habitat_clustering_method, mode,
                n_clusters_supervoxel, n_clusters_habitats_max, n_clusters_habitats_min,
                habitat_cluster_selection_method, best_n_clusters, one_step_settings,
                n_processes, random_state, verbose, images_dir, masks_dir,
                plot_curves, progress_callback, save_intermediate_results,
                config_file, log_level
            )
        
        self.config = config
        self._setup_logging()
        
        # Initialize Managers
        self.feature_manager = FeatureManager(config, self.logger)
        self.clustering_manager = ClusteringManager(config, self.logger)
        self.result_manager = ResultManager(config, self.logger)
        
        # Setup Data
        self._setup_data_paths()
        
        # Initialize Pipeline
        self.pipeline = create_pipeline(self.config, self.logger)
        
        # Pass logging info to managers for subprocesses
        self.feature_manager.set_logging_info(self._log_file_path, self._log_level)
        self.result_manager.set_logging_info(self._log_file_path, self._log_level)

    def _build_config_from_legacy(self, *args, **kwargs) -> HabitatConfig:
        """Helper to build HabitatConfig from legacy parameters."""
        # Map args to dict based on signature order if needed, or just use kwargs 
        # But since we passed them explicitly in __init__, we construct dict here
        # For simplicity, I'll rely on the arguments passed to this helper
        
        # Note: This method signature is simplified here, but in __init__ we pass all args.
        # It's cleaner to construct the dict in __init__ or have a static factory.
        # Let's revert to constructing inside __init__ using locals() or explicit dict
        # to avoid massive signature duplication.
        # Re-implementing __init__ with cleaner logic.
        return HabitatConfig.from_dict({
            'root_folder': args[0],
            'out_folder': args[1],
            'feature_config': args[2] or {},
            'clustering_strategy': args[3],
            'supervoxel_clustering_method': args[4],
            'habitat_clustering_method': args[5],
            'mode': args[6],
            'n_clusters_supervoxel': args[7],
            'n_clusters_habitats_max': args[8],
            'n_clusters_habitats_min': args[9],
            'habitat_cluster_selection_method': args[10],
            'best_n_clusters': args[11],
            'one_step_settings': args[12],
            'n_processes': args[13],
            'random_state': args[14],
            'verbose': args[15],
            'images_dir': args[16],
            'masks_dir': args[17],
            'plot_curves': args[18],
            'progress_callback': args[19],
            'save_intermediate_results': args[20],
            'config_file': args[21],
            'log_level': args[22],
        })
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
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
            level = self.config.runtime.get_log_level_int()
            self.logger = setup_logger(
                name='habitat',
                output_dir=self.config.io.out_folder,
                log_filename='habitat_analysis.log',
                level=level
            )
            self._log_file_path = manager.get_log_file()
            self._log_level = level
    
    def _setup_data_paths(self) -> None:
        """Setup data paths and create output directory."""
        os.makedirs(self.config.io.out_folder, exist_ok=True)
        
        # Get image and mask paths
        images_paths, mask_paths = get_image_and_mask_paths(
            self.config.io.root_folder,
            keyword_of_raw_folder=self.config.io.images_dir,
            keyword_of_mask_folder=self.config.io.masks_dir
        )
        
        # Auto-detect image names if not provided
        voxel_config = self.config.feature_config['voxel_level']
        if isinstance(voxel_config, dict) and 'image_names' not in voxel_config:
            if self.config.runtime.verbose:
                self.logger.info(
                    "Image names not provided in voxel_level config, "
                    "automatically detecting from data directory..."
                )
            image_names = detect_image_names(images_paths)
            self.config.feature_config['voxel_level']['image_names'] = image_names
            if self.config.runtime.verbose:
                self.logger.info(f"Detected image names: {image_names}")
        
        # Validate data structure
        voxel_config = self.config.feature_config['voxel_level']
        if isinstance(voxel_config, dict) and 'image_names' in voxel_config:
            check_data_structure(
                images_paths, 
                mask_paths,
                voxel_config['image_names'], 
                None
            )
        
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
        strategy_class = get_strategy(self.config.clustering.strategy)
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