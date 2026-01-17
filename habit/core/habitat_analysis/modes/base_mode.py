"""
Base mode interface for habitat analysis.
"""

import os
import json
import logging
import shutil
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple

from ..config import HabitatConfig

class BaseMode(ABC):
    """
    Abstract base class for habitat analysis modes.
    
    This class defines the interface for both training and testing modes,
    ensuring consistent behavior.
    """
    
    def __init__(
        self,
        config: HabitatConfig,
        logger: logging.Logger,
    ):
        """
        Initialize mode.
        
        Args:
            config: Habitat analysis configuration
            logger: Logger instance for status messages
        """
        self.config = config
        self.logger = logger
        self.out_dir = config.io.out_folder
    
    @abstractmethod
    def cluster_habitats(
        self,
        features: pd.DataFrame,
        clustering_algorithm: Any,
    ) -> Tuple[np.ndarray, int, Optional[Dict]]:
        """
        Perform population-level clustering to determine habitats.
        
        Args:
            features: DataFrame of supervoxel features for clustering
            clustering_algorithm: Clustering algorithm instance
            
        Returns:
            Tuple of (habitat_labels, optimal_n_clusters, scores_dict)
        """
        pass
    
    @abstractmethod
    def process_features(
        self,
        features: pd.DataFrame,
        methods: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Process features using group-level statistics (stateful).
        
        Args:
            features: DataFrame to process
            methods: List of preprocessing method configurations
            
        Returns:
            Processed DataFrame
        """
        pass
    
    @abstractmethod
    def save_model(self, model: Any, model_name: str) -> None:
        """
        Save a trained model.
        
        Args:
            model: Model to save
            model_name: Name for the saved model file
        """
        pass
    
    @abstractmethod
    def load_model(self, model_name: str) -> Any:
        """
        Load a previously saved model.
        
        Args:
            model_name: Name of the model file to load
            
        Returns:
            Loaded model
        """
        pass
    
    def save_config(self, optimal_n_clusters: Optional[int] = None) -> None:
        """
        Save configuration to output directory.
        
        Args:
            optimal_n_clusters: Optimal number of clusters (if determined)
        """
        os.makedirs(self.out_dir, exist_ok=True)
        
        # If original config file exists, copy it
        if self.config.io.config_file and os.path.exists(self.config.io.config_file):
            config_out_path = os.path.join(self.out_dir, 'config.yaml')
            shutil.copy2(self.config.io.config_file, config_out_path)
            if self.config.runtime.verbose:
                self.logger.info(f"Original config file copied to: {config_out_path}")
        else:
            # Save current config as JSON
            if self.config.runtime.verbose:
                self.logger.info(
                    "No original config file provided, saving current config as JSON"
                )
            
            config_dict = self.config.to_dict()
            if optimal_n_clusters is not None:
                config_dict['optimal_n_clusters_habitat'] = int(optimal_n_clusters)
            
            config_path = os.path.join(self.out_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=4)
