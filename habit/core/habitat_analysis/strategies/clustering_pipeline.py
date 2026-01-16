"""
Pipeline classes for Habitat Analysis.

This module implements the Strategy pattern to separate training and testing
logic, making the code more maintainable and extensible.
"""

import os
import json
import pickle
import logging
import shutil
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from ..config import HabitatConfig, ResultColumns


class BasePipeline(ABC):
    """
    Abstract base class for habitat analysis pipelines.
    
    This class defines the interface for both training and testing pipelines,
    ensuring consistent behavior across different modes.
    """
    
    def __init__(
        self,
        config: HabitatConfig,
        logger: logging.Logger,
    ):
        """
        Initialize pipeline.
        
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


class TrainingPipeline(BasePipeline):
    """
    Pipeline for training mode: determines optimal clusters and saves models.
    """
    
    def cluster_habitats(
        self,
        features: pd.DataFrame,
        clustering_algorithm: Any,
    ) -> Tuple[np.ndarray, int, Optional[Dict]]:
        """
        Perform population-level clustering with optimal cluster selection.
        
        Args:
            features: DataFrame of supervoxel features for clustering
            clustering_algorithm: Clustering algorithm instance
            
        Returns:
            Tuple of (habitat_labels, optimal_n_clusters, scores_dict)
        """
        from habit.core.habitat_analysis.clustering.base_clustering import (
            get_clustering_algorithm
        )
        
        scores = None
        
        # Check if best_n_clusters is already specified
        if self.config.clustering.best_n_clusters is not None:
            optimal_n_clusters = self.config.clustering.best_n_clusters
            if self.config.runtime.verbose:
                self.logger.info(
                    f"Using specified best number of clusters: {optimal_n_clusters}"
                )
        else:
            # Find optimal number of clusters
            if self.config.runtime.verbose:
                self.logger.info("Finding optimal number of clusters...")
            
            optimal_n_clusters, scores = self._find_optimal_clusters(
                features, clustering_algorithm
            )
        
        if self.config.runtime.verbose:
            self.logger.info(f"Optimal number of clusters: {optimal_n_clusters}")
            self.logger.info("Performing population-level clustering...")
        
        # Perform clustering with optimal number
        clustering_algorithm.n_clusters = optimal_n_clusters
        clustering_algorithm.fit(features)
        habitat_labels = clustering_algorithm.predict(features) + 1  # Start from 1
        
        return habitat_labels, optimal_n_clusters, scores
    
    def _find_optimal_clusters(
        self,
        features: pd.DataFrame,
        clustering_algorithm: Any,
    ) -> Tuple[int, Optional[Dict]]:
        """
        Find optimal number of clusters using validation methods.
        
        Args:
            features: Feature DataFrame
            clustering_algorithm: Clustering algorithm instance
            
        Returns:
            Tuple of (optimal_n_clusters, scores_dict)
        """
        from habit.core.habitat_analysis.clustering.base_clustering import (
            get_clustering_algorithm
        )
        
        try:
            min_clusters = max(2, self.config.clustering.n_clusters_habitats_min)
            max_clusters = min(
                self.config.clustering.n_clusters_habitats_max,
                len(features) - 1
            )
            
            if max_clusters <= min_clusters:
                if self.config.runtime.verbose:
                    self.logger.warning(
                        f"Invalid cluster range [{min_clusters}, {max_clusters}], "
                        "using default value"
                    )
                return min_clusters, None
            
            # Create new clustering algorithm for optimization
            cluster_for_best_n = get_clustering_algorithm(
                self.config.clustering.habitat_method
            )
            
            optimal_n_clusters, scores = cluster_for_best_n.find_optimal_clusters(
                features,
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                methods=self.config.clustering.selection_methods,
                show_progress=True
            )
            
            return optimal_n_clusters, scores
            
        except Exception as e:
            if self.config.runtime.verbose:
                self.logger.error(
                    f"Exception when determining optimal clusters: {e}"
                )
                self.logger.info("Using default number of clusters")
            return 3, None
    
    def save_model(self, model: Any, model_name: str) -> None:
        """Save trained model to pickle file."""
        model_path = os.path.join(self.out_dir, f'{model_name}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        if self.config.runtime.verbose:
            self.logger.info(f"Model saved to: {model_path}")
    
    def load_model(self, model_name: str) -> Any:
        """Load model from pickle file (for testing mode compatibility)."""
        model_path = os.path.join(self.out_dir, f'{model_name}.pkl')
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def save_mean_values(self, features: pd.DataFrame) -> None:
        """
        Save mean values for handling missing values in test data.
        
        Args:
            features: Feature DataFrame
        """
        mean_values = features.mean()
        mean_values.name = 'mean_values'
        mean_path = os.path.join(
            self.out_dir, 'mean_values_of_all_supervoxels_features.csv'
        )
        mean_values.to_csv(mean_path, index=True, header=True)
        if self.config.runtime.verbose:
            self.logger.info(f"Mean values saved to: {mean_path}")


class TestingPipeline(BasePipeline):
    """
    Pipeline for testing mode: loads pre-trained models and applies them.
    """
    
    def cluster_habitats(
        self,
        features: pd.DataFrame,
        clustering_algorithm: Any,
    ) -> Tuple[np.ndarray, int, Optional[Dict]]:
        """
        Apply pre-trained clustering model to new data.
        
        Args:
            features: DataFrame of supervoxel features for clustering
            clustering_algorithm: Clustering algorithm instance (not used, loaded instead)
            
        Returns:
            Tuple of (habitat_labels, n_clusters, None)
        """
        # Load pre-trained model
        model = self.load_model('supervoxel2habitat_clustering_strategy')
        
        if self.config.runtime.verbose:
            model_path = os.path.join(
                self.out_dir, 'supervoxel2habitat_clustering_strategy.pkl'
            )
            self.logger.info(
                f"Performing clustering using pre-trained model: {model_path}"
            )
        
        # Apply model
        habitat_labels = model.predict(features) + 1  # Start from 1
        optimal_n_clusters = model.n_clusters
        
        return habitat_labels, optimal_n_clusters, None
    
    def save_model(self, model: Any, model_name: str) -> None:
        """Testing mode doesn't save models - no-op."""
        pass
    
    def load_model(self, model_name: str) -> Any:
        """Load pre-trained model from pickle file."""
        model_path = os.path.join(self.out_dir, f'{model_name}.pkl')
        
        if not os.path.exists(model_path):
            raise ValueError(f"No clustering model found at {model_path}")
        
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def load_mean_values(self) -> pd.Series:
        """
        Load mean values for handling missing values.
        
        Returns:
            Series of mean values
        """
        mean_path = os.path.join(
            self.out_dir, 'mean_values_of_all_supervoxels_features.csv'
        )
        return pd.read_csv(mean_path, index_col=0).squeeze()


def create_pipeline(
    config: HabitatConfig,
    logger: logging.Logger,
) -> BasePipeline:
    """
    Factory function to create appropriate pipeline based on mode.
    
    Args:
        config: Habitat analysis configuration
        logger: Logger instance
        
    Returns:
        BasePipeline: Either TrainingPipeline or TestingPipeline
        
    Raises:
        ValueError: If mode is invalid
    """
    if config.runtime.mode == 'training':
        return TrainingPipeline(config, logger)
    elif config.runtime.mode == 'testing':
        return TestingPipeline(config, logger)
    else:
        raise ValueError(f"Invalid mode: {config.runtime.mode}")
