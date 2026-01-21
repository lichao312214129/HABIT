"""
Training mode implementation.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

from .base_mode import BaseMode
from ..utils.preprocessing_state import PreprocessingState

class TrainingMode(BaseMode):
    """
    Mode for training: determines optimal clusters and saves models.
    """
    
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.preprocessing_state = PreprocessingState()
    
    def cluster_habitats(
        self,
        features: pd.DataFrame,
        clustering_algorithm: Any,
    ) -> Tuple[np.ndarray, int, Optional[Dict]]:
        """
        Perform population-level clustering with optimal cluster selection.
        """
        # Updated import path
        from habit.core.habitat_analysis.algorithms.base_clustering import (
            get_clustering_algorithm
        )
        
        scores = None
        
        # Check if best_n_clusters is already specified
        if self.config.HabitatsSegmention.habitat.best_n_clusters is not None:
            optimal_n_clusters = self.config.HabitatsSegmention.habitat.best_n_clusters
            if self.config.verbose:
                self.logger.info(
                    f"Using specified best number of clusters: {optimal_n_clusters}"
                )
        else:
            # Find optimal number of clusters
            if self.config.verbose:
                self.logger.info("Finding optimal number of clusters...")
            
            optimal_n_clusters, scores = self._find_optimal_clusters(
                features, clustering_algorithm
            )
        
        if self.config.verbose:
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
        """Find optimal number of clusters using validation methods."""
        # Updated import path
        from habit.core.habitat_analysis.algorithms.base_clustering import (
            get_clustering_algorithm
        )
        
        try:
            min_clusters = max(2, self.config.HabitatsSegmention.habitat.min_clusters or 2)
            max_clusters = min(
                self.config.HabitatsSegmention.habitat.max_clusters,
                len(features) - 1
            )
            
            if max_clusters <= min_clusters:
                if self.config.verbose:
                    self.logger.warning(
                        f"Invalid cluster range [{min_clusters}, {max_clusters}], "
                        "using default value"
                    )
                return min_clusters, None
            
            # Create new clustering algorithm for optimization
            cluster_for_best_n = get_clustering_algorithm(
                self.config.HabitatsSegmention.habitat.algorithm
            )
            
            optimal_n_clusters, scores = cluster_for_best_n.find_optimal_clusters(
                features,
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                methods=self.config.HabitatsSegmention.habitat.habitat_cluster_selection_method,
                show_progress=True
            )
            
            return optimal_n_clusters, scores
            
        except Exception as e:
            if self.config.verbose:
                self.logger.error(
                    f"Exception when determining optimal clusters: {e}"
                )
                self.logger.info("Using default number of clusters")
            return 3, None
            
    def process_features(
        self,
        features: pd.DataFrame,
        methods: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Process features using group-level statistics (Training: fit & transform).
        
        Args:
            features: DataFrame to process
            methods: List of preprocessing method configurations
            
        Returns:
            Processed DataFrame
        """
        if self.config.verbose:
            self.logger.info("Computing and applying group-level preprocessing...")
            
        # Fit state and transform data
        self.preprocessing_state.fit(features, methods)
        return self.preprocessing_state.transform(features)
    
    def save_model(self, model: Any, model_name: str = 'habitat_model') -> None:
        """
        Save trained model and preprocessing state together.
        
        Args:
            model: Clustering model to save
            model_name: Base name for saved files
        """
        # Create bundled training artifacts
        training_bundle = {
            'clustering_model': model,
            'preprocessing_state': self.preprocessing_state,
            'model_name': model_name
        }
        
        # Save to single file
        bundle_path = os.path.join(self.out_dir, f'{model_name}_bundle.pkl')
        with open(bundle_path, 'wb') as f:
            pickle.dump(training_bundle, f)
            
        if self.config.verbose:
            self.logger.info(f"Training bundle (model + preprocessing state) saved to: {bundle_path}")
    
    def load_model(self, model_name: str = 'habitat_model') -> Any:
        """
        Load model from bundle file.
        
        Args:
            model_name: Base name of saved files
            
        Returns:
            Clustering model
        """
        bundle_path = os.path.join(self.out_dir, f'{model_name}_bundle.pkl')
        with open(bundle_path, 'rb') as f:
            bundle = pickle.load(f)
            
        # Load preprocessing state into instance
        self.preprocessing_state = bundle['preprocessing_state']
        return bundle['clustering_model']
