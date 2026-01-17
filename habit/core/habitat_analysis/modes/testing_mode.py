"""
Testing mode implementation.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

from .base_mode import BaseMode
from ..utils.preprocessing_state import PreprocessingState

class TestingMode(BaseMode):
    """
    Mode for testing: loads pre-trained models and applies them.
    """
    
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.preprocessing_state = None  # Will be loaded from bundle
    
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
        # Load pre-trained model (and preprocessing state)
        model = self.load_model('supervoxel2habitat_clustering_strategy')
        
        if self.config.runtime.verbose:
            bundle_path = os.path.join(
                self.out_dir, 'supervoxel2habitat_clustering_strategy_bundle.pkl'
            )
            self.logger.info(
                f"Performing clustering using pre-trained model: {bundle_path}"
            )
        
        # Apply model
        habitat_labels = model.predict(features) + 1  # Start from 1
        optimal_n_clusters = model.n_clusters
        
        return habitat_labels, optimal_n_clusters, None
    
    def process_features(
        self,
        features: pd.DataFrame,
        methods: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Process features using loaded group-level statistics (Testing: transform only).
        
        Args:
            features: DataFrame to process
            methods: List of preprocessing method configurations (not used, loaded from state)
            
        Returns:
            Processed DataFrame
        """
        # Auto-load preprocessing state if not already loaded
        if self.preprocessing_state is None:
            if self.config.runtime.verbose:
                self.logger.info(
                    "Preprocessing state not loaded yet, loading from training bundle..."
                )
            # Load the bundle to get preprocessing state
            self._load_preprocessing_state()
            
        if self.config.runtime.verbose:
            self.logger.info("Applying group-level preprocessing from training state...")
            
        return self.preprocessing_state.transform(features)
    
    def _load_preprocessing_state(self) -> None:
        """
        Load only the preprocessing state from the bundle.
        This is called automatically by process_features if needed.
        """
        model_name = 'supervoxel2habitat_clustering_strategy'
        bundle_path = os.path.join(self.out_dir, f'{model_name}_bundle.pkl')
        
        if not os.path.exists(bundle_path):
            raise ValueError(
                f"No training bundle found at {bundle_path}. "
                "Please ensure you are using the correct output directory from training."
            )
        
        with open(bundle_path, 'rb') as f:
            bundle = pickle.load(f)
        
        # Load only preprocessing state
        self.preprocessing_state = bundle['preprocessing_state']
        
        if self.config.runtime.verbose:
            self.logger.info(f"Loaded preprocessing state from: {bundle_path}")
    
    def save_model(self, model: Any, model_name: str = 'habitat_model') -> None:
        """Testing mode doesn't save models - no-op."""
        pass
    
    def load_model(self, model_name: str = 'habitat_model') -> Any:
        """
        Load pre-trained model bundle (model + preprocessing state).
        
        Args:
            model_name: Base name of saved files
            
        Returns:
            Clustering model
        """
        bundle_path = os.path.join(self.out_dir, f'{model_name}_bundle.pkl')
        
        if not os.path.exists(bundle_path):
            raise ValueError(f"No training bundle found at {bundle_path}")
        
        with open(bundle_path, 'rb') as f:
            bundle = pickle.load(f)
        
        # Load preprocessing state into instance
        self.preprocessing_state = bundle['preprocessing_state']
        
        if self.config.runtime.verbose:
            self.logger.info(f"Loaded training bundle from: {bundle_path}")
        
        return bundle['clustering_model']
