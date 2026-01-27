"""
Calculate mean voxel features step for habitat analysis pipeline.

This step calculates the mean of voxel features within each supervoxel.
It's always executed in two-step strategy to provide baseline features.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

from ..base_pipeline import IndividualLevelStep
from ...managers.feature_manager import FeatureManager
from ...config_schemas import HabitatAnalysisConfig


class CalculateMeanVoxelFeaturesStep(IndividualLevelStep):
    """
    Calculate mean of voxel features for each supervoxel.
    
    This step aggregates voxel-level features to supervoxel-level by computing
    the mean value of each feature within each supervoxel. This provides a
    simple but effective representation of supervoxel characteristics.
    
    **Purpose**:
    - Reduce dimensionality from voxel-level to supervoxel-level
    - Capture average tissue characteristics within each supervoxel
    - Provide baseline features for habitat clustering
    
    **When to use**:
    - Always included in two-step strategy
    - Can be used alone or combined with advanced supervoxel features
    
    **Individual-level step**: Processes each subject independently in parallel.
    
    Attributes:
        feature_manager: FeatureManager instance
        config: Configuration object
        fitted_: bool indicating whether the step has been fitted
    """
    
    def __init__(
        self,
        feature_manager: FeatureManager,
        config: HabitatAnalysisConfig
    ):
        """
        Initialize calculate mean voxel features step.
        
        Args:
            feature_manager: FeatureManager instance
            config: Configuration object
        """
        super().__init__()
        self.feature_manager = feature_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: Dict[str, Dict], y: Optional[Any] = None, **fit_params) -> 'CalculateMeanVoxelFeaturesStep':
        """
        Fit the step (stateless operation, just mark as fitted).
        
        Args:
            X: Dict of subject_id -> {
                'features': pd.DataFrame,
                'raw': pd.DataFrame,
                'mask_info': dict,
                'supervoxel_labels': np.ndarray
            }
            y: Optional target data (not used)
            **fit_params: Additional fitting parameters (not used)
            
        Returns:
            self
        """
        # Stateless step - no parameters to learn
        self.fitted_ = True
        return self
    
    def transform(self, X: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Calculate mean voxel features for each supervoxel in each subject.
        
        Args:
            X: Dict of subject_id -> {
                'features': pd.DataFrame,
                'raw': pd.DataFrame,
                'mask_info': dict,
                'supervoxel_labels': np.ndarray
            }
            
        Returns:
            Dict of subject_id -> {
                'features': pd.DataFrame (original voxel features),
                'raw': pd.DataFrame (original raw features),
                'mask_info': dict,
                'supervoxel_labels': np.ndarray,
                'mean_voxel_features': pd.DataFrame  # NEW: mean features
            }
        """
        results = {}
        
        for subject_id, data in X.items():
            try:
                feature_df = data['features']
                raw_df = data['raw']
                supervoxel_labels = data['supervoxel_labels']
                
                # Get number of clusters
                unique_labels = np.unique(supervoxel_labels)
                n_clusters = len(unique_labels)
                
                # Calculate mean voxel features per supervoxel
                mean_features_df = self.feature_manager.calculate_supervoxel_means(
                    subject_id, 
                    feature_df, 
                    raw_df, 
                    supervoxel_labels, 
                    n_clusters
                )
                
                # Store result with NEW key name
                results[subject_id] = {
                    'features': data['features'],
                    'raw': data['raw'],
                    'mask_info': data['mask_info'],
                    'supervoxel_labels': supervoxel_labels,
                    'mean_voxel_features': mean_features_df  # NEW key
                }
                
            except Exception as e:
                self.logger.error(f"Error calculating mean voxel features for subject {subject_id}: {e}")
                raise
        
        return results
