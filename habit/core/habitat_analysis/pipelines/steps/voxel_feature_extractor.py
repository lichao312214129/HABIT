"""
Voxel-level feature extraction step for habitat analysis pipeline.

This step extracts voxel-level features from images for each subject.
"""

from typing import Dict, Any, Optional
import pandas as pd

from ..base_pipeline import BasePipelineStep
from ...managers.feature_manager import FeatureManager


class VoxelFeatureExtractor(BasePipelineStep):
    """
    Extract voxel-level features from images.
    
    Stateless: feature extraction logic is fixed, based on configuration.
    
    Attributes:
        feature_manager: FeatureManager instance for feature extraction
        fitted_: bool indicating whether the step has been fitted (always True after fit)
    """
    
    def __init__(self, feature_manager: FeatureManager, result_manager: Optional[Any] = None):
        """
        Initialize voxel feature extractor.
        
        Args:
            feature_manager: FeatureManager instance
            result_manager: ResultManager instance (optional)
        """
        super().__init__()
        self.feature_manager = feature_manager
        self.result_manager = result_manager
    
    def fit(self, X: Dict[str, Any], y: Optional[Any] = None, **fit_params) -> 'VoxelFeatureExtractor':
        """
        Fit the step (stateless operation, just mark as fitted).
        
        Args:
            X: Dict of subject_id -> data (not used in fit, but required by interface)
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
        Extract voxel features for each subject.
        
        Args:
            X: Dict of subject_id -> {
                'images': Dict of image_name -> image_path (optional, if not set in feature_manager),
                'masks': Dict of mask_name -> mask_path (optional, if not set in feature_manager)
            }
            Note: If images_paths and mask_paths are already set in feature_manager,
            this dict can be empty or just contain subject IDs.
            
        Returns:
            Dict of subject_id -> {
                'features': pd.DataFrame,  # Processed voxel features
                'raw': pd.DataFrame,        # Raw voxel features
                'mask_info': dict           # Mask metadata for image reconstruction
            }
        """
        results = {}
        
        for subject_id, data in X.items():
            # Extract voxel-level features
            # Returns: (subject_id, feature_df, raw_df, mask_info)
            _, feature_df, raw_df, mask_info = self.feature_manager.extract_voxel_features(
                subject_id
            )
            
            # Store mask_info in result_manager if available for later image reconstruction
            if self.result_manager is not None:
                if not hasattr(self.result_manager, 'mask_info_cache'):
                    self.result_manager.mask_info_cache = {}
                self.result_manager.mask_info_cache[subject_id] = mask_info
            
            results[subject_id] = {
                'features': feature_df,
                'raw': raw_df,
                'mask_info': mask_info
            }
        
        return results
