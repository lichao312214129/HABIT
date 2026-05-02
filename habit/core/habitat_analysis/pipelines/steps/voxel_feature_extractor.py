"""
Voxel-level feature extraction step for habitat analysis pipeline.

This step extracts voxel-level features from images for each subject.
"""

from typing import Dict, Any, Optional
import pandas as pd
import logging

from ..base_pipeline import IndividualLevelStep
from ...services.feature_service import FeatureService


class VoxelFeatureExtractor(IndividualLevelStep):
    """
    Extract voxel-level features from images.
    
    Stateless: feature extraction logic is fixed, based on configuration.
    
    Attributes:
        feature_service: FeatureService instance for feature extraction
        fitted_: bool indicating whether the step has been fitted (always True after fit)
    """
    
    def __init__(self, feature_service: FeatureService, result_writer: Optional[Any] = None):
        """
        Initialize voxel feature extractor.
        
        Args:
            feature_service: FeatureService instance
            result_writer: ResultWriter instance (optional)
        """
        super().__init__()
        self.feature_service = feature_service
        self.result_writer = result_writer
        self.logger = logging.getLogger(__name__)
    
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
        Extract voxel features for each subject sequentially.
        
        Pipeline handles parallelization at subject level, so this method
        processes subjects sequentially without parallel logic.
        
        Args:
            X: Dict of subject_id -> {
                'images': Dict of image_name -> image_path (optional, if not set in feature_service),
                'masks': Dict of mask_name -> mask_path (optional, if not set in feature_service)
            }
            Note: If images_paths and mask_paths are already set in feature_service,
            this dict can be empty or just contain subject IDs.
            
        Returns:
            Dict of subject_id -> {
                'features': pd.DataFrame,  # Processed voxel features
                'raw': pd.DataFrame,        # Raw voxel features
                'mask_info': dict           # Mask metadata for image reconstruction
            }
        """
        results = {}
        
        # Process each subject sequentially (pipeline handles parallelization)
        for subject_id in X.keys():
            try:
                # Extract voxel features for this subject
                subject_id_result, feature_df, raw_df, mask_info = \
                    self.feature_service.extract_voxel_features(subject_id)
                
                # Store mask_info in result_writer if available
                if self.result_writer is not None:
                    if not hasattr(self.result_writer, 'mask_info_cache'):
                        self.result_writer.mask_info_cache = {}
                    self.result_writer.mask_info_cache[subject_id] = mask_info
                
                results[subject_id] = {
                    'features': feature_df,
                    'raw': raw_df,
                    'mask_info': mask_info
                }
                
            except Exception as e:
                self.logger.error(f"Failed to extract voxel features for subject {subject_id}: {e}")
                raise
        
        return results
