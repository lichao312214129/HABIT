"""
Voxel-level feature extraction step for habitat analysis pipeline.

This step extracts voxel-level features from images for each subject.
"""

from typing import Dict, Any, Optional
import pandas as pd
import logging

from ..base_pipeline import BasePipelineStep
from ...managers.feature_manager import FeatureManager
from habit.utils.parallel_utils import parallel_map


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
        Extract voxel features for each subject with parallel processing.
        
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
        subject_ids = list(X.keys())
        
        # Get number of processes from config
        n_processes = getattr(self.feature_manager.config, 'processes', 1)
        
        # Process subjects in parallel with progress bar
        successful_results, failed_subjects = parallel_map(
            func=self.feature_manager.extract_voxel_features,
            items=subject_ids,
            n_processes=n_processes,
            desc="Extracting voxel features",
            logger=self.logger,
            show_progress=True,
            log_file_path=self.feature_manager._log_file_path,
            log_level=self.feature_manager._log_level,
        )
        
        # Convert successful results to dict format
        results = {}
        for proc_result in successful_results:
            subject_id, feature_df, raw_df, mask_info = proc_result.result
            
            # Store mask_info in result_manager if available
            if self.result_manager is not None:
                if not hasattr(self.result_manager, 'mask_info_cache'):
                    self.result_manager.mask_info_cache = {}
                self.result_manager.mask_info_cache[subject_id] = mask_info
            
            results[subject_id] = {
                'features': feature_df,
                'raw': raw_df,
                'mask_info': mask_info
            }
        
        # Log failed subjects
        if failed_subjects:
            self.logger.error(
                f"Failed to extract voxel features for {len(failed_subjects)} subject(s): "
                f"{', '.join(str(s) for s in failed_subjects)}"
            )
        
        return results
