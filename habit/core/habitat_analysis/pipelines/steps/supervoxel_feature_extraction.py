"""
Supervoxel feature extraction step for habitat analysis pipeline.

This step extracts advanced features (texture, shape, radiomics) from supervoxel maps.
Conditionally executed based on configuration.
"""

from typing import Dict, Any, Optional
import pandas as pd
import logging

from ..base_pipeline import IndividualLevelStep
from ...managers.feature_manager import FeatureManager
from ...config_schemas import HabitatAnalysisConfig


class SupervoxelFeatureExtractionStep(IndividualLevelStep):
    """
    Extract advanced features for each supervoxel based on supervoxel maps.
    
    This step extracts advanced features (texture, shape, radiomics) from 
    supervoxel label maps. It runs after supervoxel clustering and requires
    supervoxel map files to be saved.
    
    **Important**: This step is conditionally included in the pipeline based on
    configuration. If only `mean_voxel_features()` is used, this step is skipped
    to save computation time.
    
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
        Initialize supervoxel feature extraction step.
        
        Args:
            feature_manager: FeatureManager instance
            config: Configuration object
        """
        super().__init__()
        self.feature_manager = feature_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: Dict[str, Dict], y: Optional[Any] = None, **fit_params) -> 'SupervoxelFeatureExtractionStep':
        """
        Fit step: setup supervoxel file discovery for feature extraction.
        
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
        # Setup supervoxel file dictionary for feature extraction
        # This discovers supervoxel map files saved in Step 3 (IndividualClusteringStep)
        subjects = list(X.keys())
        self.feature_manager.setup_supervoxel_files(
            subjects, 
            failed_subjects=[],
            out_folder=self.config.out_dir
        )
        
        self.fitted_ = True
        return self
    
    def transform(self, X: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
        """
        Extract supervoxel-level features for each subject with parallel processing.
        
        Args:
            X: Dict of subject_id -> {
                'features': pd.DataFrame,
                'raw': pd.DataFrame,
                'mask_info': dict,
                'supervoxel_labels': np.ndarray
            }
            
        Returns:
            Dict of subject_id -> {
                'features': pd.DataFrame,
                'raw': pd.DataFrame,
                'mask_info': dict,
                'supervoxel_labels': np.ndarray,
                'supervoxel_features': pd.DataFrame
            }
        """
        results = {}
        
        # Process each subject sequentially (pipeline handles parallelization)
        for subject_id, data in X.items():
            try:
                mask_info = data['mask_info']
                supervoxel_labels = data['supervoxel_labels']
                
                # Extract advanced features from supervoxel maps
                supervoxel_features_df = self.feature_manager.extract_supervoxel_features(
                    subject_id,
                    supervoxel_labels,
                    mask_info
                )
                
                # Add supervoxel features to the data
                results[subject_id] = {
                    'features': data['features'],
                    'raw': data['raw'],
                    'mask_info': mask_info,
                    'supervoxel_labels': supervoxel_labels,
                    'supervoxel_features': supervoxel_features_df
                }
                
            except Exception as e:
                self.logger.error(f"Failed to extract supervoxel features for subject {subject_id}: {e}")
                raise
        
        return results
