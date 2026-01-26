"""
Supervoxel feature extraction step for habitat analysis pipeline.

This step extracts advanced features (texture, shape, radiomics) from supervoxel maps.
Conditionally executed based on configuration.
"""

from typing import Dict, Any, Optional
import pandas as pd

from ..base_pipeline import BasePipelineStep
from ...managers.feature_manager import FeatureManager
from ...config_schemas import HabitatAnalysisConfig


class SupervoxelFeatureExtractionStep(BasePipelineStep):
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
        Extract supervoxel-level features for each subject.
        
        Args:
            X: Dict of subject_id -> {
                'features': pd.DataFrame,
                'raw': pd.DataFrame,
                'mask_info': dict,
                'supervoxel_labels': np.ndarray
            }
            
        Returns:
            Dict of subject_id -> {
                'features': pd.DataFrame,        # Original voxel features (unchanged)
                'raw': pd.DataFrame,             # Original raw features (unchanged)
                'mask_info': dict,               # Original mask info (unchanged)
                'supervoxel_labels': np.ndarray, # Original labels (unchanged)
                'supervoxel_features': pd.DataFrame  # NEW: Advanced supervoxel features
            }
            The supervoxel_features DataFrame contains advanced features (texture, shape, radiomics) 
            for each supervoxel in that subject.
        """
        results = {}
        
        for subject_id, data in X.items():
            # Extract advanced features from supervoxel maps
            # This uses the supervoxel map file saved in Step 3
            # Uses extract_supervoxel_features() method
            # Returns: (subject_id, features_df or Exception)
            result = self.feature_manager.extract_supervoxel_features(subject_id)
            
            if isinstance(result, tuple):
                subj_id, features_df = result
                if isinstance(features_df, Exception):
                    raise ValueError(
                        f"Failed to extract supervoxel features for subject {subject_id}: {features_df}"
                    )
                supervoxel_features_df = features_df
            else:
                # Handle case where method returns directly
                supervoxel_features_df = result
            
            # Add supervoxel_features to the data dict for next step
            results[subject_id] = {
                'features': data['features'],
                'raw': data['raw'],
                'mask_info': data['mask_info'],
                'supervoxel_labels': data['supervoxel_labels'],
                'supervoxel_features': supervoxel_features_df  # Add advanced features
            }
        
        return results
