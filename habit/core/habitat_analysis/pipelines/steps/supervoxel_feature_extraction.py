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
        Fit step: record subjects for later supervoxel file discovery.
        
        Note: Supervoxel files are discovered in transform() rather than fit()
        because the files are created by the previous step's transform(), which
        hasn't run yet during the fit() phase.
        
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
        # Store subjects for later supervoxel file discovery in transform()
        # Note: We cannot call setup_supervoxel_files() here because supervoxel
        # files haven't been saved yet (they're saved during IndividualClusteringStep.transform())
        self.subjects_ = list(X.keys())
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
        # Setup supervoxel files dictionary NOW (not in fit())
        # This must be done here because supervoxel files are created by the
        # previous step's transform(), which runs AFTER all fit() calls
        # 
        # IMPORTANT: In parallel processing, each worker process has its own
        # copy of FeatureManager, so each worker needs to setup its own mapping.
        # We always setup for the current subjects being processed.
        
        # Get subjects from current input data (works in both serial and parallel modes)
        current_subjects = list(X.keys())
        
        # Setup file mapping for current subjects
        # In parallel mode: each worker processes one subject at a time
        # In serial mode: may process multiple subjects at once
        self.feature_manager.setup_supervoxel_files(
            subjects=current_subjects,
            failed_subjects=[],
            out_folder=self.config.out_dir
        )
        
        results = {}
        
        # Process each subject sequentially (pipeline handles parallelization)
        for subject_id, data in X.items():
            try:
                mask_info = data['mask_info']
                supervoxel_labels = data['supervoxel_labels']
                
                # Extract advanced features from supervoxel maps (from saved files)
                # Note: extract_supervoxel_features reads from supervoxel map files
                # that were saved during the clustering step
                subject_id_returned, result = self.feature_manager.extract_supervoxel_features(
                    subject_id
                )
                
                # Handle potential errors returned from feature extraction
                if isinstance(result, Exception):
                    raise result
                
                supervoxel_features_df = result
                
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
