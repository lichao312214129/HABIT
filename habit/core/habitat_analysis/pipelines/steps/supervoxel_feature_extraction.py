"""
Supervoxel feature extraction step for habitat analysis pipeline.

This step extracts advanced features (texture, shape, radiomics) from supervoxel maps.
Conditionally executed based on configuration.
"""

from typing import Dict, Any, Optional
import pandas as pd
import logging

from ..base_pipeline import BasePipelineStep
from ...managers.feature_manager import FeatureManager
from ...config_schemas import HabitatAnalysisConfig
from habit.utils.parallel_utils import parallel_map


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
        subject_ids = list(X.keys())
        
        # Get number of processes from config
        n_processes = getattr(self.config, 'processes', 1)
        
        # Extract supervoxel features in parallel
        successful_results, failed_subjects = parallel_map(
            func=self.feature_manager.extract_supervoxel_features,
            items=subject_ids,
            n_processes=n_processes,
            desc="Extracting supervoxel features",
            logger=self.logger,
            show_progress=True,
        )
        
        # Convert results to dict
        results = {}
        for proc_result in successful_results:
            # proc_result.item_id contains subject_id
            # proc_result.result contains features_df or Exception
            subject_id = proc_result.item_id
            features_df = proc_result.result
            
            if isinstance(features_df, Exception):
                self.logger.error(
                    f"Failed to extract supervoxel features for {subject_id}: {features_df}"
                )
                continue
            
            # Add supervoxel features to result
            results[subject_id] = {
                'features': X[subject_id]['features'],
                'raw': X[subject_id]['raw'],
                'mask_info': X[subject_id]['mask_info'],
                'supervoxel_labels': X[subject_id]['supervoxel_labels'],
                'supervoxel_features': features_df
            }
        
        # Log failed subjects
        if failed_subjects:
            self.logger.error(
                f"Failed to extract supervoxel features for {len(failed_subjects)} subject(s): "
                f"{', '.join(str(s) for s in failed_subjects)}"
            )
        
        return results
