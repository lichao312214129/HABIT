"""
Supervoxel aggregation step for habitat analysis pipeline.

This step aggregates voxel features to supervoxel level and optionally merges
with advanced features from SupervoxelFeatureExtractionStep.
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging

from ..base_pipeline import IndividualLevelStep
from ...managers.feature_manager import FeatureManager
from ...config_schemas import HabitatAnalysisConfig, ResultColumns


class SupervoxelAggregationStep(IndividualLevelStep):
    """
    Aggregate voxel features to supervoxel level for each subject.
    
    For two-step strategy:
    1. Calculate mean voxel features per supervoxel (always done)
    2. Optionally merge with advanced features from Step 4 (if Step 4 was executed)
    
    **Important**: 
    - This is an individual-level step that processes each subject independently
    - Returns Dict[str, Dict] with each subject's supervoxel DataFrame
    - Use CombineSupervoxelsStep (group-level) to merge all subjects afterwards
    
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
        Initialize supervoxel aggregation step.
        
        Args:
            feature_manager: FeatureManager instance
            config: Configuration object
        """
        super().__init__()
        self.feature_manager = feature_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: Dict[str, Dict], y: Optional[Any] = None, **fit_params) -> 'SupervoxelAggregationStep':
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
        Aggregate features to supervoxel level for each subject.
        
        Pipeline handles parallelization at subject level, so this method
        processes subjects sequentially without parallel logic.
        
        **Important**: This step returns Dict (individual-level), not DataFrame.
        Use CombineSupervoxelsStep (group-level) to combine all subjects.
        
        Args:
            X: Dict of subject_id -> {
                'features': pd.DataFrame,
                'raw': pd.DataFrame,
                'mask_info': dict,
                'supervoxel_labels': np.ndarray,
                'supervoxel_features': pd.DataFrame (optional)
            }
            
        Returns:
            Dict of subject_id -> {
                'supervoxel_df': pd.DataFrame  # Supervoxel features for this subject
            }
        """
        results = {}
        
        # Process each subject sequentially (pipeline handles parallelization)
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
                
                # If Step 4 was executed, merge advanced features
                if 'supervoxel_features' in data:
                    advanced_features_df = data['supervoxel_features'].copy()
                    
                    # Standardize column names
                    if 'SupervoxelID' in advanced_features_df.columns and ResultColumns.SUPERVOXEL not in advanced_features_df.columns:
                        advanced_features_df[ResultColumns.SUPERVOXEL] = advanced_features_df['SupervoxelID']
                    
                    if ResultColumns.SUBJECT not in advanced_features_df.columns:
                        advanced_features_df[ResultColumns.SUBJECT] = subject_id
                    
                    # Merge
                    merge_keys = [ResultColumns.SUBJECT, ResultColumns.SUPERVOXEL]
                    if all(key in mean_features_df.columns for key in merge_keys):
                        if all(key in advanced_features_df.columns for key in merge_keys):
                            if 'SupervoxelID' in advanced_features_df.columns:
                                advanced_features_df = advanced_features_df.drop(columns=['SupervoxelID'])
                            
                            mean_features_df = mean_features_df.merge(
                                advanced_features_df,
                                on=merge_keys,
                                how='left',
                                suffixes=('', '_advanced')
                            )
                        else:
                            if self.config.verbose:
                                self.logger.warning(
                                    f"Advanced features for {subject_id} don't have merge keys, "
                                    "attempting index-based merge"
                                )
                            if len(mean_features_df) == len(advanced_features_df):
                                mean_features_df = pd.concat(
                                    [mean_features_df.reset_index(drop=True), 
                                     advanced_features_df.reset_index(drop=True)], 
                                    axis=1
                                )
                
                # Store result for this subject
                results[subject_id] = {
                    'supervoxel_df': mean_features_df
                }
                
            except Exception as e:
                self.logger.error(f"Error aggregating subject {subject_id}: {e}")
                raise
        
        return results
