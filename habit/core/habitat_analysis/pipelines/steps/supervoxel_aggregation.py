"""
Supervoxel aggregation step for habitat analysis pipeline.

This step aggregates voxel features to supervoxel level and optionally merges
with advanced features from SupervoxelFeatureExtractionStep.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from ..base_pipeline import BasePipelineStep
from ...managers.feature_manager import FeatureManager
from ...config_schemas import HabitatAnalysisConfig, ResultColumns


class SupervoxelAggregationStep(BasePipelineStep):
    """
    Aggregate voxel features to supervoxel level.
    
    For two-step strategy:
    1. Calculate mean voxel features per supervoxel (always done)
    2. Optionally merge with advanced features from Step 4 (if Step 4 was executed)
    
    **Important**: This step always calculates mean features. If Step 4 was executed,
    it merges the advanced features from Step 4's output.
    
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
    
    def transform(self, X: Dict[str, Dict]) -> pd.DataFrame:
        """
        Aggregate features to supervoxel level.
        
        Args:
            X: Dict of subject_id -> {
                'features': pd.DataFrame,        # Processed voxel features
                'raw': pd.DataFrame,             # Raw voxel features
                'mask_info': dict,               # Mask info
                'supervoxel_labels': np.ndarray, # Supervoxel labels (1-indexed)
                'supervoxel_features': pd.DataFrame (optional)  # Advanced features from Step 4
            }
            Note: If Step 4 was executed, 'supervoxel_features' will be present in the dict.
            If Step 4 was skipped, this key will not exist.
            
        Returns:
            Combined DataFrame with supervoxel-level features for all subjects
            Columns include: Subject, Supervoxel, feature columns, and optionally
            advanced feature columns from Step 4.
        """
        all_supervoxel_features = []
        
        for subject_id, data in X.items():
            feature_df = data['features']
            raw_df = data['raw']
            supervoxel_labels = data['supervoxel_labels']
            
            # Get number of clusters from unique labels
            unique_labels = np.unique(supervoxel_labels)
            n_clusters = len(unique_labels)
            
            # Always calculate mean voxel features per supervoxel
            # This uses calculate_supervoxel_means() which aggregates voxel features
            mean_features_df = self.feature_manager.calculate_supervoxel_means(
                subject_id, 
                feature_df, 
                raw_df, 
                supervoxel_labels, 
                n_clusters
            )
            
            # If Step 4 was executed, merge advanced features
            # Check if 'supervoxel_features' exists in the data dict
            if 'supervoxel_features' in data:
                advanced_features_df = data['supervoxel_features'].copy()
                
                # Ensure advanced_features_df has Subject and Supervoxel columns for merging
                # The extract_supervoxel_features() may return DataFrame with 'SupervoxelID' instead of 'Supervoxel'
                # We need to standardize the column names
                if 'SupervoxelID' in advanced_features_df.columns and ResultColumns.SUPERVOXEL not in advanced_features_df.columns:
                    advanced_features_df[ResultColumns.SUPERVOXEL] = advanced_features_df['SupervoxelID']
                
                # Add Subject column if not present
                if ResultColumns.SUBJECT not in advanced_features_df.columns:
                    advanced_features_df[ResultColumns.SUBJECT] = subject_id
                
                # Merge mean features with advanced features
                # Use subject and supervoxel columns as keys
                merge_keys = [ResultColumns.SUBJECT, ResultColumns.SUPERVOXEL]
                
                # Ensure both DataFrames have the merge keys
                if all(key in mean_features_df.columns for key in merge_keys):
                    if all(key in advanced_features_df.columns for key in merge_keys):
                        # Drop 'SupervoxelID' if it exists to avoid duplicate columns
                        if 'SupervoxelID' in advanced_features_df.columns:
                            advanced_features_df = advanced_features_df.drop(columns=['SupervoxelID'])
                        
                        mean_features_df = mean_features_df.merge(
                            advanced_features_df,
                            on=merge_keys,
                            how='left',
                            suffixes=('', '_advanced')
                        )
                    else:
                        # If advanced features don't have merge keys, try to match by index
                        # This is a fallback - ideally both should have Subject and Supervoxel columns
                        if self.config.verbose:
                            self.feature_manager.logger.warning(
                                f"Advanced features for {subject_id} don't have merge keys "
                                f"({merge_keys}), attempting index-based merge"
                            )
                        # Try to align by index if possible
                        if len(mean_features_df) == len(advanced_features_df):
                            # Reset index and merge
                            mean_features_df = pd.concat(
                                [mean_features_df.reset_index(drop=True), 
                                 advanced_features_df.reset_index(drop=True)], 
                                axis=1
                            )
            
            all_supervoxel_features.append(mean_features_df)
        
        # Combine all subjects' supervoxel features
        if not all_supervoxel_features:
            raise ValueError("No supervoxel features to aggregate")
        
        combined_df = pd.concat(all_supervoxel_features, ignore_index=True)
        
        return combined_df
