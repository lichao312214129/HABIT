"""
Combine supervoxels step for habitat analysis pipeline.

This group-level step combines supervoxel features from all subjects into
a single DataFrame for population-level processing.
"""

from typing import Dict, Any, Optional
import pandas as pd
import logging

from ..base_pipeline import GroupLevelStep


class CombineSupervoxelsStep(GroupLevelStep):
    """
    Combine supervoxel features from all subjects into a single DataFrame.
    
    This is a group-level step that runs after individual-level SupervoxelAggregationStep.
    It simply concatenates all subjects' supervoxel DataFrames.
    
    Stateless: simple concatenation operation.
    
    Attributes:
        fitted_: bool indicating whether the step has been fitted
    """
    
    def __init__(self):
        """Initialize combine supervoxels step."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: Dict[str, Dict], y: Optional[Any] = None, **fit_params) -> 'CombineSupervoxelsStep':
        """
        Fit the step (stateless operation, just mark as fitted).
        
        Args:
            X: Dict of subject_id -> {
                'supervoxel_df': pd.DataFrame
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
        Combine all subjects' supervoxel features into a single DataFrame.
        
        Args:
            X: Dict of subject_id -> {
                'supervoxel_df': pd.DataFrame  # Supervoxel features for this subject
            }
            
        Returns:
            Combined DataFrame with supervoxel-level features for all subjects
        """
        all_supervoxel_dfs = []
        
        for subject_id, data in X.items():
            try:
                supervoxel_df = data['supervoxel_df']
                all_supervoxel_dfs.append(supervoxel_df)
            except KeyError as e:
                self.logger.error(
                    f"Subject {subject_id} missing 'supervoxel_df' key. "
                    "Make sure SupervoxelAggregationStep was executed."
                )
                raise
        
        if not all_supervoxel_dfs:
            raise ValueError("No supervoxel features to combine")
        
        # Concatenate all subjects' supervoxel DataFrames
        combined_df = pd.concat(all_supervoxel_dfs, ignore_index=True)
        
        self.logger.info(f"Combined supervoxel features from {len(all_supervoxel_dfs)} subjects")
        self.logger.info(f"Combined DataFrame shape: {combined_df.shape}")
        self.logger.info(f"Combined DataFrame columns: {list(combined_df.columns)[:10]}...")
        self.logger.info(f"Combined DataFrame dtypes sample: {combined_df.dtypes.value_counts().to_dict()}")
        
        return combined_df
