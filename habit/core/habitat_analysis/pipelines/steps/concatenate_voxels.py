"""
Concatenate voxels step for direct pooling strategy.

This step concatenates all voxels from all subjects into a single DataFrame.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from ..base_pipeline import BasePipelineStep
from ...config_schemas import ResultColumns


class ConcatenateVoxelsStep(BasePipelineStep):
    """
    Concatenate all voxels from all subjects into a single DataFrame.
    
    Used in direct pooling strategy where all voxels are pooled together
    before population-level clustering.
    
    Stateless: simple concatenation operation.
    
    Attributes:
        fitted_: bool indicating whether the step has been fitted
    """
    
    def __init__(self):
        """Initialize concatenate voxels step."""
        super().__init__()
    
    def fit(self, X: Dict[str, Dict], y: Optional[Any] = None, **fit_params) -> 'ConcatenateVoxelsStep':
        """
        Fit the step (stateless operation, just mark as fitted).
        
        Args:
            X: Dict of subject_id -> {
                'features': pd.DataFrame,
                'raw': pd.DataFrame,
                'mask_info': dict
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
        Concatenate all voxels from all subjects.
        
        Args:
            X: Dict of subject_id -> {
                'features': pd.DataFrame,  # Voxel features for this subject
                'raw': pd.DataFrame,       # Raw features (not used)
                'mask_info': dict          # Mask info (not used)
            }
            
        Returns:
            Combined DataFrame with all voxels from all subjects
            Columns include: Subject (added), and all feature columns
        """
        all_voxels = []
        
        for subject_id, data in X.items():
            feature_df = data['features'].copy()
            
            # Add subject ID column
            feature_df[ResultColumns.SUBJECT] = subject_id
            
            all_voxels.append(feature_df)
        
        if not all_voxels:
            raise ValueError("No voxel features to concatenate")
        
        # Concatenate all subjects' voxels
        combined_df = pd.concat(all_voxels, ignore_index=True)
        
        # Reorder columns: Subject first, then features
        cols = [ResultColumns.SUBJECT] + [c for c in combined_df.columns if c != ResultColumns.SUBJECT]
        combined_df = combined_df[cols]
        
        return combined_df
