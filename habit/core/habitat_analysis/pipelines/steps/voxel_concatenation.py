"""
Concatenate voxels step for direct pooling strategy.

This step concatenates all voxels from all subjects into a single DataFrame.
"""

from typing import Dict, Any, Optional
import pandas as pd

from ..base_pipeline import GroupLevelStep
from ..habitat_subject_data import HabitatSubjectData
from ...config_schemas import ResultColumns


class ConcatenateVoxelsStep(GroupLevelStep):
    """
    Concatenate all voxels from all subjects into a single DataFrame.
    
    Used in direct pooling strategy where all voxels are pooled together
    before group-level clustering.
    
    Stateless: simple concatenation operation.
    
    Attributes:
        fitted_: bool indicating whether the step has been fitted
    """
    
    def __init__(self):
        """Initialize concatenate voxels step."""
        super().__init__()
    
    def fit(self, X: Dict[str, HabitatSubjectData], y: Optional[Any] = None, **fit_params) -> 'ConcatenateVoxelsStep':
        """Fit the stateless step by marking it as fitted."""
        self.fitted_ = True
        return self
    
    def transform(self, X: Dict[str, HabitatSubjectData]) -> pd.DataFrame:
        """Concatenate all voxel features from all subjects."""
        all_voxels = []
        
        for subject_id, data in X.items():
            feature_df = data.require_features(self.__class__.__name__).copy()
            feature_df[ResultColumns.SUBJECT] = subject_id
            all_voxels.append(feature_df)
        
        if not all_voxels:
            raise ValueError("No voxel features to concatenate")
        
        combined_df = pd.concat(all_voxels, ignore_index=True)
        cols = [ResultColumns.SUBJECT] + [c for c in combined_df.columns if c != ResultColumns.SUBJECT]
        combined_df = combined_df[cols]
        
        return combined_df
