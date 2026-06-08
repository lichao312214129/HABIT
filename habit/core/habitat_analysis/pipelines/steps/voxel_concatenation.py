# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
"""
Concatenate voxels step for direct pooling strategy.

This step concatenates all voxels from all subjects into a single DataFrame.
"""

from typing import Dict, Any, Optional
import pandas as pd

from habit.utils.log_utils import get_module_logger

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
        self.logger = get_module_logger(__name__)
    
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

        from habit.core.habitat_analysis.feature_preprocessing.dataframe_utils import (
            split_metadata_and_features,
        )
        _, feature_df = split_metadata_and_features(combined_df)
        self.logger.info(
            f"Concatenated voxel features from {len(all_voxels)} subject(s) "
            f"(rows={len(combined_df)}, feature_columns={feature_df.shape[1]})"
        )

        return combined_df
