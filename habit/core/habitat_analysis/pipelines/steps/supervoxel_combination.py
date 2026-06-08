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
Combine supervoxels step for habitat analysis pipeline.

This group-level step combines supervoxel features from all subjects into
a single DataFrame for group-level processing.
"""

from typing import Dict, Any, Optional
import pandas as pd
import logging
from habit.utils.log_utils import get_module_logger

from ..base_pipeline import GroupLevelStep
from ..habitat_subject_data import HabitatSubjectData


class CombineSupervoxelsStep(GroupLevelStep):
    """
    Combine supervoxel features from all subjects into a single DataFrame.

    This is a group-level step that runs after the individual-level
    MergeSupervoxelFeaturesStep. It simply concatenates all subjects'
    supervoxel DataFrames.
    
    Stateless: simple concatenation operation.
    
    Attributes:
        fitted_: bool indicating whether the step has been fitted
    """
    
    def __init__(self):
        """Initialize combine supervoxels step."""
        super().__init__()
        self.logger = get_module_logger(__name__)
    
    def fit(self, X: Dict[str, HabitatSubjectData], y: Optional[Any] = None, **fit_params) -> 'CombineSupervoxelsStep':
        """Fit the stateless step by marking it as fitted."""
        self.fitted_ = True
        return self
    
    def transform(self, X: Dict[str, HabitatSubjectData]) -> pd.DataFrame:
        """Combine all subjects' supervoxel features into a single DataFrame."""
        all_supervoxel_dfs = []
        
        for subject_id, data in X.items():
            try:
                supervoxel_df = data.require_supervoxel_df(self.__class__.__name__)
                all_supervoxel_dfs.append(supervoxel_df)
            except ValueError:
                self.logger.error(
                    f"Subject {subject_id} missing supervoxel_df. "
                    "Make sure MergeSupervoxelFeaturesStep was executed."
                )
                raise
        
        if not all_supervoxel_dfs:
            raise ValueError("No supervoxel features to combine")
        
        combined_df = pd.concat(all_supervoxel_dfs, ignore_index=True)

        self.logger.info(
            f"Combined supervoxel features from {len(all_supervoxel_dfs)} "
            f"subjects (shape={combined_df.shape})"
        )
        self.logger.debug(f"Combined columns sample: {list(combined_df.columns)[:10]}")
        self.logger.debug(f"Combined dtypes: {combined_df.dtypes.value_counts().to_dict()}")

        return combined_df
