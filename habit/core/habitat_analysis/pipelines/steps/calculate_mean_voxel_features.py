"""
Calculate mean voxel features step for habitat analysis pipeline.

This step calculates the mean of voxel features within each supervoxel.
It's always executed in two-step strategy to provide baseline features.
"""

from typing import Any, Dict
import logging

import numpy as np

from ..base_pipeline import IndividualLevelStep
from ...clustering_features import calculate_supervoxel_means
from ...config_schemas import HabitatAnalysisConfig


class CalculateMeanVoxelFeaturesStep(IndividualLevelStep):
    """
    Aggregate per-voxel features to per-supervoxel features by taking the
    arithmetic mean of each feature within each supervoxel.

    Always inserted in two-step / one-step pipelines as a baseline supervoxel
    feature (advanced features are merged on top by ``MergeSupervoxelFeaturesStep``).

    Attributes:
        config: Configuration object.
    """

    def __init__(self, config: HabitatAnalysisConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)

    def transform_one(self, subject_id: str, subject_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute per-supervoxel mean features for one subject.

        Returns the same dict shape as input plus ``mean_voxel_features``
        (a DataFrame with one row per supervoxel).
        """
        feature_df = subject_data['features']
        raw_df = subject_data['raw']
        supervoxel_labels = subject_data['supervoxel_labels']
        n_clusters = len(np.unique(supervoxel_labels))

        mean_features_df = calculate_supervoxel_means(
            subject_id,
            feature_df,
            raw_df,
            supervoxel_labels,
            n_clusters,
        )

        return {
            'features': feature_df,
            'raw': raw_df,
            'mask_info': subject_data['mask_info'],
            'supervoxel_labels': supervoxel_labels,
            'mean_voxel_features': mean_features_df,
        }
