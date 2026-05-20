"""
Calculate mean voxel features step for habitat analysis pipeline.

This step calculates the mean of voxel features within each supervoxel.
It's always executed in two-step strategy to provide baseline features.
"""

from typing import Any
import logging
from habit.utils.log_utils import get_module_logger

import numpy as np

from ..base_pipeline import IndividualLevelStep
from ..habitat_subject_data import HabitatSubjectData
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
        self.logger = get_module_logger(__name__)

    def transform_one(self, subject_id: str, subject_data: HabitatSubjectData) -> HabitatSubjectData:
        """Compute per-supervoxel mean features for one subject."""
        feature_df = subject_data.require_features(self.__class__.__name__)
        raw_df = subject_data.require_raw(self.__class__.__name__)
        mask_info = subject_data.require_mask_info(self.__class__.__name__)
        supervoxel_labels = subject_data.require_supervoxel_labels(self.__class__.__name__)
        n_clusters = len(np.unique(supervoxel_labels))

        mean_features_df = calculate_supervoxel_means(
            subject_id,
            feature_df,
            raw_df,
            supervoxel_labels,
            n_clusters,
        )

        return HabitatSubjectData(
            features=feature_df,
            raw=raw_df,
            mask_info=mask_info,
            supervoxel_labels=supervoxel_labels,
            mean_voxel_features=mean_features_df,
        )
