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
Voxel-level feature extraction step for habitat analysis pipeline.

This step extracts voxel-level features from images for each subject.
"""

from typing import Any, Optional
import logging
from habit.utils.log_utils import get_module_logger

from ..base_pipeline import IndividualLevelStep
from ..habitat_subject_data import HabitatSubjectData
from ...services.feature_service import FeatureService


class VoxelFeatureExtractor(IndividualLevelStep):
    """
    Extract voxel-level features from images for one subject at a time.

    Stateless: feature extraction logic is fully driven by configuration;
    no cross-subject state is learnt.
    """

    def __init__(self, feature_service: FeatureService, habitat_image_writer: Optional[Any] = None):
        """
        Args:
            feature_service: FeatureService instance.
            habitat_image_writer: Kept only as an injected collaborator for recipes
                that still construct this step with a writer. The step does not
                mutate writer state.
        """
        super().__init__()
        self.feature_service = feature_service
        self.habitat_image_writer = habitat_image_writer
        self.logger = get_module_logger(__name__)

    def transform_one(self, subject_id: str, subject_data: HabitatSubjectData) -> HabitatSubjectData:
        """
        Extract voxel features for a single subject.

        ``subject_data`` is intentionally unused because this is the first
        individual-level step; image and mask paths live on ``feature_service``.
        """
        _, feature_df, raw_df, mask_info = self.feature_service.extract_voxel_features(subject_id)
        return HabitatSubjectData(
            features=feature_df,
            raw=raw_df,
            mask_info=mask_info,
        )
