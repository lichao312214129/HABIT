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
Individual-level preprocessing step for habitat analysis pipeline.

This step applies preprocessing at the per-subject level (stateless).
"""

from habit.utils.log_utils import get_module_logger

from ..base_pipeline import IndividualLevelStep
from ..habitat_subject_data import HabitatSubjectData
from ...services.feature_service import FeatureService


class IndividualPreprocessingStep(IndividualLevelStep):
    """
    Individual-level preprocessing (stateless).

    Each subject is preprocessed independently using its own statistics.
    No state needs to be saved between training and testing.
    """

    def __init__(self, feature_service: FeatureService):
        super().__init__()
        self.feature_service = feature_service
        self.logger = get_module_logger(__name__)

    def transform_one(self, subject_id: str, subject_data: HabitatSubjectData) -> HabitatSubjectData:
        """Apply individual-level preprocessing to one subject's voxel features."""
        feature_df = subject_data.require_features(self.__class__.__name__)
        self.logger.info(
            "Applying subject-level preprocessing for subject '%s': "
            "rows=%d, columns=%d",
            subject_id,
            len(feature_df),
            feature_df.shape[1],
        )
        processed = self.feature_service.apply_preprocessing(feature_df, level='subject')
        processed = self.feature_service.clean_features(processed)

        return HabitatSubjectData(
            features=processed,
            raw=subject_data.require_raw(self.__class__.__name__),
            mask_info=subject_data.require_mask_info(self.__class__.__name__),
        )
