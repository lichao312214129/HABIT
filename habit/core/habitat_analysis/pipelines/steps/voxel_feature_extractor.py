"""
Voxel-level feature extraction step for habitat analysis pipeline.

This step extracts voxel-level features from images for each subject.
"""

from typing import Any, Optional
import logging

from ..base_pipeline import IndividualLevelStep
from ..subject_state import SubjectHabitatState
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
        self.logger = logging.getLogger(__name__)

    def transform_one(self, subject_id: str, subject_data: SubjectHabitatState) -> SubjectHabitatState:
        """
        Extract voxel features for a single subject.

        ``subject_data`` is intentionally unused because this is the first
        individual-level step; image and mask paths live on ``feature_service``.
        """
        _, feature_df, raw_df, mask_info = self.feature_service.extract_voxel_features(subject_id)
        return SubjectHabitatState(
            features=feature_df,
            raw=raw_df,
            mask_info=mask_info,
        )
