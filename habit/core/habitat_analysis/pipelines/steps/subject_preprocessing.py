"""
Subject-level preprocessing step for habitat analysis pipeline.

This step applies preprocessing at the individual subject level (stateless).
"""

from typing import Any, Dict
import logging

from ..base_pipeline import IndividualLevelStep
from ...services.feature_service import FeatureService


class SubjectPreprocessingStep(IndividualLevelStep):
    """
    Subject-level preprocessing (stateless).

    Each subject is preprocessed independently using its own statistics.
    No state needs to be saved between training and testing.

    Attributes:
        feature_service: FeatureService instance for preprocessing.
    """

    def __init__(self, feature_service: FeatureService):
        super().__init__()
        self.feature_service = feature_service
        self.logger = logging.getLogger(__name__)

    def transform_one(self, subject_id: str, subject_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply subject-level preprocessing to one subject's voxel features.

        Args:
            subject_id: Subject identifier.
            subject_data: Dict with ``features``, ``raw``, ``mask_info`` from
                :class:`VoxelFeatureExtractor`.

        Returns:
            Same dict shape as input, with ``features`` replaced by the
            preprocessed and cleaned DataFrame.
        """
        feature_df = subject_data['features']
        processed = self.feature_service.apply_preprocessing(feature_df, level='subject')
        processed = self.feature_service.clean_features(processed)

        return {
            'features': processed,
            'raw': subject_data['raw'],
            'mask_info': subject_data['mask_info'],
        }
