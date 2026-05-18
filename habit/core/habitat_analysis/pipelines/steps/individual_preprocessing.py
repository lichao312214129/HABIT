"""
Individual-level preprocessing step for habitat analysis pipeline.

This step applies preprocessing at the per-subject level (stateless).
"""

from habit.utils.log_utils import get_module_logger

from ..base_pipeline import IndividualLevelStep
from ..subject_state import SubjectHabitatState
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

    def transform_one(self, subject_id: str, subject_data: SubjectHabitatState) -> SubjectHabitatState:
        """Apply individual-level preprocessing to one subject's voxel features."""
        feature_df = subject_data.require_features(self.__class__.__name__)
        processed = self.feature_service.apply_preprocessing(feature_df, level='subject')
        processed = self.feature_service.clean_features(processed)

        return SubjectHabitatState(
            features=processed,
            raw=subject_data.require_raw(self.__class__.__name__),
            mask_info=subject_data.require_mask_info(self.__class__.__name__),
        )
