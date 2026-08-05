# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
