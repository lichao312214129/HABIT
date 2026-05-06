"""
Supervoxel feature extraction step for habitat analysis pipeline.

This step extracts advanced features (texture, shape, radiomics) from supervoxel maps.
Conditionally executed based on configuration.
"""

from typing import Any
import logging

from ..base_pipeline import IndividualLevelStep
from ..subject_state import SubjectHabitatState
from ...services.feature_service import FeatureService
from ...config_schemas import HabitatAnalysisConfig


class SupervoxelFeatureExtractionStep(IndividualLevelStep):
    """
    Extract advanced supervoxel-level features (texture / shape / radiomics)
    for one subject at a time using the supervoxel label NRRD that
    :class:`IndividualClusteringStep` wrote to disk.

    This step is only inserted into the pipeline when the configuration asks
    for advanced supervoxel features (i.e. not ``mean_voxel_features()``).

    Attributes:
        feature_service: FeatureService instance.
        config: Configuration object.
    """

    def __init__(
        self,
        feature_service: FeatureService,
        config: HabitatAnalysisConfig,
    ):
        super().__init__()
        self.feature_service = feature_service
        self.config = config
        self.logger = logging.getLogger(__name__)

    def transform_one(self, subject_id: str, subject_data: SubjectHabitatState) -> SubjectHabitatState:
        """
        Extract supervoxel features for one subject.

        We re-resolve the supervoxel file map per subject because:
          1. Supervoxel files are produced by the previous step's
             ``transform_one`` (so the mapping is stale at fit-time).
          2. Under multi-processing each worker has its own FeatureService
             copy, so the mapping must be rebuilt inside the worker.
        ``setup_supervoxel_files`` is a glob + dict build; per-subject cost
        is negligible compared to the actual feature extraction.
        """
        self.feature_service.setup_supervoxel_files(
            subjects=[subject_id],
            failed_subjects=[],
            out_folder=self.config.out_dir,
        )

        _, result = self.feature_service.extract_supervoxel_features(subject_id)
        if isinstance(result, Exception):
            raise result

        return SubjectHabitatState(
            features=subject_data.require_features(self.__class__.__name__),
            raw=subject_data.require_raw(self.__class__.__name__),
            mask_info=subject_data.require_mask_info(self.__class__.__name__),
            supervoxel_labels=subject_data.require_supervoxel_labels(self.__class__.__name__),
            supervoxel_features=result,
        )
