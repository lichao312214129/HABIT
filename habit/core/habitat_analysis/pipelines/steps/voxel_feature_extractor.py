"""
Voxel-level feature extraction step for habitat analysis pipeline.

This step extracts voxel-level features from images for each subject.
"""

from typing import Any, Dict, Optional
import logging

from ..base_pipeline import IndividualLevelStep
from ...services.feature_service import FeatureService


class VoxelFeatureExtractor(IndividualLevelStep):
    """
    Extract voxel-level features from images for one subject at a time.

    Stateless: feature extraction logic is fully driven by configuration;
    no cross-subject state is learnt.

    Attributes:
        feature_service: FeatureService instance for feature extraction.
        result_writer: Optional ResultWriter — if provided, this step caches
            ``mask_info`` on it so downstream image-saving code can reconstruct
            label volumes.
    """

    def __init__(self, feature_service: FeatureService, result_writer: Optional[Any] = None):
        """
        Args:
            feature_service: FeatureService instance.
            result_writer: Deprecated; retained only for backward-compatible
                construction. The step no longer writes to
                ``result_writer.mask_info_cache`` (see ``transform_one`` note).
                Will be removed in a future release.
        """
        super().__init__()
        self.feature_service = feature_service
        # Kept on the instance for backward compat but intentionally unused.
        self.result_writer = result_writer
        self.logger = logging.getLogger(__name__)

    def transform_one(self, subject_id: str, subject_data: Any) -> Dict[str, Any]:
        """
        Extract voxel features for a single subject.

        Args:
            subject_id: Subject identifier.
            subject_data: Unused payload from the orchestrator (the first
                pipeline step is fed an empty stub; image and mask paths
                live on ``feature_service``).

        Returns:
            Dict with keys ``features`` (processed voxel features),
            ``raw`` (raw voxel features) and ``mask_info`` (metadata for
            later image reconstruction).

        Note:
            ``mask_info`` rides through the per-subject result dict and is
            collected back into ``HabitatPipeline.mask_info_cache`` by the
            parent process (see ``HabitatPipeline._process_subjects_parallel``).
            This step does NOT write into ``result_writer.mask_info_cache``
            directly: under multi-processing each worker has its own forked
            copy of ``result_writer``, so any state mutated here is dropped
            when the worker exits.
        """
        _, feature_df, raw_df, mask_info = self.feature_service.extract_voxel_features(subject_id)
        return {
            'features': feature_df,
            'raw': raw_df,
            'mask_info': mask_info,
        }
