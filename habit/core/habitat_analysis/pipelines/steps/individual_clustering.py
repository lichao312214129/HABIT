"""
Individual-level clustering step for habitat analysis pipeline.

This step clusters voxels to supervoxels (or habitats) for each subject independently.
"""

from typing import Any, Dict, Literal
import logging
from habit.utils.log_utils import get_module_logger

import numpy as np

from ..base_pipeline import IndividualLevelStep
from ..habitat_subject_data import HabitatSubjectData
from ...services.clustering_service import ClusteringService
from ...services.habitat_image_writer import HabitatImageWriter
from ...config_schemas import HabitatAnalysisConfig
from habit.utils.habitat_postprocess_utils import remove_small_connected_components


class IndividualClusteringStep(IndividualLevelStep):
    """
    Individual-level clustering (voxel to supervoxel or voxel to habitat).

    Stateless: clustering parameters are fixed by configuration or computed
    per subject. Each subject is clustered independently in
    :meth:`transform_one`.

    Attributes:
        feature_service: FeatureService instance (for accessing data).
        clustering_service: ClusteringService instance.
        habitat_image_writer: HabitatImageWriter instance (for saving supervoxel maps).
        config: Configuration object.
        target: 'supervoxel' for two-step strategy, 'habitat' for one-step.
        find_optimal: Whether to find optimal cluster number (one-step only).
    """

    def __init__(
        self,
        feature_service: Any,  # FeatureService
        clustering_service: ClusteringService,
        habitat_image_writer: HabitatImageWriter,
        config: HabitatAnalysisConfig,
        target: Literal['supervoxel', 'habitat'] = 'supervoxel',
        find_optimal: bool = False
    ):
        super().__init__()
        self.feature_service = feature_service
        self.clustering_service = clustering_service
        self.habitat_image_writer = habitat_image_writer
        self.config = config
        self.target = target
        self.find_optimal = find_optimal
        self.logger = get_module_logger(__name__)

    def transform_one(self, subject_id: str, subject_data: HabitatSubjectData) -> HabitatSubjectData:
        """Cluster one subject's voxels to supervoxels (or habitats)."""
        feature_df = subject_data.require_features(self.__class__.__name__)
        raw_df = subject_data.require_raw(self.__class__.__name__)
        mask_info = subject_data.require_mask_info(self.__class__.__name__)

        n_clusters = self._resolve_n_clusters(subject_id, feature_df, mask_info)
        labels = self.clustering_service.cluster_subject_voxels(
            subject_id,
            feature_df,
            n_clusters=n_clusters,
            mask_info=mask_info,
        )
        labels = self._postprocess_labels_if_enabled(labels, mask_info)

        if self.config.save_images:
            self._save_label_image(subject_id, labels, mask_info)

        if self.config.plot_curves:
            self._visualize(subject_id, feature_df, labels, n_clusters)

        return HabitatSubjectData(
            features=feature_df,
            raw=raw_df,
            mask_info=mask_info,
            supervoxel_labels=labels,
        )

    def _resolve_n_clusters(
        self,
        subject_id: str,
        feature_df: Any,
        mask_info: Any,
    ) -> int:
        """
        Pick the number of clusters for this subject according to ``target``
        and the one-step configuration block.
        """
        if self.target == 'supervoxel':
            return self.config.HabitatSegmentation.supervoxel.n_clusters

        # target == 'habitat' (one-step mode)
        one_step_cfg = self.config.HabitatSegmentation.supervoxel.one_step_settings

        if one_step_cfg.fixed_n_clusters is not None:
            return one_step_cfg.fixed_n_clusters
        if self.find_optimal:
            return self.clustering_service.find_optimal_clusters_for_subject(
                subject_id,
                feature_df,
                min_clusters=one_step_cfg.min_clusters,
                max_clusters=one_step_cfg.max_clusters,
                selection_method=one_step_cfg.selection_method,
                plot_validation=self.config.plot_curves,
                mask_info=mask_info,
            )
        # Fall back to supervoxel n_clusters when one-step is not set up to find optimal.
        return self.config.HabitatSegmentation.supervoxel.n_clusters

    def _save_label_image(self, subject_id: str, labels: np.ndarray, mask_info: Any) -> None:
        if self.target == 'supervoxel':
            self.habitat_image_writer.save_supervoxel_image(subject_id, labels, mask_info)
        else:
            self.habitat_image_writer.save_habitat_image_from_voxels(subject_id, labels, mask_info)

    def _visualize(
        self,
        subject_id: str,
        feature_df: Any,
        labels: np.ndarray,
        n_clusters: int,
    ) -> None:
        if self.target == 'supervoxel':
            self.clustering_service.visualize_supervoxel_clustering(
                subject_id, feature_df, labels
            )
        else:
            self.clustering_service.visualize_habitat_clustering(
                feature_df.values,
                labels,
                n_clusters,
                subject=subject_id,
                output_dir=self.config.out_dir,
            )

    def _postprocess_labels_if_enabled(
        self,
        labels: np.ndarray,
        mask_info: Dict[str, Any]
    ) -> np.ndarray:
        """
        Optionally clean tiny connected components in voxel-level labels.

        Args:
            labels: 1D voxel labels (1-indexed) for ROI voxels.
            mask_info: Dictionary containing mask_array for spatial reconstruction.

        Returns:
            np.ndarray: Post-processed 1D labels (1-indexed).
        """
        if not isinstance(mask_info, dict) or "mask_array" not in mask_info:
            return labels

        if self.target == "supervoxel":
            pp_cfg = self.config.HabitatSegmentation.postprocess_supervoxel
        else:
            pp_cfg = self.config.HabitatSegmentation.postprocess_habitat

        if not pp_cfg.enabled:
            return labels

        mask_array = mask_info["mask_array"]
        roi_mask = mask_array > 0
        label_map = np.zeros_like(mask_array, dtype=np.int32)
        label_map[roi_mask] = labels.astype(np.int32)

        cleaned = remove_small_connected_components(
            label_map=label_map,
            roi_mask=roi_mask,
            settings=pp_cfg.model_dump()
        )
        return cleaned[roi_mask].astype(labels.dtype, copy=False)
