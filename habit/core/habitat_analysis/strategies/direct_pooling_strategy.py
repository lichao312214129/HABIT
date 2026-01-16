"""
Direct pooling strategy: concatenate all voxel features across subjects and cluster once.
"""

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk

from habit.utils.parallel_utils import parallel_map
from habit.core.habitat_analysis.config import ResultColumns
from .base_strategy import BaseHabitatStrategy

if TYPE_CHECKING:
    from habit.core.habitat_analysis.habitat_analysis import HabitatAnalysis


class DirectPoolingStrategy(BaseHabitatStrategy):
    """
    Direct pooling strategy.

    Flow:
    1) Extract voxel features for all subjects
    2) Concatenate all voxel features across subjects (e.g., 50 subjects × 100 voxels = 5000 rows)
    3) Cluster all voxels directly to identify habitats (single population-level clustering)
    4) Assign habitat labels back to each subject's voxels

    This strategy skips supervoxel generation entirely and works with raw voxel features.
    """

    def __init__(self, analysis: "HabitatAnalysis"):
        """
        Initialize direct pooling strategy.

        Args:
            analysis: HabitatAnalysis instance with shared utilities
        """
        super().__init__(analysis)

    def run(
        self,
        subjects: Optional[List[str]] = None,
        save_results_csv: bool = True
    ) -> pd.DataFrame:
        """
        Execute direct pooling clustering.

        Args:
            subjects: List of subjects to process (None means all subjects)
            save_results_csv: Whether to save results to CSV

        Returns:
            Results DataFrame
        """
        if subjects is None:
            subjects = list(self.analysis.images_paths.keys())

        features_all, subject_meta, failed_subjects = self._extract_all_voxel_features(subjects)

        if features_all.empty:
            raise ValueError("No valid voxel features for analysis")

        # Clean and preprocess features
        features_all = self.analysis.feature_manager.clean_features(features_all)
        features_all = self.analysis.feature_manager.apply_preprocessing(features_all, level='group')
        self.analysis.feature_manager.handle_mean_values(features_all, self.analysis.pipeline)

        # Perform clustering
        habitat_labels, optimal_n_clusters, scores = self.analysis.pipeline.cluster_habitats(
            features_all, self.analysis.clustering_manager.supervoxel2habitat_clustering
        )

        # Plot scores if available
        if scores and self.config.runtime.plot_curves:
            self.analysis.clustering_manager.plot_habitat_scores(scores, optimal_n_clusters)

        # Visualize clustering results
        if self.config.runtime.plot_curves:
            self.analysis.clustering_manager.visualize_habitat_clustering(
                features_all, habitat_labels, optimal_n_clusters
            )

        # Build results DataFrame
        results_df = self._build_results_df(subject_meta, habitat_labels)
        self.analysis.result_manager.results_df = results_df

        # Save results
        if save_results_csv:
            self._save_direct_pooling_results(results_df, subject_meta, habitat_labels)

        return results_df

    def _extract_all_voxel_features(
        self,
        subjects: List[str]
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[str]]:
        """
        Extract voxel features for all subjects and concatenate.

        Args:
            subjects: List of subject IDs

        Returns:
            features_all: Concatenated feature DataFrame
            subject_meta: List of metadata dicts for each subject slice
            failed_subjects: List of failed subject IDs
        """
        if self.config.runtime.verbose:
            self.logger.info("Extracting voxel features for direct pooling...")

        results, failed_subjects = parallel_map(
            func=self.analysis.feature_manager.extract_voxel_features,
            items=subjects,
            n_processes=self.config.runtime.n_processes,
            desc="Extracting voxel features",
            logger=self.logger,
            show_progress=True,
            log_file_path=self.analysis._log_file_path,
            log_level=self.analysis._log_level,
        )

        features_list = []
        subject_meta: List[Dict[str, Any]] = []
        current_start = 0

        for result in results:
            if not result.success or result.result is None:
                continue

            subject, features, _, mask_info = result.result

            # Apply subject-level preprocessing if configured
            features = self.analysis.feature_manager.apply_preprocessing(features, level='subject')

            n_voxels = len(features)
            if n_voxels == 0:
                failed_subjects.append(subject)
                continue

            features_list.append(features)
            subject_meta.append({
                "subject": subject,
                "start_idx": current_start,
                "end_idx": current_start + n_voxels,
                "mask_info": mask_info,
            })
            current_start += n_voxels

        if not features_list:
            return pd.DataFrame(), subject_meta, failed_subjects

        features_all = pd.concat(features_list, ignore_index=True)
        return features_all, subject_meta, failed_subjects

    def _build_results_df(
        self,
        subject_meta: List[Dict[str, Any]],
        habitat_labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Build results DataFrame with voxel-level habitat assignments.

        Args:
            subject_meta: Subject metadata list
            habitat_labels: Cluster labels for all voxels

        Returns:
            Results DataFrame
        """
        rows = []
        for meta in subject_meta:
            subject = meta["subject"]
            start_idx = meta["start_idx"]
            end_idx = meta["end_idx"]
            labels_slice = habitat_labels[start_idx:end_idx]

            for voxel_idx, label in enumerate(labels_slice):
                rows.append({
                    ResultColumns.SUBJECT: subject,
                    "VoxelIndex": voxel_idx,
                    ResultColumns.HABITATS: int(label),
                })

        return pd.DataFrame(rows)

    def _save_direct_pooling_results(
        self,
        results_df: pd.DataFrame,
        subject_meta: List[Dict[str, Any]],
        habitat_labels: np.ndarray
    ) -> None:
        """
        Save voxel-level results and habitat maps.

        Args:
            results_df: Results DataFrame
            subject_meta: Subject metadata list
        """
        os.makedirs(self.config.io.out_folder, exist_ok=True)

        # Save results CSV
        csv_path = os.path.join(self.config.io.out_folder, "habitats.csv")
        results_df.to_csv(csv_path, index=False)
        if self.config.runtime.verbose:
            self.logger.info(f"Results saved to {csv_path}")

        # Save habitat maps
        for meta in subject_meta:
            subject = meta["subject"]
            start_idx = meta["start_idx"]
            end_idx = meta["end_idx"]
            mask_info = meta["mask_info"]

            labels_slice = habitat_labels[start_idx:end_idx]
            self._save_direct_habitat_image(subject, labels_slice, mask_info)

    def _save_direct_habitat_image(
        self,
        subject: str,
        labels: np.ndarray,
        mask_info: Dict[str, Any]
    ) -> None:
        """
        Save habitat image for a single subject using voxel-level labels.

        Args:
            subject: Subject ID
            labels: Habitat labels for voxels
            mask_info: Mask information dict with mask and mask_array
        """
        if not isinstance(mask_info, dict):
            return
        if "mask_array" not in mask_info or "mask" not in mask_info:
            return

        mask_array = mask_info["mask_array"]
        mask_indices = mask_array > 0
        if np.sum(mask_indices) != len(labels):
            self.logger.warning(
                f"Subject {subject}: voxel count mismatch "
                f"(mask voxels={np.sum(mask_indices)}, labels={len(labels)})"
            )

        habitat_map = np.zeros_like(mask_array)
        habitat_map[mask_indices] = labels[:np.sum(mask_indices)]

        habitat_img = sitk.GetImageFromArray(habitat_map)
        habitat_img.CopyInformation(mask_info["mask"])

        output_path = os.path.join(self.config.io.out_folder, f"{subject}_habitat.nrrd")
        sitk.WriteImage(habitat_img, output_path)
