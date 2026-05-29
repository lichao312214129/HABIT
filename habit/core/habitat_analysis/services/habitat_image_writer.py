"""
Habitat image writer for habitat analysis.

This module writes image artefacts only: per-subject supervoxel maps and
habitat maps reconstructed from pipeline label outputs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import SimpleITK as sitk

from habit.utils.habitat_postprocess_utils import remove_small_connected_components
from habit.utils.parallel_utils import default_thread_worker_count, thread_map
from ..config_schemas import HabitatAnalysisConfig, ResultColumns
from ..pipelines.pipeline_serialization import apply_mask_metadata_to_sitk_image

SupervoxelHabitatSaveTask = Tuple[
    str,
    pd.DataFrame,
    str,
    str,
    Dict[str, Any],
]

VoxelHabitatSaveTask = Tuple[
    str,
    np.ndarray,
    Dict[str, Any],
    str,
    Dict[str, Any],
]


def save_habitat_from_supervoxel_mapping(
    subject: str,
    habitats_df: pd.DataFrame,
    supervoxel_path: str,
    out_dir: str,
    postprocess_settings: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save one habitat NRRD by mapping supervoxel labels to habitat labels.

    Used by two_step publish-time batch export when ``*_supervoxel.nrrd`` files
    already exist on disk.

    Args:
        subject: Subject identifier.
        habitats_df: Per-subject rows containing ``Supervoxel`` and ``Habitats``.
        supervoxel_path: Path to the subject supervoxel NRRD file.
        out_dir: Output directory for ``<subject>_habitats.nrrd``.
        postprocess_settings: Optional post-process settings dictionary.

    Returns:
        str: Path to the saved habitat NRRD file.
    """
    supervoxel = sitk.ReadImage(supervoxel_path)
    supervoxel_array = sitk.GetArrayFromImage(supervoxel)

    habitats_array = np.zeros_like(supervoxel_array)
    habitats_subj = habitats_df.loc[subject]
    n_clusters_supervoxel = habitats_subj.shape[0]
    for cluster_idx in range(n_clusters_supervoxel):
        supervoxel_id = cluster_idx + 1
        if (supervoxel_array == supervoxel_id).sum() > 0:
            habitat_rows = habitats_subj[
                habitats_subj[ResultColumns.SUPERVOXEL] == supervoxel_id
            ]
            habitats_array[supervoxel_array == supervoxel_id] = habitat_rows[
                ResultColumns.HABITATS
            ].values[0]

    roi_mask = supervoxel_array > 0
    if postprocess_settings and postprocess_settings.get("enabled", False):
        habitats_array = remove_small_connected_components(
            label_map=habitats_array.astype(np.int32, copy=False),
            roi_mask=roi_mask,
            settings=postprocess_settings,
        )

    habitats_img = sitk.GetImageFromArray(habitats_array)
    habitats_img.CopyInformation(supervoxel)

    output_path = os.path.join(out_dir, f"{subject}_habitats.nrrd")
    sitk.WriteImage(habitats_img, output_path)
    return output_path


def save_habitat_from_voxel_labels(
    subject: str,
    labels: np.ndarray,
    mask_info: Dict[str, Any],
    out_dir: str,
    postprocess_settings: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Save one habitat NRRD directly from per-voxel habitat labels.

    Used by one_step (immediate save after clustering) and direct_pooling
    (publish-time batch export).

    Args:
        subject: Subject identifier.
        labels: 1D voxel habitat labels aligned with ROI voxels in ``mask_info``.
        mask_info: Mask metadata containing ``mask_array`` and SITK header fields.
        out_dir: Output directory for ``<subject>_habitats.nrrd``.
        postprocess_settings: Post-process settings dictionary.
        logger: Optional logger for non-fatal warnings.

    Returns:
        str: Path to the saved habitat NRRD file.
    """
    mask_array = mask_info["mask_array"]
    mask_indices = mask_array > 0
    roi_voxel_count = int(np.sum(mask_indices))
    labels = np.asarray(labels).ravel()

    if roi_voxel_count != len(labels):
        if logger is not None:
            logger.warning(
                "Subject %s: voxel count mismatch "
                "(mask voxels=%s, labels=%s)",
                subject,
                roi_voxel_count,
                len(labels),
            )
        if len(labels) > roi_voxel_count:
            labels = labels[:roi_voxel_count]
        else:
            padded_labels = np.zeros(roi_voxel_count, dtype=labels.dtype)
            padded_labels[: len(labels)] = labels
            labels = padded_labels

    habitat_map = np.zeros_like(mask_array)
    habitat_map[mask_indices] = labels
    habitat_map = remove_small_connected_components(
        label_map=habitat_map.astype(np.int32, copy=False),
        roi_mask=mask_indices,
        settings=postprocess_settings,
    )

    habitat_img = sitk.GetImageFromArray(habitat_map)
    apply_mask_metadata_to_sitk_image(habitat_img, mask_info)

    output_path = os.path.join(out_dir, f"{subject}_habitats.nrrd")
    sitk.WriteImage(habitat_img, output_path)
    return output_path


def _save_habitat_from_supervoxel_mapping_worker(
    item: SupervoxelHabitatSaveTask,
) -> Tuple[str, Optional[Exception]]:
    """
    Thread-pool worker for one two_step habitat image export task.
    """
    subject, subject_df, out_dir, supervoxel_path, postprocess_settings = item
    try:
        save_habitat_from_supervoxel_mapping(
            subject=subject,
            habitats_df=subject_df,
            supervoxel_path=supervoxel_path,
            out_dir=out_dir,
            postprocess_settings=postprocess_settings,
        )
        return subject, None
    except Exception as exc:
        return subject, Exception(str(exc))


def _save_habitat_from_voxel_labels_worker(
    item: VoxelHabitatSaveTask,
) -> Tuple[str, Optional[Exception]]:
    """
    Thread-pool worker for one voxel-level habitat image export task.
    """
    subject, labels, mask_info, out_dir, postprocess_settings = item
    try:
        save_habitat_from_voxel_labels(
            subject=subject,
            labels=labels,
            mask_info=mask_info,
            out_dir=out_dir,
            postprocess_settings=postprocess_settings,
            logger=None,
        )
        return subject, None
    except Exception as exc:
        return subject, Exception(str(exc))


class HabitatImageWriter:
    """
    Writes habitat-analysis image artefacts to disk.

    Handles per-subject habitat and supervoxel label volumes:
    ``<subject>_habitats.nrrd`` and ``<subject>_supervoxel.nrrd``.
    CSV publishing lives in :class:`HabitatResultPublisher`.
    """

    def __init__(self, config: HabitatAnalysisConfig, logger: logging.Logger):
        """
        Initialize HabitatImageWriter.

        Args:
            config: Habitat analysis configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger

        # Result DataFrame is supplied by HabitatResultPublisher after the
        # pipeline finishes; per-subject save methods read it from here.
        self.results_df: Optional[pd.DataFrame] = None

        # Mask metadata cache, populated by VoxelFeatureExtractor (in the main
        # process) and consumed by save_*_habitat_image* methods. Declared here
        # so callers do not need ``hasattr`` guards.
        self.mask_info_cache: Dict[str, Dict[str, Any]] = {}

        # Log file path retained for API compatibility with existing callers.
        self._log_file_path: Optional[Union[str, Path]] = None
        self._log_level = logging.INFO

    def set_logging_info(self, log_file_path: str, log_level: int) -> None:
        """Set logging info for subprocesses."""
        self._log_file_path = log_file_path
        self._log_level = log_level

    def save_supervoxel_image(
        self,
        subject: str,
        supervoxel_labels: np.ndarray,
        mask_info: dict,
    ) -> None:
        """Save supervoxel labels as an image file."""
        if not isinstance(mask_info, dict):
            return
        if "mask_array" not in mask_info:
            return

        supervoxel_map = np.zeros_like(mask_info["mask_array"])
        mask_indices = mask_info["mask_array"] > 0
        supervoxel_map[mask_indices] = supervoxel_labels

        supervoxel_img = sitk.GetImageFromArray(supervoxel_map)
        apply_mask_metadata_to_sitk_image(supervoxel_img, mask_info)

        output_path = os.path.join(
            self.config.out_dir, f"{subject}_supervoxel.nrrd"
        )
        sitk.WriteImage(supervoxel_img, output_path)

    def save_all_habitat_images(self, failed_subjects: List[str]) -> None:
        """Save habitat images for all successfully processed subjects."""
        if self.results_df is None:
            raise ValueError("results_df must be set before saving habitat images")

        results_df = self.results_df.copy(deep=True)
        is_voxel_level = (
            "VoxelIndex" in results_df.columns
            or ResultColumns.SUPERVOXEL not in results_df.columns
        )
        postprocess_settings = (
            self.config.HabitatSegmentation.postprocess_habitat.model_dump()
        )
        out_dir = self.config.out_dir

        if is_voxel_level:
            self._save_all_habitat_images_from_voxels(
                results_df=results_df,
                failed_subjects=failed_subjects,
                out_dir=out_dir,
                postprocess_settings=postprocess_settings,
            )
            return

        if ResultColumns.SUBJECT in results_df.columns:
            results_df = results_df.set_index(ResultColumns.SUBJECT, drop=False)

        subjects_to_save = sorted(set(results_df.index))
        if not subjects_to_save:
            return

        if self.config.verbose:
            self.logger.info(
                "Saving habitat images for %s subjects...",
                len(subjects_to_save),
            )

        items: List[SupervoxelHabitatSaveTask] = [
            (
                subject,
                results_df.loc[[subject]],
                out_dir,
                os.path.join(out_dir, f"{subject}_supervoxel.nrrd"),
                postprocess_settings,
            )
            for subject in subjects_to_save
        ]
        self._run_batch_habitat_save(
            worker=_save_habitat_from_supervoxel_mapping_worker,
            items=items,
        )

    def _save_all_habitat_images_from_voxels(
        self,
        results_df: pd.DataFrame,
        failed_subjects: List[str],
        out_dir: str,
        postprocess_settings: Dict[str, Any],
    ) -> None:
        """Save habitat images for voxel-level results (direct pooling)."""
        subjects = results_df[ResultColumns.SUBJECT].unique()

        if self.config.verbose:
            self.logger.info(
                "Saving voxel-level habitat images for %s subjects...",
                len(subjects),
            )

        items: List[VoxelHabitatSaveTask] = []
        for subject in subjects:
            if subject in failed_subjects:
                continue

            subject_df = results_df[results_df[ResultColumns.SUBJECT] == subject]
            labels = subject_df[ResultColumns.HABITATS].values
            mask_info = self.mask_info_cache.get(subject)
            if mask_info is None:
                self.logger.warning(
                    "Mask info for %s not found in cache, skipping image saving",
                    subject,
                )
                continue

            items.append(
                (subject, labels, mask_info, out_dir, postprocess_settings)
            )

        self._run_batch_habitat_save(
            worker=_save_habitat_from_voxel_labels_worker,
            items=items,
        )

    def _run_batch_habitat_save(
        self,
        worker: Any,
        items: List[Any],
    ) -> None:
        """
        Run publish-time habitat image export with a bounded thread pool.

        Batch export is I/O-heavy; threads avoid duplicating full 3D volumes
        across many spawn processes.
        """
        if not items:
            return

        _, failed = thread_map(
            func=worker,
            items=items,
            max_workers=default_thread_worker_count(),
            desc="Saving habitat images",
            logger=self.logger,
            show_progress=True,
        )

        if failed and self.config.verbose:
            self.logger.warning(
                "Failed to save habitat images for %s subject(s)",
                len(failed),
            )

    def save_habitat_image_from_voxels(
        self,
        subject: str,
        labels: np.ndarray,
        mask_info: Dict[str, Any],
    ) -> None:
        """
        Save habitat image directly from voxel-level labels.

        Used by one_step mode immediately after per-subject clustering so
        successful subjects are persisted even when later subjects fail.

        Args:
            subject: Subject ID
            labels: Voxel-level habitat labels (1-indexed)
            mask_info: Mask metadata for reconstruction
        """
        save_habitat_from_voxel_labels(
            subject=subject,
            labels=labels,
            mask_info=mask_info,
            out_dir=self.config.out_dir,
            postprocess_settings=(
                self.config.HabitatSegmentation.postprocess_habitat.model_dump()
            ),
            logger=self.logger,
        )
