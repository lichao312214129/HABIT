"""
Habitat image writer for habitat analysis.

This module writes image artefacts only: per-subject supervoxel maps and
habitat maps reconstructed from pipeline label outputs.
"""

import os
import logging
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import SimpleITK as sitk
from typing import List, Optional, Tuple, Dict, Any, Union

from habit.utils.io_utils import save_habitat_image
from habit.utils.parallel_utils import parallel_map
from habit.utils.habitat_postprocess_utils import remove_small_connected_components
from ..config_schemas import HabitatAnalysisConfig, ResultColumns
from ..pipelines.pipeline_serialization import apply_mask_metadata_to_sitk_image

# Lightweight per-subject payload for spawn workers. Must stay picklable without
# binding HabitatImageWriter (bound methods pickle the whole writer instance).
HabitatImageSaveTask = Tuple[
    str,
    pd.DataFrame,
    str,
    Dict[str, Any],
    Optional[Union[str, Path]],
    int,
]


def _save_habitat_image_worker(
    item: HabitatImageSaveTask,
) -> Tuple[str, Optional[Exception]]:
    """
    Save one subject habitat NRRD in an isolated spawn child.

    Args:
        item: ``(subject_id, subject_df, out_dir, postprocess_settings,
            log_file_path, log_level)``.

    Returns:
        ``(subject_id, None)`` on success or ``(subject_id, Exception)`` on failure.
    """
    subject, subject_df, out_dir, postprocess_settings, log_file_path, log_level = item

    if log_file_path is not None:
        from habit.utils.log_utils import restore_logging_in_subprocess

        restore_logging_in_subprocess(log_file_path, log_level)

    try:
        supervoxel_path = os.path.join(out_dir, f"{subject}_supervoxel.nrrd")
        save_habitat_image(
            subject,
            subject_df,
            supervoxel_path,
            out_dir,
            postprocess_settings=postprocess_settings,
        )
        return subject, None
    except Exception as exc:
        return subject, Exception(str(exc))


def estimate_habitat_image_worker_pickle_bytes(item: HabitatImageSaveTask) -> int:
    """
    Estimate unpickled payload size for one spawn worker item.

    Used in tests to guard against accidentally reintroducing large bound-method
    payloads when saving habitat images in parallel.
    """
    return len(pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL))


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

        # Log file path for subprocesses (set via set_logging_info).
        self._log_file_path = None
        self._log_level = logging.INFO

    def set_logging_info(self, log_file_path: str, log_level: int):
        """Set logging info for subprocesses."""
        self._log_file_path = log_file_path
        self._log_level = log_level

    def _ensure_logging_in_subprocess(self) -> None:
        """
        Ensure logging is properly configured in child processes.
        """
        from habit.utils.log_utils import restore_logging_in_subprocess
        
        if self._log_file_path:
            restore_logging_in_subprocess(self._log_file_path, self._log_level)

    def save_supervoxel_image(
        self,
        subject: str,
        supervoxel_labels: np.ndarray,
        mask_info: dict
    ) -> None:
        """Save supervoxel labels as an image file."""
        if not isinstance(mask_info, dict):
            return
        if "mask_array" not in mask_info:
            return
        
        supervoxel_map = np.zeros_like(mask_info['mask_array'])
        mask_indices = mask_info['mask_array'] > 0
        supervoxel_map[mask_indices] = supervoxel_labels
        
        supervoxel_img = sitk.GetImageFromArray(supervoxel_map)
        apply_mask_metadata_to_sitk_image(supervoxel_img, mask_info)
        
        output_path = os.path.join(
            self.config.out_dir, f"{subject}_supervoxel.nrrd"
        )
        sitk.WriteImage(supervoxel_img, output_path)

    def save_habitat_for_subject(
        self,
        item: Tuple[str, pd.DataFrame],
    ) -> Tuple[str, Optional[Exception]]:
        """
        Save the habitat NRRD for one subject (in-process helper).

        Parallel saves use :func:`_save_habitat_image_worker` instead of this
        bound method so spawn workers do not pickle ``self.results_df`` or
        ``self.mask_info_cache``.
        """
        subject, subject_df = item
        return _save_habitat_image_worker(
            (
                subject,
                subject_df,
                self.config.out_dir,
                self.config.HabitatSegmentation.postprocess_habitat.model_dump(),
                self._log_file_path,
                self._log_level,
            )
        )

    def save_all_habitat_images(self, failed_subjects: List[str]) -> None:
        """Save habitat images for all successfully processed subjects."""
        if self.results_df is None:
            raise ValueError("results_df must be set before saving habitat images")

        results_df = self.results_df.copy(deep=True)
        is_voxel_level = (
            'VoxelIndex' in results_df.columns
            or ResultColumns.SUPERVOXEL not in results_df.columns
        )
        if is_voxel_level:
            self._save_all_voxel_habitat_images(results_df, failed_subjects)
            return

        if ResultColumns.SUBJECT in results_df.columns:
            results_df = results_df.set_index(ResultColumns.SUBJECT, drop=False)

        subjects_to_save = sorted(set(results_df.index))
        if not subjects_to_save:
            return

        if self.config.verbose:
            self.logger.info(
                f"Saving habitat images for {len(subjects_to_save)} subjects..."
            )

        postprocess_settings = (
            self.config.HabitatSegmentation.postprocess_habitat.model_dump()
        )
        out_dir = self.config.out_dir
        items: List[HabitatImageSaveTask] = [
            (
                subject,
                results_df.loc[[subject]],
                out_dir,
                postprocess_settings,
                self._log_file_path,
                self._log_level,
            )
            for subject in subjects_to_save
        ]

        _, failed = parallel_map(
            func=_save_habitat_image_worker,
            items=items,
            n_processes=self.config.processes,
            desc="Saving habitat images",
            logger=self.logger,
            show_progress=True,
            log_file_path=self._log_file_path,
            log_level=self._log_level,
            oom_backoff=getattr(self.config, "oom_backoff", True),
        )

        if failed and self.config.verbose:
            self.logger.warning(
                f"Failed to save habitat images for {len(failed)} subject(s)"
            )

    def _save_all_voxel_habitat_images(self, results_df: pd.DataFrame, failed_subjects: List[str]) -> None:
        """Save habitat images for voxel-level results (direct pooling)."""
        subjects = results_df[ResultColumns.SUBJECT].unique()
        
        if self.config.verbose:
            self.logger.info(f"Saving voxel-level habitat images for {len(subjects)} subjects...")

        for subject in subjects:
            if subject in failed_subjects:
                continue
            
            # Get labels for this subject
            subject_df = results_df[results_df[ResultColumns.SUBJECT] == subject]
            labels = subject_df[ResultColumns.HABITATS].values
            
            # mask_info_cache is always initialised in __init__, so we only
            # need to check membership for this subject.
            mask_info = self.mask_info_cache.get(subject)
            if mask_info is None:
                self.logger.warning(
                    f"Mask info for {subject} not found in cache, skipping image saving"
                )
                continue

            self._save_direct_habitat_image(subject, labels, mask_info)

    def _save_direct_habitat_image(
        self,
        subject: str,
        labels: np.ndarray,
        mask_info: Dict[str, Any]
    ) -> None:
        """
        Save habitat image for a single subject using voxel-level labels.
        """
        mask_array = mask_info["mask_array"]
        mask_indices = mask_array > 0
        
        if np.sum(mask_indices) != len(labels):
            self.logger.warning(
                f"Subject {subject}: voxel count mismatch "
                f"(mask voxels={np.sum(mask_indices)}, labels={len(labels)})"
            )
            # Adjust labels length if mismatch
            if len(labels) > np.sum(mask_indices):
                labels = labels[:np.sum(mask_indices)]
            else:
                # Pad with zeros
                new_labels = np.zeros(np.sum(mask_indices))
                new_labels[:len(labels)] = labels
                labels = new_labels

        habitat_map = np.zeros_like(mask_array)
        habitat_map[mask_indices] = labels

        habitat_map = remove_small_connected_components(
            label_map=habitat_map.astype(np.int32, copy=False),
            roi_mask=mask_indices,
            settings=self.config.HabitatSegmentation.postprocess_habitat.model_dump()
        )

        habitat_img = sitk.GetImageFromArray(habitat_map)
        apply_mask_metadata_to_sitk_image(habitat_img, mask_info)

        output_path = os.path.join(self.config.out_dir, f"{subject}_habitats.nrrd")
        sitk.WriteImage(habitat_img, output_path)

    def save_habitat_image_from_voxels(
        self,
        subject: str,
        labels: np.ndarray,
        mask_info: Dict[str, Any]
    ) -> None:
        """
        Save habitat image directly from voxel-level labels.

        Args:
            subject: Subject ID
            labels: Voxel-level habitat labels (1-indexed)
            mask_info: Mask metadata for reconstruction
        """
        self._save_direct_habitat_image(subject, labels, mask_info)
