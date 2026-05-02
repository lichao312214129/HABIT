"""
Result Writer for Habitat Analysis.
Handles result storage, saving habitat NRRD images, and exporting CSV data.
"""

import os
import logging
import pandas as pd
import numpy as np
import SimpleITK as sitk
from typing import List, Optional, Tuple, Dict, Any

from habit.utils.io_utils import save_habitat_image
from habit.utils.parallel_utils import parallel_map
from habit.utils.habitat_postprocess_utils import remove_small_connected_components
from ..config_schemas import HabitatAnalysisConfig, ResultColumns

class ResultWriter:
    """
    Writes habitat-analysis results to disk.

    Handles two artefact families:
        * the result DataFrame -> ``habitats.csv``;
        * per-subject habitat label volumes -> ``<subject>_habitat.nrrd`` /
          ``<subject>_supervoxel.nrrd``.
    """

    def __init__(self, config: HabitatAnalysisConfig, logger: logging.Logger):
        """
        Initialize ResultWriter.
        
        Args:
            config: Habitat analysis configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger

        # Result DataFrame is published by HabitatAnalysis after the pipeline
        # finishes; per-subject save methods read it from here.
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
        if 'mask_array' not in mask_info or 'mask' not in mask_info:
            return
        
        supervoxel_map = np.zeros_like(mask_info['mask_array'])
        mask_indices = mask_info['mask_array'] > 0
        supervoxel_map[mask_indices] = supervoxel_labels
        
        supervoxel_img = sitk.GetImageFromArray(supervoxel_map)
        supervoxel_img.CopyInformation(mask_info['mask'])
        
        output_path = os.path.join(
            self.config.out_dir, f"{subject}_supervoxel.nrrd"
        )
        sitk.WriteImage(supervoxel_img, output_path)

    def save_habitat_for_subject(
        self,
        item: Tuple[str, pd.DataFrame],
    ) -> Tuple[str, Optional[Exception]]:
        """
        Save the habitat NRRD for one subject (parallel worker entry).

        Args:
            item: ``(subject_id, subject_df)`` — a small DataFrame indexed by
                ``Subject`` containing only this subject's rows. We pass the
                slice explicitly instead of relying on ``self.results_df`` so
                ``parallel_map`` only pickles per-subject data into each
                worker, not the full results table.

        Returns:
            Tuple of (subject_id, None on success / Exception on failure).
        """
        subject, subject_df = item
        self._ensure_logging_in_subprocess()
        try:
            supervoxel_path = os.path.join(
                self.config.out_dir, f"{subject}_supervoxel.nrrd"
            )
            save_habitat_image(
                subject,
                subject_df,
                supervoxel_path,
                self.config.out_dir,
                postprocess_settings=self.config.HabitatsSegmention.postprocess_habitat.model_dump(),
            )
            return subject, None
        except Exception as exc:
            return subject, Exception(str(exc))

    def save_all_habitat_images(self, failed_subjects: List[str]) -> None:
        """Save habitat images for all successfully processed subjects."""
        is_voxel_level = (
            'VoxelIndex' in self.results_df.columns
            or ResultColumns.SUPERVOXEL not in self.results_df.columns
        )
        if is_voxel_level:
            self._save_all_voxel_habitat_images(failed_subjects)
            return

        # Use Subject as the row index so save_habitat_image can do .loc[subject].
        if ResultColumns.SUBJECT in self.results_df.columns:
            self.results_df.set_index(ResultColumns.SUBJECT, inplace=True)

        subjects_to_save = sorted(set(self.results_df.index))
        if not subjects_to_save:
            return

        if self.config.verbose:
            self.logger.info(
                f"Saving habitat images for {len(subjects_to_save)} subjects..."
            )

        # Pre-slice in the parent so each worker only receives the rows it
        # needs. ``loc[[subject]]`` keeps the Subject index in place even for
        # a single row, which is what save_habitat_image expects internally.
        items: List[Tuple[str, pd.DataFrame]] = [
            (subject, self.results_df.loc[[subject]])
            for subject in subjects_to_save
        ]

        _, failed = parallel_map(
            func=self.save_habitat_for_subject,
            items=items,
            n_processes=self.config.processes,
            desc="Saving habitat images",
            logger=self.logger,
            show_progress=True,
            log_file_path=self._log_file_path,
            log_level=self._log_level,
        )

        if failed and self.config.verbose:
            self.logger.warning(
                f"Failed to save habitat images for {len(failed)} subject(s)"
            )

    def _save_all_voxel_habitat_images(self, failed_subjects: List[str]) -> None:
        """Save habitat images for voxel-level results (direct pooling)."""
        subjects = self.results_df[ResultColumns.SUBJECT].unique()
        
        if self.config.verbose:
            self.logger.info(f"Saving voxel-level habitat images for {len(subjects)} subjects...")

        for subject in subjects:
            if subject in failed_subjects:
                continue
            
            # Get labels for this subject
            subject_df = self.results_df[self.results_df[ResultColumns.SUBJECT] == subject]
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
            settings=self.config.HabitatsSegmention.postprocess_habitat.model_dump()
        )

        habitat_img = sitk.GetImageFromArray(habitat_map)
        habitat_img.CopyInformation(mask_info["mask"])

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
