"""
Result Manager for Habitat Analysis.
Handles result storage, saving images, and exporting data.
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

class ResultManager:
    """
    Manages result storage and output for habitat analysis.
    """
    
    def __init__(self, config: HabitatAnalysisConfig, logger: logging.Logger):
        """
        Initialize ResultManager.
        
        Args:
            config: Habitat analysis configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        self.X = None
        self.supervoxel_labels = {}
        self.habitat_labels = None
        self.results_df = None
        
        # Log file path for subprocesses
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
        subject: str
    ) -> Tuple[str, Optional[Exception]]:
        """
        Save habitat image for a single subject.
        
        Args:
            subject: Subject ID
            
        Returns:
            Tuple of (subject_id, None or Exception)
        """
        self._ensure_logging_in_subprocess()
        
        try:
            # We need results_df to be set before calling this in parallel, 
            # or passed as argument. Since this method is usually called via
            # parallel_map which pickles the instance/method, self.results_df
            # should be available if it was set before mapping.
            # However, for safety with multiprocessing, it's better if results_df 
            # is self-contained or accessible.
            # Here we assume self.results_df is available in the instance state.
            
            supervoxel_path = os.path.join(
                self.config.out_dir, f"{subject}_supervoxel.nrrd"
            )
            save_habitat_image(
                subject,
                self.results_df,
                supervoxel_path,
                self.config.out_dir,
                postprocess_settings=self.config.HabitatsSegmention.postprocess_habitat.model_dump()
            )
            return subject, None
        except Exception as e:
            return subject, Exception(str(e))

    def save_all_habitat_images(self, failed_subjects: List[str]) -> None:
        """Save habitat images for all successfully processed subjects."""
        # Determine if we have supervoxel mapping or direct voxel results
        is_voxel_level = 'VoxelIndex' in self.results_df.columns or ResultColumns.SUPERVOXEL not in self.results_df.columns
        
        if is_voxel_level:
            self._save_all_voxel_habitat_images(failed_subjects)
            return

        # Set Subject as index for save function if not already
        if ResultColumns.SUBJECT in self.results_df.columns:
            self.results_df.set_index(ResultColumns.SUBJECT, inplace=True)
        
        # Get unique subjects
        subjects_to_save = list(set(self.results_df.index))
        
        if self.config.verbose:
            self.logger.info(f"Saving habitat images for {len(subjects_to_save)} subjects...")
        
        results, failed = parallel_map(
            func=self.save_habitat_for_subject,
            items=subjects_to_save,
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
            
            # Get mask info
            mask_info = None
            if hasattr(self, 'mask_info_cache') and subject in self.mask_info_cache:
                mask_info = self.mask_info_cache[subject]
            else:
                # Fallback: try to load mask info if not cached (should not happen if using pipeline)
                self.logger.warning(f"Mask info for {subject} not found in cache, skipping image saving")
                continue
                
            if mask_info:
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
