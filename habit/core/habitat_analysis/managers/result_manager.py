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
from ..config import ResultColumns
from ..config_schemas import HabitatAnalysisConfig

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
                subject, self.results_df, supervoxel_path, self.config.out_dir
            )
            return subject, None
        except Exception as e:
            return subject, Exception(str(e))

    def save_all_habitat_images(self, failed_subjects: List[str]) -> None:
        """Save habitat images for all successfully processed subjects."""
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
