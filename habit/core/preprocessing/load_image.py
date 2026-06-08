# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
from typing import Dict, Any, Optional, Union, List, Tuple, Sequence
import math
import logging
import SimpleITK as sitk
import os
import numpy as np
from .base_preprocessor import BasePreprocessor
from .preprocessor_factory import PreprocessorFactory
from ...utils.log_utils import get_module_logger

logger: logging.Logger = get_module_logger(__name__)

@PreprocessorFactory.register("load_image")
class LoadImagePreprocessor(BasePreprocessor):
    """Load images from file paths and convert them to SimpleITK Image objects.
    
    This preprocessor takes keys from the subject_data dictionary, loads the corresponding
    files as SimpleITK images, and replaces the file paths with the loaded image objects.
    Keys not specified will remain unchanged.
    """
    
    def __init__(
        self,
        keys: Union[str, List[str]],
        allow_missing_keys: bool = True,
        **kwargs
    ):
        """Initialize the LoadImage preprocessor.
        
        Args:
            keys (Union[str, List[str]]): Keys of the items to load as SimpleITK images.
            allow_missing_keys (bool): If True, allows missing keys in the input data.
            **kwargs: Additional parameters.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        
        # Convert single key to list
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        
    def _load_sitk_image(self, image_path: str) -> sitk.Image:
        """Load a SimpleITK image from a file path.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            sitk.Image: Loaded SimpleITK image
            
        Raises:
            FileNotFoundError: If the image file does not exist
            RuntimeError: If the image cannot be loaded
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} does not exist")
            
        try:
            # Load the image with SimpleITK
            sitk_image = sitk.ReadImage(image_path)
            return sitk_image
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")

    def _warn_if_invalid_spacing(
        self,
        sitk_image: sitk.Image,
        image_path: str,
        key: str,
    ) -> None:
        """Emit log warning when spacing is not strictly positive and finite.

        Negative slice spacing, zeros, NaN, or Inf in metadata (often from bad
        DICOM/NIfTI export) break or distort resampling and registration.

        Args:
            sitk_image: Loaded SimpleITK image (spacing is in mm per axis).
            image_path: Source file path (for diagnostics in log).
            key: Batch key for the image (e.g. modality name).
        """
        spacing_t: Tuple[float, ...] = sitk_image.GetSpacing()
        bad_dims: List[int] = [
            i
            for i, s in enumerate(spacing_t)
            if (not math.isfinite(s)) or s <= 0
        ]
        if not bad_dims:
            return
        logger.warning(
            "Invalid voxel spacing on dimension index(es) %s (each spacing must be "
            "finite and > 0). Common causes include incorrect DICOM/NIfTI metadata "
            "(e.g. negative slice spacing) or corrupted headers; downstream "
            "resampling and registration may fail or be incorrect. key=%s path=%s "
            "spacing=%s",
            bad_dims,
            key,
            image_path,
            tuple(spacing_t),
        )

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load specified keys from the data dictionary as SimpleITK images.
        
        Args:
            data (Dict[str, Any]): Input data dictionary containing file paths.
            
        Returns:
            Dict[str, Any]: Data dictionary with file paths replaced by SimpleITK images.
        """
        self._check_keys(data)
        
        # Process each specified key
        for key in self.keys:
            # Skip if key is missing and we allow missing keys
            if key not in data:
                if self.allow_missing_keys:
                    continue
                else:
                    raise KeyError(f"Key {key} not found in data dictionary")
                
            # Get the file path
            image_path = data[key]
            
            # Skip if not a string (already processed or not a path)
            if not isinstance(image_path, str):
                continue
                
            try:
                # Load the image
                sitk_image = self._load_sitk_image(image_path)
                self._warn_if_invalid_spacing(sitk_image, image_path, key)

                # Replace the file path with the SimpleITK image
                data[key] = sitk_image
                
                # Initialize or update metadata
                meta_key = f"{key}_meta_dict"
                if meta_key not in data:
                    data[meta_key] = {}
                    
                # Store important image properties in metadata
                data[meta_key]["spacing"] = sitk_image.GetSpacing()
                data[meta_key]["size"] = sitk_image.GetSize()
                data[meta_key]["origin"] = sitk_image.GetOrigin()
                data[meta_key]["direction"] = sitk_image.GetDirection()
                data[meta_key]["pixel_type"] = sitk_image.GetPixelIDTypeAsString()
                data[meta_key]["image_path"] = image_path  # Add image path to metadata
                
            except Exception as e:
                print(f"Error loading image for key {key}: {e}")
                if not self.allow_missing_keys:
                    raise
    