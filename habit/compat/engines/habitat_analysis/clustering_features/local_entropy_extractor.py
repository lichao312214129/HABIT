# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Voxel-level local entropy feature extractor

This module provides a feature extractor that calculates local entropy for each voxel
within a mask region. Local entropy is a measure of the randomness or information content
in the local neighborhood of a voxel.

Example usage:
    ```python
    from habit.compat.engines.habitat_analysis.clustering_features import LocalEntropyExtractor
    
    # Initialize extractor with custom parameters
    extractor = LocalEntropyExtractor(
        kernel_size=5,  # 5x5x5 neighborhood
        bins=32,        # 32 intensity bins for histogram
    )
    
    # Extract features
    features_df = extractor.extract_features(
        image_data='path/to/image.nii.gz',
        mask_data='path/to/mask.nii.gz'
    )
    ```
"""

import os
import logging
import numpy as np
import pandas as pd
import SimpleITK as sitk
from typing import Union, List, Dict, Optional, Tuple
from .base_extractor import BaseClusteringExtractor, FeatureExtractorRegistry
from .method_param_spec import MethodParamSpec
from habit.kernels.voxel_texture import local_entropy_map
from habit.utils.progress_utils import CustomTqdm

@FeatureExtractorRegistry.register('local_entropy')
class LocalEntropyExtractor(BaseClusteringExtractor):
    """
    Extract voxel-level local entropy features from image within mask region
    Local entropy is a measure of the randomness in the local neighborhood of a voxel
    """

    # DSL contract: local_entropy(<modality>, kernel_size, bins) — both optional.
    method_param_spec = MethodParamSpec(
        required=(),
        optional={"kernel_size": 3, "bins": 32},
        takes_image=True,
    )
    
    def __init__(self, **kwargs):
        """
        Initialize local entropy feature extractor
        
        Args:
            **kwargs: Additional parameters including:
                - kernel_size: Size of the local neighborhood kernel (default: 3)
                - bins: Number of bins for histogram calculation (default: 32)
        """
        super().__init__(**kwargs)
        self.kernel_size = kwargs.get('kernel_size', 3)
        self.bins = kwargs.get('bins', 32)
        self.feature_names = f'local_entropy-{kwargs["image"]}'
    
    def _calculate_entropy(self, image_array: np.ndarray, mask_array: np.ndarray) -> np.ndarray:
        """
        Calculate local entropy for each voxel inside the mask.

        The entropy map itself comes from
        :func:`habit.kernels.voxel_texture.local_entropy_map`, shared with the
        v1.0 domain extractor so both paths produce identical values.

        Args:
            image_array: 3D image array
            mask_array: 3D mask array

        Returns:
            np.ndarray: Local entropy values for each voxel in the mask
        """
        pbar = CustomTqdm(total=2, desc="Calculating local entropy")

        entropy_map = local_entropy_map(
            image_array,
            kernel_size=self.kernel_size,
            bins=self.bins,
        )
        pbar.update(1)

        mask_coords = np.where(mask_array > 0)
        entropy_values = entropy_map[mask_coords]
        pbar.update(1)

        return entropy_values
        
    def extract_features(self, image_data: Union[str, sitk.Image],
                         mask_data: Union[str, sitk.Image],
                         **kwargs) -> pd.DataFrame:
        """
        Extract local entropy features from image within mask region
        
        Args:
            image_data: Path to image file or SimpleITK image object
            mask_data: Path to mask file or SimpleITK mask object
            **kwargs: Additional parameters:
                - kernel_size: Size of the local neighborhood kernel
                - bins: Number of bins for histogram calculation
            
        Returns:
            pd.DataFrame: DataFrame with local entropy values for each voxel in the mask
        """
            
        # Load image
        if isinstance(image_data, str):
            if os.path.exists(image_data):
                image = sitk.ReadImage(image_data)
            else:
                raise FileNotFoundError(f"Image file not found: {image_data}")
        else:
            image = image_data
            
        # Load mask
        if isinstance(mask_data, str):
            if os.path.exists(mask_data):
                mask = sitk.ReadImage(mask_data)
            else:
                raise FileNotFoundError(f"Mask file not found: {mask_data}")
        else:
            mask = mask_data
            
        # Convert to numpy arrays
        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)
        
        # Check if mask has non-zero values
        if np.sum(mask_array > 0) == 0:
            raise ValueError("Mask has no non-zero values, cannot extract features")
        
        try:
            # Calculate local entropy for each voxel in the mask
            entropy_values = self._calculate_entropy(image_array, mask_array)
            
            # Create DataFrame
            feature_df = pd.DataFrame({
                self.feature_names: entropy_values
            })
            
            return feature_df
            
        except Exception as e:
            logging.error(f"Failed to extract local entropy features: {str(e)}")
            raise
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names
        
        Returns:
            List[str]: List of feature names
        """
        return self.feature_names 