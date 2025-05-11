"""
Voxel-level local entropy feature extractor

This module provides a feature extractor that calculates local entropy for each voxel
within a mask region. Local entropy is a measure of the randomness or information content
in the local neighborhood of a voxel.

Example usage:
    ```python
    from habit.core.habitat_analysis.features import LocalEntropyExtractor
    
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
from scipy import ndimage
from typing import Union, List, Dict, Optional, Tuple
from .base_feature_extractor import BaseFeatureExtractor, register_feature_extractor
from habit.utils.progress_utils import CustomTqdm

@register_feature_extractor('local_entropy')
class LocalEntropyExtractor(BaseFeatureExtractor):
    """
    Extract voxel-level local entropy features from image within mask region
    Local entropy is a measure of the randomness in the local neighborhood of a voxel
    """
    
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
        Calculate local entropy for each voxel using scipy.ndimage for faster processing
        
        Args:
            image_array: 3D image array
            mask_array: 3D mask array
            
        Returns:
            np.ndarray: Local entropy values for each voxel in the mask
        """
        # Create progress bar for tracking
        pbar = CustomTqdm(total=4, desc="Calculating local entropy")
        
        # Normalize image to [0, 1] for entropy calculation
        img_min = np.min(image_array)
        img_max = np.max(image_array)
        if img_max > img_min:
            norm_image = (image_array - img_min) / (img_max - img_min)
        else:
            # Handle constant image case
            norm_image = np.zeros_like(image_array)
        pbar.update(1)
        
        # Define kernel size (must be odd)
        if self.kernel_size % 2 == 0:
            kernel_size = self.kernel_size + 1
        else:
            kernel_size = self.kernel_size
            
        # Create a footprint for the local neighborhood
        footprint = np.ones((kernel_size, kernel_size, kernel_size))
        
        # Calculate entropy for the entire image using scipy.ndimage
        # First discretize the image into bins
        binned_image = np.round(norm_image * (self.bins - 1)).astype(int)
        pbar.update(1)
        
        # Initialize entropy map
        entropy_map = np.zeros_like(norm_image, dtype=float)
        
        # For each possible value in the binned image
        for i in range(self.bins):
            # Create a binary map for this bin value
            bin_map = (binned_image == i).astype(float)
            
            # Count occurrences of this value in each neighborhood
            count_map = ndimage.convolve(bin_map, footprint, mode='constant', cval=0.0)
            
            # Calculate probability (avoid division by zero)
            total_voxels = kernel_size**3  # Total voxels in the neighborhood
            prob_map = count_map / total_voxels
            
            # Update entropy map (handling zeros to avoid log(0))
            with np.errstate(divide='ignore', invalid='ignore'):
                entropy_update = -prob_map * np.log2(prob_map)
                entropy_update[~np.isfinite(entropy_update)] = 0
                entropy_map += entropy_update
        
        pbar.update(1)
        
        # Extract entropy values for mask voxels
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