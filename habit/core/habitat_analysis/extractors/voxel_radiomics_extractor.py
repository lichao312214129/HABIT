"""
Voxel-level radiomics feature extractor
"""

import os
import logging
import numpy as np
import pandas as pd
import SimpleITK as sitk
import six
from typing import Union, List, Dict, Optional, Tuple
from radiomics import featureextractor
from .base_extractor import BaseClusteringExtractor, register_feature_extractor

@register_feature_extractor('voxel_radiomics')
class VoxelRadiomicsExtractor(BaseClusteringExtractor):
    """
    Extract voxel-level radiomics features from image within mask region
    using PyRadiomics' voxel-based extraction
    """
    
    def __init__(self, **kwargs):
        """
        Initialize voxel-level radiomics feature extractor
        
        Args:
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        # self.params_file是kwargs中的path
        # 用os判断,如果没有file则报错
        for key, value in kwargs.items():
            if os.path.exists(value):
                self.params_file = value
                break
        if self.params_file is None:
            raise ValueError("params_file not found in kwargs")
        
    def extract_features(self, image_data: Union[str, sitk.Image],
                         mask_data: Union[str, sitk.Image],
                         **kwargs) -> pd.DataFrame:
        """
        Extract voxel-level radiomics features from image within mask region
        
        Args:
            image_data: Path to image file or SimpleITK image object
            mask_data: Path to mask file or SimpleITK mask object
            **kwargs: Additional parameters
                subj: subject name
                img_name: Name of the image to append to feature names
            
        Returns:
            pd.DataFrame: Extracted voxel-level radiomics features
        """
        # Load image
        if isinstance(image_data, str):
            if os.path.exists(image_data):
                image = sitk.ReadImage(image_data)
            else:
                raise FileNotFoundError(f"Image file not found: {image_data}")
        else:
            image = image_data

        # Get image name
        image_name = kwargs.get('image', None)
        if image_name is None:
            image_name = os.path.basename(os.path.dirname(image_data))
            
        # Load mask
        if isinstance(mask_data, str):
            if os.path.exists(mask_data):
                mask = sitk.ReadImage(mask_data)
            else:
                raise FileNotFoundError(f"Mask file not found: {mask_data}")
        else:
            mask = mask_data

        # Ensure mask has the same geometric information as image
        # to avoid geometry mismatch errors in PyRadiomics
        mask.CopyInformation(image)

        # Check if mask has non-zero values
        mask_array = sitk.GetArrayFromImage(mask)
        if np.sum(mask_array > 0) == 0:
            raise ValueError("Mask has no non-zero values, cannot extract features")
        
        try:
            # Initialize PyRadiomics feature extractor
            extractor = featureextractor.RadiomicsFeatureExtractor(self.params_file)
            # kernelRadius controls the size of the local neighborhood (in voxels) 
            # used for voxel-based feature extraction. A radius of 1 means a 3×3×3 cube
            # centered on each voxel, radius of 2 means 5×5×5, etc.
            kernelRadius = kwargs.get('kernelRadius', 1)
            extractor.settings.update({
                'kernelRadius': kernelRadius,
                'geometryTolerance': 1e-3  # Allow small geometric differences
            })
            
            # Extract voxel-based features  计算GLCM时会报错，可能是由于局部太均质所致
            result = extractor.execute(image, mask, voxelBased=True)

            # # mask中多少不为0的voxels
            # num_non_zero_voxels = np.sum(mask_array > 0)
            # logging.info(f"Mask has {num_non_zero_voxels} non-zero voxels")

            # Filter out diagnostic features
            result = {k: v for k, v in result.items() if not k.startswith('diagnostic')}
            
            # Get mask coordinates and prepare data structure
            # Get 3D coordinates (z,y,x) of non-zero voxels in the mask
            coords = list(zip(*np.where(mask_array > 0)))
            num_voxels = len(coords)

            # Process each feature map and organize into DataFrame
            feature_names = []
            feature_matrix = []
            
            for key, val in six.iteritems(result):
                if isinstance(val, sitk.Image):  # Feature map
                    # Append image name to feature name
                    feature_name = f"{key}-{image_name}" if image_name else key
                    feature_names.append(feature_name)
                    feature_array = sitk.GetArrayFromImage(val)
                    # Extract values none zero voxels
                    values = feature_array[feature_array > 0]
                    feature_matrix.append(values)
        
            # Store feature names
            self.feature_names = feature_names
            
            # Create DataFrame with voxels as rows and features as columns
            feature_df = pd.DataFrame(feature_matrix)
            feature_df = feature_df.T
            feature_df.columns = feature_names
            
            return feature_df
            
        except Exception as e:
            logging.error(f"Failed to extract voxel-based features: {str(e)}")
            raise
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names
        
        Returns:
            List[str]: List of feature names
        """
        return self.feature_names

