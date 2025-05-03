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
from .base_feature_extractor import BaseFeatureExtractor, register_feature_extractor

@register_feature_extractor('voxel_radiomics')
class VoxelRadiomicsExtractor(BaseFeatureExtractor):
    """
    Extract voxel-level radiomics features from image within mask region
    using PyRadiomics' voxel-based extraction
    """
    
    def __init__(self, params_file: str = None, **kwargs):
        """
        Initialize voxel-level radiomics feature extractor
        
        Args:
            params_file: Path to PyRadiomics parameter file
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
            
        # Load mask
        if isinstance(mask_data, str):
            if os.path.exists(mask_data):
                mask = sitk.ReadImage(mask_data)
            else:
                raise FileNotFoundError(f"Mask file not found: {mask_data}")
        else:
            mask = mask_data
            
        # Check if mask has non-zero values
        mask_array = sitk.GetArrayFromImage(mask)
        if np.sum(mask_array > 0) == 0:
            raise ValueError("Mask has no non-zero values, cannot extract features")
        
        try:
            # Initialize PyRadiomics feature extractor
            extractor = featureextractor.RadiomicsFeatureExtractor(self.params_file)
            kernelRadius = kwargs.get('kernelRadius', 1)
            extractor.settings.update({'kernelRadius': kernelRadius})
            
            # Extract voxel-based features
            result = extractor.execute(image, mask, voxelBased=True)

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
                    feature_names.append(key)
                    feature_array = sitk.GetArrayFromImage(val)
                    # Extract values none zero voxels
                    values = feature_array[feature_array > 0]
                    feature_matrix.append(values)
            
                # # 创建一个空image
                # empty_image = sitk.Image(image.GetSize(), sitk.sitkFloat64)
                # # 设置spacing
                # empty_image.SetSpacing(image.GetSpacing())
                # # 设置origin
                # empty_image.SetOrigin(image.GetOrigin())

                # # 设置array
                # empty_image_array = sitk.GetArrayFromImage(empty_image)
                # # 将values的值填充到empty_image中
                # empty_image_array[mask_array > 0] = values
                # # 将empty_image_array转换为sitk.Image
                # empty_image = sitk.GetImageFromArray(empty_image_array)
                # # 保存empty_image
                # sitk.WriteImage(empty_image, "empty_image.nii.gz")
            
            
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

