"""
Raw feature extractor that extracts raw image intensities from ROI
"""

import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
from typing import Union, List
from .base_feature_extractor import BaseFeatureExtractor, register_feature_extractor, \
    register_feature_extractor


@register_feature_extractor('raw')
class RawFeatureExtractor(BaseFeatureExtractor):
    """
    Simple feature extractor that directly extracts image intensities from ROI

    This extractor works with:
    1. Image paths and mask paths
    2. SimpleITK image objects and mask objects
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the raw feature extractor

        Args:
            **kwargs: Parameters that will be passed to the parent class
        """
        super().__init__(**kwargs)

    def extract_features(self, image_data: Union[str, sitk.Image],
                         mask_data: Union[str, sitk.Image],
                         **_) -> np.ndarray:
        """
        Extract raw voxel values from image within mask region

        Args:
            image_data: Path to image file or SimpleITK image object
            mask_data: Path to mask file or SimpleITK mask object

        Returns:
            pd.DataFrame: Extracted voxel values
        """
        # 加载图像
        if isinstance(image_data, str):
            if os.path.exists(image_data):
                image = sitk.ReadImage(image_data)
            else:
                raise FileNotFoundError(f"Image file not found: {image_data}")
        else:
            image = image_data

        # 加载掩码
        if isinstance(mask_data, str):
            if os.path.exists(mask_data):
                mask = sitk.ReadImage(mask_data)
            else:
                raise FileNotFoundError(f"Mask file not found: {mask_data}")
        else:
            mask = mask_data

        # 转换为numpy数组
        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)

        # 提取掩码内的值
        roi_mask = mask_array > 0
        if np.sum(roi_mask) == 0:
            raise ValueError("Mask has no non-zero values, cannot extract features")

        # 提取掩码内的原始值
        voxel_values = image_array[roi_mask]

        # 如果是单通道图像，将其转换为2D数组 [n_voxels, 1]
        if len(voxel_values.shape) == 1:
            voxel_values = voxel_values.reshape(-1, 1)

        # to df
        voxel_values = pd.DataFrame(voxel_values, columns=['raw_intensity'])

        self.feature_names = ['raw_intensity']  # 设置特征名称为 'raw_intensity'，表示提取的原始强度值
        return voxel_values
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names

        Returns:
            List[str]: List of feature names
        """
        return self.feature_names