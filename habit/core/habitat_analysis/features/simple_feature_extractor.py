"""
Simple feature extractor that uses raw image intensities
"""

import numpy as np
from typing import List, Any, Optional
from habitat_clustering.features.base_feature_extractor import BaseFeatureExtractor, register_feature_extractor


@register_feature_extractor('simple')
class SimpleFeatureExtractor(BaseFeatureExtractor):
    """
    Simple feature extractor that directly uses image intensities as features
    """
    
    def __init__(self, normalize: bool = False, image_names: Optional[List[str]] = None, **kwargs: Any) -> None:
        """
        Initialize the simple feature extractor
        
        Args:
            normalize: Whether to normalize features
            image_names: Optional list of image names to use as feature names
            **kwargs: Other parameters that will be passed to the parent class
        """
        super().__init__(**kwargs)
        self.normalize = normalize
        
        self.feature_names = ["intensity_precontrast", "intensity_lap", "intensity_pvp", "intensity_delay_3min"]
    
    def extract_features(self, image_data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Extract features from image data (directly using intensity values)
        
        Args:
            image_data: 2D array with shape [n_voxels, n_timepoints] or DataFrame
            **kwargs: Additional parameters (not used by this extractor)
            
        Returns:
            np.ndarray: Extracted features with the same shape as input (rows are voxels, columns are features)
        """
        # 处理不同类型的输入
        if hasattr(image_data, 'columns') and hasattr(image_data, 'values'):
            # 如果输入是DataFrame
            features = image_data.values.copy()
        else:
            # 如果输入是numpy数组
            features = image_data.copy()
        
        # Normalize features (optional)
        if self.normalize:
            for i in range(features.shape[1]):
                column = features[:, i]
                min_val = np.min(column)
                max_val = np.max(column)
                if max_val > min_val:
                    features[:, i] = (column - min_val) / (max_val - min_val)
        
        return features