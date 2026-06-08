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
"""
Custom Feature Extractor Template

Usage Instructions:
1. Copy this file and rename it to your_method_feature_extractor.py
2. Change the class name CustomFeatureExtractorTemplate to your feature extractor name
3. Modify the name in the register_feature_extractor decorator to your method's abbreviation
4. Implement the extract_features method and set the feature_names attribute
5. No need to modify __init__.py, the system will automatically discover and register your feature extractor

注意：必须在__init__中初始化feature_names属性，以确保get_feature_names方法可以被调用
而不依赖于extract_features方法的执行。
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from .base_extractor import BaseClusteringExtractor, register_feature_extractor


@register_feature_extractor('custom_template')  # Register feature extractor (please change to your method name)
class CustomFeatureExtractorTemplate(BaseClusteringExtractor):
    """
    Custom Feature Extractor Template Class - Please replace with your feature extractor description
    """
    
    def __init__(self, normalize: bool = False, image_names: Optional[List[str]] = None, **kwargs: Any) -> None:
        """
        Initialize the feature extractor
        
        Args:
            normalize: Whether to normalize features
            image_names: Optional list of image names to use as feature names
            **kwargs: Other parameters that will be passed to the parent class
        """
        super().__init__(**kwargs)
        self.normalize = normalize
        
        # 强制设置feature_names
        self.feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4']

    
    def extract_features(self, image_data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Extract features from image data
        
        Args:
            image_data: Input image data with shape [n_voxels, n_timepoints] or other format
            **kwargs: Additional parameters such as subject, mask, etc.
            
        Returns:
            np.ndarray: Extracted features with shape [n_voxels, n_features]
        """
        # Implement your feature extraction logic here
        # For example: calculate texture features, shape features, etc.
        
        # Example code for demonstration (please replace with actual implementation)
        n_samples = image_data.shape[0]
        n_features = 3  # Example: extract 3 features
        
        # Create random features as an example (please replace with actual feature calculation)
        features = np.random.random((n_samples, n_features))
        
        # Normalize if needed
        if self.normalize:
            for i in range(features.shape[1]):
                column = features[:, i]
                min_val = np.min(column)
                max_val = np.max(column)
                if max_val > min_val:
                    features[:, i] = (column - min_val) / (max_val - min_val)
        
        
        return features 