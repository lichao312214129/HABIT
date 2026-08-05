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
Custom Feature Extractor Template

Usage Instructions:
1. Copy this file and rename it to your_method_feature_extractor.py
2. Change the class name CustomFeatureExtractorTemplate to your feature extractor name
3. Modify the name in the FeatureExtractorRegistry.register decorator to your method's abbreviation
4. Implement the extract_features method and set the feature_names attribute
5. No need to modify __init__.py, the system will automatically discover and register your feature extractor

注意：必须在__init__中初始化feature_names属性，以确保get_feature_names方法可以被调用
而不依赖于extract_features方法的执行。
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from .base_extractor import BaseClusteringExtractor, FeatureExtractorRegistry


@FeatureExtractorRegistry.register('custom_template')  # Register feature extractor (please change to your method name)
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