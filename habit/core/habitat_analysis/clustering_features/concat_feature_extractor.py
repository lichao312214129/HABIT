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
Multi-image feature extractor that concatenates features from multiple images
"""

import numpy as np
import pandas as pd
from typing import List, Set, Union

from .base_extractor import BaseClusteringExtractor, FeatureExtractorRegistry
from .method_param_spec import MethodParamSpec


@FeatureExtractorRegistry.register('concat')
class ConcatImageFeatureExtractor(BaseClusteringExtractor):
    """
    Feature extractor that concatenates features from multiple images
    
    This extractor works with multiple images as input and concatenates their features
    horizontally to create a combined feature matrix.
    """

    # DSL contract: concat(<sub-expr>, ...) — combiner, no own parameters.
    method_param_spec = MethodParamSpec(
        required=(),
        optional={},
        combiner=True,
        takes_image=False,
    )

    def __init__(self, feature_extractor_name: str = 'raw', **kwargs) -> None:
        """
        Initialize the multi-image feature extractor
        
        Args:
            feature_extractor_name (str): Name of feature extractor to use for each image
            **kwargs: Parameters that will be passed to the parent class
        """
        super().__init__(**kwargs)
        self.feature_extractor_name = feature_extractor_name
        self.feature_names = []

    def extract_features(self, image_data: List[Union[np.ndarray, pd.DataFrame]], **kwargs) -> pd.DataFrame:
        """
        Extract features from multiple images and concatenate them
        
        Args:
            image_data: List of image data, each element can be a numpy array [n_voxels, n_features] 
                       or pandas DataFrame
            **kwargs: Additional parameters for feature extraction
            
        Returns:
            pd.DataFrame: Concatenated features from all images
        """
        if not image_data:
            raise ValueError("At least one image is required")

        # Build per-modality frames first, then a single horizontal concat. Repeated
        # pd.concat in a loop creates O(n_modality) full-width temporary DataFrames;
        # batching cuts peak memory when many modalities are merged.
        frames: List[pd.DataFrame] = []
        columns_so_far: Set[str] = set()

        for i, values in enumerate(image_data):
            key = f"img{i+1}"

            if isinstance(values, pd.DataFrame):
                # Drop columns that already appeared on a previous modality (e.g. supervoxel_id)
                # so the final table has one copy, matching incremental concat semantics.
                if columns_so_far:
                    duplicate_cols = [
                        col for col in values.columns if col in columns_so_far
                    ]
                    if duplicate_cols:
                        values = values.drop(columns=duplicate_cols)
                columns_so_far.update(values.columns)
                frames.append(values)
            else:
                if len(values.shape) == 1:
                    col_name = f"feature_{key}"
                    df = pd.DataFrame(values, columns=[col_name])
                else:
                    n_cols = values.shape[1]
                    col_names = [f"feature{j+1}_{key}" for j in range(n_cols)]
                    df = pd.DataFrame(values, columns=col_names)
                columns_so_far.update(df.columns)
                frames.append(df)

        image_df = pd.concat(frames, axis=1) if frames else pd.DataFrame()

        # Check if all images have the same number of voxels
        if image_df.empty:
            raise ValueError("No valid data found in input list")
            
        # Ensure all feature columns (except supervoxel_id) are numeric type.
        # pd.concat may convert numeric columns to object type in some cases.
        
        for col in image_df.columns:
            if col != 'supervoxel_id':
                image_df[col] = pd.to_numeric(image_df[col], errors='coerce')
        
        # Update feature names
        self.feature_names = list(image_df.columns)
        
        return image_df
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names
        
        Returns:
            List[str]: List of feature names
        """
        return self.feature_names 