"""
Multi-image feature extractor that concatenates features from multiple images
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any

from .base_extractor import BaseClusteringExtractor, register_feature_extractor


@register_feature_extractor('concat')
class ConcatImageFeatureExtractor(BaseClusteringExtractor):
    """
    Feature extractor that concatenates features from multiple images
    
    This extractor works with multiple images as input and concatenates their features
    horizontally to create a combined feature matrix.
    """

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
            
        # Convert image_data list to DataFrame
        image_df = pd.DataFrame()
        for i, values in enumerate(image_data):
            # Generate a key for naming features
            key = f"img{i+1}"
            
            # If values is already a DataFrame, use its column names
            if isinstance(values, pd.DataFrame):
                # Remove duplicate index columns (like SupervoxelID) that already exist
                # in image_df to avoid multiple columns with the same name after concat
                if not image_df.empty:
                    # Find columns that already exist in image_df (e.g., SupervoxelID)
                    duplicate_cols = [col for col in values.columns if col in image_df.columns]
                    if duplicate_cols:
                        values = values.drop(columns=duplicate_cols)
                
                image_df = pd.concat([image_df, values], axis=1)
            else:
                # If values is a numpy array, convert to DataFrame with prefixed column name
                if len(values.shape) == 1:
                    # For 1D arrays, create a single column
                    col_name = f"feature_{key}"
                    df = pd.DataFrame(values, columns=[col_name])
                else:
                    # For 2D arrays, create multiple columns with prefixed names
                    n_cols = values.shape[1]
                    col_names = [f"feature{j+1}_{key}" for j in range(n_cols)]
                    df = pd.DataFrame(values, columns=col_names)
                image_df = pd.concat([image_df, df], axis=1)
        
        # Check if all images have the same number of voxels
        if image_df.empty:
            raise ValueError("No valid data found in input list")
            
        # Ensure all feature columns (except SupervoxelID) are numeric type.
        # pd.concat may convert numeric columns to object type in some cases.
        
        for col in image_df.columns:
            if col != 'SupervoxelID':
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