"""
Simple feature extractor that uses raw image intensities
"""

import numpy as np
import pandas as pd
from typing import List, Any, Optional, Union
from .base_extractor import BaseClusteringExtractor, register_feature_extractor


@register_feature_extractor('my_feature_extractor')
class MyFeatureExtractor(BaseClusteringExtractor):
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
        
        self.feature_names = ["intensity_precontrast", "intensity_lap", "intensity_pvp", "intensity_delay_3min", "delta"]
    
    def extract_features(self, image_data: Union[np.ndarray, pd.DataFrame, List[Union[np.ndarray, pd.DataFrame]]], **kwargs: Any) -> pd.DataFrame:
        """
        Extract features from image data including delta calculation
        
        Args:
            image_data: DataFrame/array with image data, or a list of them for multi-image extraction
            **kwargs: Additional parameters (not used by this extractor)
            
        Returns:
            pd.DataFrame: Extracted features with voxels as rows and features as columns
        """
        # Handle list input (multi-image case)
        if isinstance(image_data, list):
            # For this simple example, we just concatenate them
            # In a real case, you might do complex cross-image calculations here
            df = pd.concat([pd.DataFrame(img) if not isinstance(img, pd.DataFrame) else img for img in image_data], axis=1)
            
            # If we concatenated multiple images, we might need to assign column names
            # For this demo, we'll try to map to our expected 4 images
            if df.shape[1] >= 4:
                df.columns = ['pre_contrast', 'LAP', 'PVP', 'delay_3min'] + list(df.columns[4:])
        
        # Ensure image_data is a DataFrame if it's not a list
        elif not isinstance(image_data, pd.DataFrame):
            # If input is a numpy array, convert to DataFrame
            if len(self.feature_names) - 1 == image_data.shape[1]:  # -1 for the delta feature we'll calculate
                column_names = self.feature_names[:-1]  # Exclude 'delta' from column names
                df = pd.DataFrame(image_data, columns=column_names)
            else:
                # Use default column names
                df = pd.DataFrame(image_data)
                # Assume the DataFrame has columns in the order: pre_contrast, LAP, PVP, delay_3min
                if df.shape[1] >= 4:
                    df.columns = ['pre_contrast', 'LAP', 'PVP', 'delay_3min'] + list(df.columns[4:])
        else:
            # If already a DataFrame, use as is
            df = image_data.copy()
        
        # Check if required columns exist for delta calculation
        if 'LAP' in df.columns and 'pre_contrast' in df.columns:
            # Calculate delta (enhancement ratio)
            df['delta'] = (df['LAP'] - df['pre_contrast']) / df['pre_contrast'].replace(0, 1e-6)
        else:
            # If columns don't exist, add a placeholder delta column
            df['delta'] = np.nan
            
        # Update feature names if necessary
        self.feature_names = list(df.columns)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names
        
        Returns:
            List[str]: List of feature names
        """
        return self.feature_names