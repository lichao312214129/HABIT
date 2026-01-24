"""
Mean voxel features extractor for supervoxels
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from typing import Union, List, Dict, Optional, Any

from .base_extractor import BaseClusteringExtractor, register_feature_extractor


@register_feature_extractor('mean_voxel_features')
class MeanVoxelFeaturesExtractor(BaseClusteringExtractor):
    """
    Extract mean voxel features for each supervoxel in the supervoxel map
    
    This extractor calculates the mean values of voxel features within each supervoxel.
    It's a simpler alternative to radiomics features, just using the average values.
    """
    
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize mean voxel features extractor
        
        Args:
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.feature_names = []
        
    def extract_features(self, image_data: Union[str, sitk.Image],
                         supervoxel_map: Union[str, sitk.Image],
                         **kwargs: Any) -> pd.DataFrame:
        """
        Extract mean voxel features for each supervoxel
        
        Args:
            image_data: Path to image file, SimpleITK image object, or DataFrame of voxel features
            supervoxel_map: Path to supervoxel map file or SimpleITK image object
            **kwargs: Additional parameters
            
        Returns:
            pd.DataFrame: DataFrame with mean voxel features for each supervoxel
        """
        # Load supervoxel map
        if isinstance(supervoxel_map, str):
            if os.path.exists(supervoxel_map):
                sv_map = sitk.ReadImage(supervoxel_map)
            else:
                raise FileNotFoundError(f"Supervoxel map file not found: {supervoxel_map}")
        else:
            sv_map = supervoxel_map
            
        # Get supervoxel map as numpy array
        sv_array = sitk.GetArrayFromImage(sv_map)
        
        # Handle different types of image_data
        if isinstance(image_data, pd.DataFrame):
            # If image_data is already a DataFrame of voxel features, use it directly
            voxel_features_df = image_data
        else:
            # Otherwise, load the image and extract intensity values
            if isinstance(image_data, str):
                if os.path.exists(image_data):
                    image = sitk.ReadImage(image_data)
                else:
                    raise FileNotFoundError(f"Image file not found: {image_data}")
            else:
                image = image_data
                
            # Get image array
            image_array = sitk.GetArrayFromImage(image)
            
            # Create DataFrame with intensity values
            # First, flatten arrays
            sv_flat = sv_array.flatten()
            img_flat = image_array.flatten()
            
            # Create DataFrame with supervoxel labels and intensity values
            voxel_features_df = pd.DataFrame({
                'supervoxel': sv_flat,
                'intensity': img_flat
            })
            
            # Remove background voxels (supervoxel label = 0)
            voxel_features_df = voxel_features_df[voxel_features_df['supervoxel'] > 0]
            
            # Set feature names
            self.feature_names = ['intensity']
            
        # Calculate mean features for each supervoxel
        # Find unique supervoxel labels
        sv_labels = np.unique(sv_array)
        sv_labels = sv_labels[sv_labels > 0]  # Exclude background (label 0)
        
        if len(sv_labels) == 0:
            raise ValueError("Supervoxel map has no non-zero values, cannot extract features")
        
        # Initialize feature data storage
        feature_data = []
        
        # For each supervoxel, calculate mean of features
        for sv_label in sv_labels:
            # If input is DataFrame, get rows for current supervoxel
            if isinstance(image_data, pd.DataFrame):
                # Get feature column names (exclude 'supervoxel' column if present)
                feature_cols = [col for col in voxel_features_df.columns if col != 'supervoxel']
                
                # Update feature names if not already set
                if not self.feature_names:
                    self.feature_names = feature_cols
                
                # Get voxels for current supervoxel
                sv_voxels = voxel_features_df[voxel_features_df['supervoxel'] == sv_label]
                
                if len(sv_voxels) == 0:
                    continue
                
                # Calculate mean of each feature for current supervoxel
                feature_row = {"SupervoxelID": int(sv_label)}
                for col in feature_cols:
                    feature_row[col] = sv_voxels[col].mean()
                
                feature_data.append(feature_row)
            else:
                # For image intensity data
                # Get indices for current supervoxel
                indices = sv_array == sv_label
                
                if np.sum(indices) == 0:
                    continue
                
                # Calculate mean intensity
                mean_intensity = np.mean(image_array[indices])
                
                # Create feature row
                feature_row = {
                    "SupervoxelID": int(sv_label),
                    "intensity": mean_intensity
                }
                
                feature_data.append(feature_row)
        
        # Create DataFrame from feature data
        feature_df = pd.DataFrame(feature_data)
        
        return feature_df
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names
        
        Returns:
            List[str]: List of feature names
        """
        return self.feature_names 