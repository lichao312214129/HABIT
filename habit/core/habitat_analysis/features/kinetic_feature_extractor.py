"""
Kinetic feature extractor for habitat analysis.
Extracts kinetic features from time series data.
"""

import numpy as np
import pandas as pd
from scipy import stats
from .base_feature_extractor import BaseFeatureExtractor, register_feature_extractor
from typing import Dict, List, Optional, Union, Any, Tuple

from habit.utils.io_utils import load_timestamp

@register_feature_extractor('kinetic')  # Register feature extractor
class KineticFeatureExtractor(BaseFeatureExtractor):
    """
    Kinetic Feature Extractor
    
    Extracts dynamic features based on time-series images, such as enhancement rate, peak enhancement, etc.
    """
    
    def __init__(self, timestamps: Optional[str] = None, **kwargs: Any) -> None:
        """
        Initialize the kinetic feature extractor
        
        Args:
            timestamps (str, optional): Path to timestamp file containing acquisition time for each image
            **kwargs: Other parameters to be passed to the parent class
        """
        super().__init__(timestamps=timestamps, **kwargs)
        self.feature_names = [
            'wash_in_slope',                    # Wash-in rate
            'wash_out_slope_lap_pvp',          # Wash-out rate from arterial phase to portal venous phase
            'wash_out_slope_pvp_dp'            # Wash-out rate from portal venous phase to delayed phase
        ]
        
        # Load timestamp file
        if timestamps:
            try:
                self.time_dict = load_timestamp(timestamps)
            except Exception as e:
                print(f"Warning: Failed to load timestamp file: {str(e)}")
                self.time_dict = None
        else:
            self.time_dict = None
    
    def extract_features(self, image_data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Extract kinetic features
        
        Args:
            image_data (dict): Dictionary of image data with keys as image names and values as 2D arrays [n_voxels, n_features]
            **kwargs: Other parameters such as timestamps, subject, etc.
            
        Returns:
            np.ndarray: Feature matrix with shape [n_voxels, n_features]
        """
        # Get parameters from kwargs
        subject = kwargs.get('subject', None)
        
        # to df
        # Convert image_data dictionary to DataFrame 
        # header =  key_name
        image_df = pd.concat(image_data, axis=1)        
        # Calculate features
        features = self._compute_kinetic_features(image_df, self.time_dict.loc[subject])
        return features
    
    def _compute_kinetic_features(self, image_array: pd.DataFrame, image_timestamp: List[str]) -> np.ndarray:
        """
        Compute kinetic features
        
        Args:
            image_array (pd.DataFrame): DataFrame of voxel intensities over time [n_voxels, n_timepoints]
            image_timestamp (list): List of scan timestamps
            
        Returns:
            np.ndarray: Computed kinetic features
        """
        # Validate input dimensions
        assert np.shape(image_array)[1] == len(image_timestamp), "Number of columns in image array should equal length of timestamp list"
        
        # Small constant to avoid division by zero
        epsilon = 1e-6
        
        # Parse timestamps
        time_format = "%H-%M-%S"
        # pd.to_datetime
        image_timestamp = pd.to_datetime(image_timestamp, format=time_format)
        # Set the first timestamp to 25 seconds before the second scan
        image_timestamp['pre_contrast'] = image_timestamp['LAP'] - pd.Timedelta(seconds=25)
        
        # Calculate time differences
        delta_t1 = (image_timestamp['LAP'] - image_timestamp['pre_contrast']).total_seconds()
        delta_t2 = (image_timestamp['PVP'] - image_timestamp['LAP']).total_seconds()
        delta_t3 = (image_timestamp['delay_3min'] - image_timestamp['PVP']).total_seconds()
        
        # Calculate relative intensity differences with epsilon to avoid division by zero
        lap_precontrast = image_array.loc[:, 'raw-LAP'] - image_array.loc[:, 'raw-pre_contrast']
        pvp_lap = image_array.loc[:, 'raw-PVP'] - image_array.loc[:, 'raw-LAP']
        delay_pvp = image_array.loc[:, 'raw-delay_3min'] - image_array.loc[:, 'raw-PVP']
        
        # Set negative enhancement values to 0
        lap_precontrast[lap_precontrast < 0] = 0
        
        # Calculate kinetic features
        wash_in_slope = lap_precontrast / (delta_t1 + epsilon)
        wash_out_slope_of_lap_and_pvp = pvp_lap / (delta_t2 + epsilon)
        wash_out_slope_of_pvp_and_dp = delay_pvp / (delta_t3 + epsilon)

        # Combine features  
        metrics = np.array([
            wash_in_slope,
            wash_out_slope_of_lap_and_pvp,
            wash_out_slope_of_pvp_and_dp
        ]).T

        # to df
        metrics_df = pd.DataFrame(metrics, columns=self.feature_names)
        
        return metrics_df 
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names
        
        Returns:
            List[str]: List of feature names
        """
        return self.feature_names
