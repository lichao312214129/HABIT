"""
Supervoxel-level radiomics feature extractor
"""

import os
import yaml
import logging
import numpy as np
import pandas as pd
import SimpleITK as sitk
from typing import Union, List, Dict, Optional, Tuple
from radiomics import featureextractor
from .base_extractor import BaseClusteringExtractor, register_feature_extractor


@register_feature_extractor('supervoxel_radiomics')
class SupervoxelRadiomicsExtractor(BaseClusteringExtractor):
    """
    Extract radiomics features for each supervoxel in the supervoxel map
    """
    
    def __init__(self, params_file: str = None, **kwargs):
        """
        Initialize supervoxel radiomics feature extractor
        
        Args:
            params_file: Path to PyRadiomics parameter file or YAML string containing parameters
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.params_file = params_file
        self.feature_names = []
        
        # 打印调试信息
        print(f"SupervoxelRadiomicsExtractor initialized with params_file: {params_file}")
        if params_file and os.path.exists(params_file):
            print(f"Parameter file exists at: {os.path.abspath(params_file)}")
        elif params_file:
            print(f"Warning: Parameter file does not exist at: {os.path.abspath(params_file) if params_file else 'None'}")
        
    def extract_features(self, image_data: Union[str, sitk.Image],
                        supervoxel_map: Union[str, sitk.Image],
                        config_file: Optional[str] = None,
                        **kwargs) -> pd.DataFrame:
        """
        Extract radiomics features for each supervoxel in the supervoxel map
        
        Args:
            image_data: Path to image file or SimpleITK image object
            supervoxel_map: Path to supervoxel map file or SimpleITK image object
            config_file: Path to PyRadiomics parameter file (overrides the one in constructor)
            **kwargs: Additional parameters
            
        Returns:
            pd.DataFrame: DataFrame with radiomics features for each supervoxel
        """
        # Use config_file if provided, otherwise use the one from constructor
        params_file = config_file or self.params_file
        
        # Get image name from kwargs or extract from path
        img_name = kwargs.get('image', '')
        if not img_name and isinstance(image_data, str):
            img_name = os.path.basename(os.path.dirname(image_data))
        
        # 打印调试信息
        print(f"SupervoxelRadiomicsExtractor.extract_features called with:")
        print(f"  - params_file: {params_file}")
        print(f"  - img_name: {img_name}")
        if isinstance(image_data, str):
            print(f"  - image_data: {image_data}")
        else:
            print(f"  - image_data: <SimpleITK.Image>")
        if isinstance(supervoxel_map, str):
            print(f"  - supervoxel_map: {supervoxel_map}")
        else:
            print(f"  - supervoxel_map: <SimpleITK.Image>")
        print(f"  - Additional kwargs: {kwargs}")
        
        # Load image
        if isinstance(image_data, str):
            if os.path.exists(image_data):
                image = sitk.ReadImage(image_data)
                print(f"Successfully loaded image from: {image_data}")
            else:
                error_msg = f"Image file not found: {image_data}"
                print(f"Error: {error_msg}")
                raise FileNotFoundError(error_msg)
        else:
            image = image_data
            
        # Load supervoxel map
        if isinstance(supervoxel_map, str):
            if os.path.exists(supervoxel_map):
                sv_map = sitk.ReadImage(supervoxel_map)
                print(f"Successfully loaded supervoxel map from: {supervoxel_map}")
            else:
                error_msg = f"Supervoxel map file not found: {supervoxel_map}"
                print(f"Error: {error_msg}")
                raise FileNotFoundError(error_msg)
        else:
            sv_map = supervoxel_map
            
        # Initialize PyRadiomics feature extractor with parameters
        if params_file:
            if os.path.exists(params_file):
                # Load parameters from file
                print(f"Loading parameters from file: {params_file}")
                try:
                    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
                    print(f"Successfully initialized radiomics extractor with parameters from: {params_file}")
                except Exception as e:
                    error_msg = f"Failed to initialize radiomics extractor with file {params_file}: {str(e)}"
                    print(f"Error: {error_msg}")
                    raise ValueError(error_msg)
            else:
                # Try to parse as YAML string
                print(f"Parameter file not found: {params_file}, trying to parse as YAML string")
                try:
                    params = yaml.safe_load(params_file)
                    extractor = featureextractor.RadiomicsFeatureExtractor(params)
                    print("Successfully initialized radiomics extractor with YAML string parameters")
                except Exception as e:
                    error_msg = f"Invalid parameter file or YAML string: {e}"
                    print(f"Error: {error_msg}")
                    raise ValueError(error_msg)
        else:
            # Use default parameters
            print("No parameter file provided, using default radiomics settings")
            extractor = featureextractor.RadiomicsFeatureExtractor()
            
        # Get supervoxel map array
        sv_array = sitk.GetArrayFromImage(sv_map)
        
        # Find unique supervoxel labels (excluding 0 background)
        sv_labels = np.unique(sv_array)
        sv_labels = sv_labels[sv_labels > 0]
        
        if len(sv_labels) == 0:
            raise ValueError("Supervoxel map has no non-zero values, cannot extract features")
            
        # Initialize feature data storage
        feature_data = []
        # Reset feature names to track the new names
        self.feature_names = []
        
        # Extract features for each supervoxel
        for sv_idx, sv_label in enumerate(sv_labels):
            # Create a mask for the current supervoxel
            sv_mask = np.zeros_like(sv_array)
            sv_mask[sv_array == sv_label] = 1
            
            # Convert to SimpleITK image
            sv_mask_img = sitk.GetImageFromArray(sv_mask)
            sv_mask_img.CopyInformation(image)
            
            try:
                # Extract features for this supervoxel
                features = extractor.execute(image, sv_mask_img, label=1)
                
                # Filter out diagnostic features and prepare for storage
                feature_row = {"SupervoxelID": int(sv_label)}
                
                for feature_name, feature_value in features.items():
                    if not feature_name.startswith('diagnostics_'):
                        # Add image name to feature name
                        new_feature_name = f"{feature_name}-{img_name}" if img_name else feature_name
                        
                        # Add to feature names list on first iteration
                        if sv_idx == 0 and new_feature_name not in self.feature_names:
                            self.feature_names.append(new_feature_name)
                        
                        feature_row[new_feature_name] = feature_value
                
                # Add row to data
                feature_data.append(feature_row)
                
            except Exception as e:
                logging.warning(f"Failed to extract features for supervoxel {sv_label}: {str(e)}")
                # We'll add a row with NaN values for this supervoxel
                if self.feature_names:
                    feature_row = {"SupervoxelID": int(sv_label)}
                    for name in self.feature_names:
                        feature_row[name] = np.nan
                    feature_data.append(feature_row)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(feature_data)
        
        return feature_df
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names
        
        Returns:
            List[str]: List of feature names
        """
        return self.feature_names 