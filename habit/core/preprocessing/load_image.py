from typing import Dict, Any, Optional, Union, List, Tuple, Sequence
import SimpleITK as sitk
import os
import numpy as np
from .base_preprocessor import BasePreprocessor
from .preprocessor_factory import PreprocessorFactory

@PreprocessorFactory.register("load_image")
class LoadImagePreprocessor(BasePreprocessor):
    """Load images from file paths and convert them to SimpleITK Image objects.
    
    This preprocessor takes keys from the subject_data dictionary, loads the corresponding
    files as SimpleITK images, and replaces the file paths with the loaded image objects.
    Keys not specified will remain unchanged.
    """
    
    def __init__(
        self,
        keys: Union[str, List[str]],
        allow_missing_keys: bool = True,
        **kwargs
    ):
        """Initialize the LoadImage preprocessor.
        
        Args:
            keys (Union[str, List[str]]): Keys of the items to load as SimpleITK images.
            allow_missing_keys (bool): If True, allows missing keys in the input data.
            **kwargs: Additional parameters.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        
        # Convert single key to list
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        
    def _load_sitk_image(self, image_path: str) -> sitk.Image:
        """Load a SimpleITK image from a file path.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            sitk.Image: Loaded SimpleITK image
            
        Raises:
            FileNotFoundError: If the image file does not exist
            RuntimeError: If the image cannot be loaded
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} does not exist")
            
        try:
            # Load the image with SimpleITK
            sitk_image = sitk.ReadImage(image_path)
            return sitk_image
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")
        
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load specified keys from the data dictionary as SimpleITK images.
        
        Args:
            data (Dict[str, Any]): Input data dictionary containing file paths.
            
        Returns:
            Dict[str, Any]: Data dictionary with file paths replaced by SimpleITK images.
        """
        self._check_keys(data)
        
        # Process each specified key
        for key in self.keys:
            # Skip if key is missing and we allow missing keys
            if key not in data:
                if self.allow_missing_keys:
                    continue
                else:
                    raise KeyError(f"Key {key} not found in data dictionary")
                
            # Get the file path
            image_path = data[key]
            
            # Skip if not a string (already processed or not a path)
            if not isinstance(image_path, str):
                continue
                
            try:
                # Load the image
                sitk_image = self._load_sitk_image(image_path)
                
                # Replace the file path with the SimpleITK image
                data[key] = sitk_image
                
                # Initialize or update metadata
                meta_key = f"{key}_meta_dict"
                if meta_key not in data:
                    data[meta_key] = {}
                    
                # Store important image properties in metadata
                data[meta_key]["spacing"] = sitk_image.GetSpacing()
                data[meta_key]["size"] = sitk_image.GetSize()
                data[meta_key]["origin"] = sitk_image.GetOrigin()
                data[meta_key]["direction"] = sitk_image.GetDirection()
                data[meta_key]["pixel_type"] = sitk_image.GetPixelIDTypeAsString()
                data[meta_key]["image_path"] = image_path  # Add image path to metadata
                
            except Exception as e:
                print(f"Error loading image for key {key}: {e}")
                if not self.allow_missing_keys:
                    raise
    