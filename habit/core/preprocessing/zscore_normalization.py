from typing import Dict, Any, Optional, Union, List, Tuple
import SimpleITK as sitk
import numpy as np
from .base_preprocessor import BasePreprocessor
from .preprocessor_factory import PreprocessorFactory

@PreprocessorFactory.register("zscore_normalization")
class ZScoreNormalization(BasePreprocessor):
    """Apply Z-score normalization to medical images.
    
    This preprocessor normalizes image intensities by subtracting the mean and 
    dividing by the standard deviation, resulting in a distribution with
    zero mean and unit variance.
    """
    
    def __init__(
        self,
        keys: Union[str, List[str]],
        mask_keys: Optional[Union[str, List[str]]] = None,
        clip_values: Optional[Tuple[float, float]] = None,
        allow_missing_keys: bool = False,
        **kwargs
    ):
        """Initialize the Z-score normalization preprocessor.
        
        Args:
            keys (Union[str, List[str]]): Keys of the images to be normalized.
            mask_keys (Optional[Union[str, List[str]]]): Optional keys of masks to use for computing 
                normalization parameters (only calculate statistics within the mask).
            clip_values (Optional[Tuple[float, float]]): Optional tuple of (min, max) values to clip 
                normalized results. Useful to prevent extreme values, e.g. (-3, 3).
            allow_missing_keys (bool): If True, allows missing keys in the input data.
            **kwargs: Additional parameters.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        
        # Convert single key to list
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        
        # Handle mask keys
        if mask_keys is None:
            self.mask_keys = None
        else:
            self.mask_keys = [mask_keys] if isinstance(mask_keys, str) else mask_keys
            
        # Clipping values
        self.clip_values = clip_values
    
    def _apply_zscore_normalization(
        self, 
        sitk_image: sitk.Image,
        sitk_mask: Optional[sitk.Image] = None
    ) -> sitk.Image:
        """Apply Z-score normalization to a SimpleITK image.
        
        Args:
            sitk_image (sitk.Image): Input SimpleITK image to normalize
            sitk_mask (Optional[sitk.Image]): Optional mask for computing statistics
            
        Returns:
            sitk.Image: Normalized SimpleITK image
        """
        # Convert to numpy array for easier computation
        image_array = sitk.GetArrayFromImage(sitk_image)
        
        # If a mask is provided, use it to calculate statistics
        if sitk_mask is not None:
            mask_array = sitk.GetArrayFromImage(sitk_mask)
            mean_val = np.mean(image_array[mask_array > 0])
            std_val = np.std(image_array[mask_array > 0])
        else:
            # Otherwise use the whole image
            mean_val = np.mean(image_array)
            std_val = np.std(image_array)
        
        # Avoid division by zero
        if std_val == 0:
            print(f"Warning: Standard deviation is zero. Using std=1 to avoid division by zero.")
            std_val = 1.0
            
        # Apply z-score normalization
        normalized_array = (image_array - mean_val) / std_val
        
        # Clip values if specified
        if self.clip_values is not None:
            normalized_array = np.clip(normalized_array, self.clip_values[0], self.clip_values[1])
        
        # Convert back to SimpleITK image
        normalized_image = sitk.GetImageFromArray(normalized_array)
        normalized_image.CopyInformation(sitk_image)  # Copy metadata
        
        return normalized_image
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Z-score normalization to the specified images.
        
        Args:
            data (Dict[str, Any]): Input data dictionary containing SimpleITK image objects.
            
        Returns:
            Dict[str, Any]: Data dictionary with normalized images.
        """
        print("Applying Z-score normalization...")
        self._check_keys(data)
        
        # Process each image
        for key in self.keys:
            # Get SimpleITK image from data
            sitk_image = data[key]
            
            # Ensure we have a SimpleITK image object
            if not isinstance(sitk_image, sitk.Image):
                print(f"Warning: {key} is not a SimpleITK Image object. Skipping.")
                continue
            
            # Get corresponding mask if specified
            sitk_mask = None
            if self.mask_keys is not None:
                for mask_key in self.mask_keys:
                    if mask_key in data:
                        sitk_mask = data[mask_key]
                        if not isinstance(sitk_mask, sitk.Image):
                            print(f"Warning: {mask_key} is not a SimpleITK Image object. Using no mask.")
                            sitk_mask = None
                        else:
                            break
            
            try:
                # Apply Z-score normalization
                normalized_image = self._apply_zscore_normalization(sitk_image, sitk_mask)
                
                # Store the normalized image
                data[key] = normalized_image
                
                # Update metadata
                meta_key = f"{key}_meta_dict"
                if meta_key not in data:
                    data[meta_key] = {}
                data[meta_key]["zscore_normalized"] = True
                
            except Exception as e:
                print(f"Error applying Z-score normalization to {key}: {e}")
                if not self.allow_missing_keys:
                    raise
        
        return data 