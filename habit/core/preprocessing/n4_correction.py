from typing import Dict, Any, Optional, Union, List, Tuple, Sequence
import torch
import numpy as np
import SimpleITK as sitk
from .base_preprocessor import BasePreprocessor
from .preprocessor_factory import PreprocessorFactory
from ...utils.image_converter import ImageConverter

@PreprocessorFactory.register("n4_correction")
class N4BiasFieldCorrection(BasePreprocessor):
    """Apply N4 bias field correction to images using SimpleITK.
    
    This preprocessor applies N4 bias field correction to correct for intensity
    inhomogeneity in medical images.
    """
    
    def __init__(
        self,
        keys: Union[str, List[str]],
        mask_keys: Optional[Union[str, List[str]]] = None,
        num_fitting_levels: int = 4,
        num_iterations: List[int] = None,
        convergence_threshold: float = 0.001,
        allow_missing_keys: bool = False,
        **kwargs
    ):
        """Initialize the N4 bias field correction preprocessor.
        
        Args:
            keys (Union[str, List[str]]): Keys of the images to be corrected.
            mask_keys (Optional[Union[str, List[str]]]): Keys of the masks to use for correction.
                If None, no mask will be used.
            num_fitting_levels (int): Number of fitting levels for the bias field correction.
            num_iterations (List[int]): Number of iterations at each fitting level.
                If None, will use [50] * num_fitting_levels.
            convergence_threshold (float): Convergence threshold for the correction.
            allow_missing_keys (bool): If True, allows missing keys in the input data.
            **kwargs: Additional parameters for N4 correction.
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
        
        # Set N4 parameters
        self.num_fitting_levels = num_fitting_levels
        self.num_iterations = num_iterations if num_iterations is not None else [50] * num_fitting_levels
        self.convergence_threshold = convergence_threshold
        
    def _apply_n4_correction(self, 
                           sitk_image: sitk.Image,
                           sitk_mask: Optional[sitk.Image] = None) -> sitk.Image:
        """Apply N4 bias field correction to a SimpleITK image.
        
        Args:
            sitk_image (sitk.Image): Input SimpleITK image to correct
            sitk_mask (Optional[sitk.Image]): Optional mask for the correction
            
        Returns:
            sitk.Image: Corrected SimpleITK image
        """
        # Create N4 bias field correction filter
        n4_filter = sitk.N4BiasFieldCorrectionImageFilter()
        
        # Set parameters
        n4_filter.SetMaximumNumberOfIterations(self.num_iterations)
        n4_filter.SetConvergenceThreshold(self.convergence_threshold)
        n4_filter.SetNumberOfFittingLevels(self.num_fitting_levels)
        
        # Apply correction
        if sitk_mask is not None:
            corrected_image = n4_filter.Execute(sitk_image, sitk_mask)
        else:
            corrected_image = n4_filter.Execute(sitk_image)
            
        return corrected_image
        
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply N4 bias field correction to the specified images.
        
        Args:
            data (Dict[str, Any]): Input data dictionary containing SimpleITK image objects.
            
        Returns:
            Dict[str, Any]: Data dictionary with corrected images.
        """
        print("Applying N4 bias field correction...")
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
                mask_key = f"mask_{key}"
                if mask_key in data:
                    sitk_mask = data[mask_key]
                    if not isinstance(sitk_mask, sitk.Image):
                        print(f"Warning: {mask_key} is not a SimpleITK Image object. Using no mask.")
                        sitk_mask = None
            
            try:
                # Apply N4 correction
                corrected_image = self._apply_n4_correction(sitk_image, sitk_mask)
                
                # Store the corrected image
                data[key] = corrected_image
                
                # Update metadata
                meta_key = f"{key}_meta_dict"
                if meta_key not in data:
                    data[meta_key] = {}
                data[meta_key]["n4_corrected"] = True
                
            except Exception as e:
                print(f"Error applying N4 correction to {key}: {e}")
                if not self.allow_missing_keys:
                    raise
        
        return data 