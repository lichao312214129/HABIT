from typing import Dict, Any, Optional, Union, List, Tuple
import torch
import numpy as np
import ants
import SimpleITK as sitk
from .base_preprocessor import BasePreprocessor
from .preprocessor_factory import PreprocessorFactory

@PreprocessorFactory.register("registration")
class RegistrationPreprocessor(BasePreprocessor):
    """Image registration preprocessor using ANTs.
    
    This preprocessor performs image registration using ANTs.
    Compatible with MONAI's LoadImaged and EnsureChannelFirstd transforms.
    Automatically handles masks and labels appropriately.
    """
    
    def __init__(
        self,
        keys: Union[str, List[str]],
        fixed_image: str,
        moving_images: Optional[Union[str, List[str]]] = None,
        use_mask: bool = False,
        type_of_transform: str = "SyN",
        allow_missing_keys: bool = False,
        **kwargs
    ):
        """Initialize the registration preprocessor.
        
        Args:
            keys (Union[str, List[str]]): Keys of the moving images to be transformed.
            fixed_image (str): Key of the fixed image.
            moving_images (Optional[Union[str, List[str]]]): Keys of the moving images. If None, uses keys parameter.
            use_mask (bool): Whether to use mask for registration. If True, expects a 'mask' key in data.
            type_of_transform (str): Type of transform to use. Defaults to "SyN".
            allow_missing_keys (bool): If True, allows missing keys in the input data.
            **kwargs: Additional parameters to pass to ANTs registration function.
                Supported parameters include:
                - optimizer_type: Type of optimizer (e.g., "gradient_descent")
                - optimizer_params: Dictionary of optimizer parameters
                - metric_type: Type of similarity metric
                - metric_params: Dictionary of metric parameters
                - interpolator_type: Type of interpolator
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.fixed_key = fixed_image
        self.use_mask = kwargs.pop('use_mask', use_mask)
        self.type_of_transform = kwargs.pop('type_of_transform', type_of_transform)
        self.moving_images = moving_images if moving_images is not None else keys
        if isinstance(self.moving_images, str):
            self.moving_images = [self.moving_images]
            
        # Get registration parameters from kwargs
        self.optimizer_type = kwargs.pop('optimizer_type', 'gradient_descent')
        self.optimizer_params = kwargs.pop('optimizer_params', {
            'learning_rate': 0.01,
            'number_of_iterations': 100
        })
        self.metric_type = kwargs.pop('metric_type', 'mutual_information')
        self.metric_params = kwargs.pop('metric_params', {
            'number_of_histogram_bins': 50
        })
        self.interpolator_type = kwargs.pop('interpolator_type', 'linear')
        self.kwargs = kwargs  # Store any remaining kwargs
        
    def _is_mask_or_label(self, key: str) -> bool:
        """Check if the key represents a mask or label.
        
        Args:
            key (str): The key to check.
            
        Returns:
            bool: True if the key represents a mask or label, False otherwise.
        """
        return 'mask' in key.lower() or 'label' in key.lower()
        
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform image registration.
        
        Args:
            data (Dict[str, Any]): Input data dictionary containing image and metadata from MONAI.
            
        Returns:
            Dict[str, Any]: Data dictionary with registered images.
        """
        self._check_keys(data)

        # to lower case compatible with MONAI FIXME
        self.fixed_key = self.fixed_key.lower()
        self.moving_images = [key.lower() for key in self.moving_images]
        # add mask_key to moving_images if it is not None
        if self.moving_images is not None:
            self.moving_images.append('mask')
        
        # Get the fixed image
        if self.fixed_key not in data:
            raise KeyError(f"Fixed image key {self.fixed_key} not found")
            
        fixed_meta_key = f"{self.fixed_key}_meta_dict"
        if fixed_meta_key not in data:
            raise ValueError(f"Metadata for fixed image {self.fixed_key} not found. Ensure LoadImaged is used before RegistrationPreprocessor.")
            
        # Convert fixed image to ANTs format
        # Read image directly using ANTs
        fixed_image = ants.image_read(data[fixed_meta_key]["filename_or_obj"])

        # Get the fixed mask if provided
        fixed_mask = None
        if self.use_mask and 'mask' in data:
            mask_meta_key = "mask_meta_dict"
            if mask_meta_key not in data:
                raise ValueError("Metadata for mask not found. Ensure LoadImaged is used before RegistrationPreprocessor.")
            # Read mask directly using ANTs
            fixed_mask = ants.image_read(data[mask_meta_key]["filename_or_obj"])
            
        # Process each moving image
        for key in self.moving_images:
            # Skip if key is missing and allowed to be missing
            if key not in data:
                if not self.allow_missing_keys:
                    raise KeyError(f"Key '{key}' is missing in the data dictionary.")
                continue
                
            # Get the moving image metadata
            moving_meta_key = f"{key}_meta_dict"
            if moving_meta_key not in data:
                raise ValueError(f"Metadata for moving image {key} not found. Ensure LoadImaged is used before RegistrationPreprocessor.")
                
            # Read moving image directly using ANTs
            moving_image = ants.image_read(data[moving_meta_key]["filename_or_obj"])
            
            # Prepare registration parameters
            reg_params = {
                'fixed': fixed_image,
                'moving': moving_image,
                'type_of_transform': self.type_of_transform,
                'mask': fixed_mask,
                'optimizer': self.optimizer_type,
                'metric': self.metric_type,
                'interpolator': self.interpolator_type,
                **self.optimizer_params,
                **self.metric_params,
                **self.kwargs
            }
            
            # Perform registration
            reg_result = ants.registration(**reg_params)
            
            # Get the warped image
            warped_image = reg_result["warpedmovout"]
            
            # Convert warped image to numpy array and then to torch tensor
            warped_array = warped_image.numpy()
            # transposed to [Z, Y, X]
            warped_array = np.transpose(warped_array, (2, 1, 0))
            # Add channel dimension and convert to torch tensor
            warped_tensor = torch.from_numpy(warped_array[np.newaxis, ...])
            
            # Update the data dictionary
            data[key] = warped_tensor
            
            # Update metadata
            data[moving_meta_key]["affine"] = warped_image.direction
            data[moving_meta_key]["origin"] = warped_image.origin
            data[moving_meta_key]["spacing"] = warped_image.spacing
            data[moving_meta_key]["registered"] = True
            data[moving_meta_key]["transform"] = reg_result["fwdtransforms"]
            
        return data 