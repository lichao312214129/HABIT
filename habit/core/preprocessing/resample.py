from typing import Dict, Any, Optional, Sequence, Union, List, Tuple
import numpy as np
from monai.transforms import Spacingd
from .base_preprocessor import BasePreprocessor
from .preprocessor_factory import PreprocessorFactory

@PreprocessorFactory.register("resample")
class ResamplePreprocessor(BasePreprocessor):
    """Resample images to a target spacing using MONAI's Spacingd transform.
    
    This preprocessor resamples images and masks to a specified target spacing.
    Images and masks are processed separately with different interpolation modes:
    - Images: use specified interpolation mode (default: bilinear)
    - Masks: use nearest neighbor interpolation
    """
    
    def __init__(
        self,
        keys: Union[str, List[str]],
        target_spacing: Sequence[float],
        img_mode: str = "bilinear",
        padding_mode: str = "border",
        align_corners: bool = False,
        allow_missing_keys: bool = False,
    ):
        """Initialize the resample preprocessor.
        
        Args:
            keys (Union[str, List[str]]): Keys of the corresponding items to be transformed.
                Should include both image and mask keys.
            target_spacing (Sequence[float]): Target spacing to resample to, e.g., (2.0, 2.0, 2.0).
            img_mode (str): Interpolation mode for image data. Defaults to "bilinear".
            padding_mode (str): Padding mode for out-of-bound values. Defaults to "border".
            align_corners (bool): Whether to align corners. Defaults to False.
            allow_missing_keys (bool): If True, allows missing keys in the input data.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        
        # Convert single key to list
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        
        # Separate image and mask keys
        self.mask_keys = [key for key in keys if 'mask' in key.lower()]
        self.img_keys = [key for key in keys if 'mask' not in key.lower()]
        
        # Initialize separate transforms for images and masks
        if self.img_keys:
            self.img_transform = Spacingd(
                keys=self.img_keys,
                pixdim=target_spacing,
                mode=img_mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
                allow_missing_keys=allow_missing_keys
            )
            
        if self.mask_keys:
            self.mask_transform = Spacingd(
                keys=self.mask_keys,
                pixdim=target_spacing,
                mode="nearest",
                padding_mode=padding_mode,
                align_corners=align_corners,
                allow_missing_keys=allow_missing_keys
            )
        
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Resample the images and masks to the target spacing.
        
        Args:
            data (Dict[str, Any]): Input data dictionary containing images, masks and metadata.
            
        Returns:
            Dict[str, Any]: Data dictionary with resampled images and masks.
        """
        self._check_keys(data)
        result = data.copy()
        
        # Process images
        if self.img_keys:
            result.update(self.img_transform(data))
            
        # Process masks
        if self.mask_keys:
            result.update(self.mask_transform(data))
            
        return result 