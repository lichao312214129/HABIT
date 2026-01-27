from typing import Dict, Any, Optional, Sequence, Union, List, Tuple
import numpy as np
import SimpleITK as sitk
import os
import logging
from .base_preprocessor import BasePreprocessor
from .preprocessor_factory import PreprocessorFactory
from ...utils.log_utils import get_module_logger

# Get module logger
logger = get_module_logger(__name__)

@PreprocessorFactory.register("resample")
class ResamplePreprocessor(BasePreprocessor):
    """Resample images to a target spacing using SimpleITK.
    
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
        **kwargs
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
            **kwargs: Additional parameters for resampling.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        
        # Convert single key to list
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        
        # Separate image and mask keys
        self.img_keys = self.keys
        self.mask_keys = [f"mask_{key}" for key in self.keys]
        
        # Get parameters from kwargs or use defaults
        self.target_spacing = kwargs.pop('target_spacing', target_spacing)
        self.img_mode = kwargs.pop('img_mode', img_mode)
        self.padding_mode = kwargs.pop('padding_mode', padding_mode)
        self.align_corners = kwargs.pop('align_corners', align_corners)
        
        # Map interpolation modes to SimpleITK interpolator objects
        self.interp_map = {
            "nearest": sitk.sitkNearestNeighbor,
            "linear": sitk.sitkLinear,
            "bilinear": sitk.sitkLinear,
            "bspline": sitk.sitkBSpline,
            "bicubic": sitk.sitkBSpline,
            "gaussian": sitk.sitkGaussian,
            "lanczos": sitk.sitkLanczosWindowedSinc,
            "hamming": sitk.sitkHammingWindowedSinc,
            "cosine": sitk.sitkCosineWindowedSinc,
            "welch": sitk.sitkWelchWindowedSinc,
            "blackman": sitk.sitkBlackmanWindowedSinc
        }
        
        # Default to linear if mode not found
        self.img_interp = self.interp_map.get(self.img_mode, sitk.sitkLinear)
        
    def _resample_image(self, 
                       sitk_image: sitk.Image, 
                       target_spacing: Sequence[float],
                       interpolator,
                       subj: Optional[str] = None,
                       key: Optional[str] = None) -> Tuple[np.ndarray, Sequence[float]]:
        """Resample a SimpleITK image.
        
        Args:
            sitk_image (sitk.Image): SimpleITK image object to resample
            target_spacing (Sequence[float]): Target spacing to resample to
            interpolator: SimpleITK interpolator object (e.g., sitk.sitkLinear)
            subj (Optional[str]): Subject identifier for logging
            key (Optional[str]): Image key for logging
            
        Returns:
            Tuple[np.ndarray, Sequence[float]]: 
                - Resampled array in original format
                - Original spacing of the image
        """
        subj_info = f"[{subj}] " if subj else ""
        key_info = f"({key}) " if key else ""
        
        # Get original spacing from the image
        original_spacing = sitk_image.GetSpacing()
        
        # Get image size
        size = sitk_image.GetSize()
        
        logger.debug(f"{subj_info}{key_info}Original spacing: {original_spacing}, size: {size}")
        
        # Calculate the new size after resampling
        zoom_factor = [orig_sz / target_sz for orig_sz, target_sz in zip(original_spacing, target_spacing)]
        new_size = [int(round(sz * factor)) for sz, factor in zip(size, zoom_factor)]
        
        logger.debug(f"{subj_info}{key_info}Target spacing: {target_spacing}, new size: {new_size}")
        
        # Create reference image with target spacing
        reference_image = sitk.Image(new_size, sitk_image.GetPixelID())
        reference_image.SetSpacing(target_spacing)
        reference_image.SetOrigin(sitk_image.GetOrigin())
        reference_image.SetDirection(sitk_image.GetDirection())
        
        # Perform resampling
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)
        resampler.SetInterpolator(interpolator)
        resampled_sitk = resampler.Execute(sitk_image)
        
        return resampled_sitk, original_spacing
        
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Resample the images and masks to the target spacing.
        
        Args:
            data (Dict[str, Any]): Input data dictionary containing SimpleITK image objects and metadata.
                The values for image and mask keys should be SimpleITK Image objects.
            
        Returns:
            Dict[str, Any]: Data dictionary with resampled images and masks.
        """
        self._check_keys(data)
        
        subj = data.get('subj', 'unknown')
        logger.debug(f"[{subj}] Resampling to {self.target_spacing}")
        
        # Process images
        for key in self.img_keys:
            meta_key = f"{key}_meta_dict"
            
            # Get SimpleITK image from data
            sitk_image = data[key]
            
            # Ensure we have a SimpleITK image object
            if not isinstance(sitk_image, sitk.Image):
                logger.warning(f"[{subj}] Warning: {key} is not a SimpleITK Image object. Skipping.")
                continue
            
            # Perform resampling with SimpleITK
            resampled_img, original_spacing = self._resample_image(
                sitk_image=sitk_image,
                target_spacing=self.target_spacing,
                interpolator=self.img_interp,
                subj=subj,
                key=key
            )

            # Store the resampled array in the data
            data[key] = resampled_img
                    
        # Process masks with nearest neighbor interpolation
        for mask_key in self.mask_keys:
            if mask_key not in data:
                continue
                
            # Get SimpleITK mask from data
            sitk_mask = data[mask_key]
            
            # Ensure we have a SimpleITK image object
            if not isinstance(sitk_mask, sitk.Image):
                logger.warning(f"[{subj}] Warning: {mask_key} is not a SimpleITK Image object. Skipping.")
                continue
            
            # Perform resampling with SimpleITK using nearest neighbor for masks
            resampled_img, original_spacing = self._resample_image(
                sitk_image=sitk_mask,
                target_spacing=self.target_spacing,
                interpolator=sitk.sitkNearestNeighbor,
                subj=subj,
                key=mask_key
            )
            
            # Store the resampled array in the data
            data[mask_key] = resampled_img
            
        return data